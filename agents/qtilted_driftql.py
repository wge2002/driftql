"""Q-Tilted DriftQL (QT-DriftQL).

Core idea
---------
Baseline DriftQL is additive ("A+B"): L_actor = alpha * L_drift + L_Q, where
L_Q = -Q(s, f(s, eps)) supplies an unconstrained gradient force that can drag
particles off the data manifold when the critic is wrong (the classic OOD /
critic-collapse failure mode of offline RL).

QT-DriftQL removes the additive Q force (by default, q_loss_coef = 0) and lets
the critic act ONLY through the *target measure* of the drift field:

    nu_lambda(. | s)  ∝  rho * delta_{a+}
                        + (1 - rho) * softmax_j[ Q(s, y_j)/lambda_eff + gate_j ] * delta_{y_j}

where the atoms y_j are the (stop-gradient) generated particles themselves,
promoted to candidate attractors. The attraction term becomes a kernel-softmax
pull toward this Q-tilted atom set instead of toward the single dataset action.
Repulsion is unchanged.

Consequences:
  * Q never produces a gradient direction in action space; it only chooses
    WHICH in-support atom to pull toward and HOW strongly. A wrong critic can
    at worst pick the wrong in-support mode; it cannot push the actor OOD.
  * rho (pos_anchor_floor) guarantees the dataset action always retains at
    least rho of the attraction prior mass -> unconditional BC anchor, guards
    against self-attractor lock-in.
  * gate_j down-weights particle atoms far from the dataset action, so only
    near-support particles can become attractors.
  * lambda is measured in units of the batch std of atom Q-values, so the
    tilt strength is scale-invariant across environments.
  * A warmup ramp keeps the tilt off while the critic is still random.

Recovering baselines:
  * self_atoms=False, q_loss_coef=1.0  ->  exactly baseline DriftQL.
  * tilt_lambda -> inf (or ramp=0)     ->  drift-BC toward {a+, gated uniform atoms}.
"""

import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import ActorVectorField, Value


def _l2_norm(x, axis=-1, eps=1e-12):
    return jnp.sqrt(jnp.sum(x * x, axis=axis) + eps)


def _pairwise_l2(a, b, eps=1e-12):
    """Pairwise L2: a [N,F], b [M,F] -> [N,M]."""
    a2 = jnp.sum(a * a, axis=-1, keepdims=True)
    b2 = jnp.sum(b * b, axis=-1, keepdims=True)
    dist2 = a2 + b2.T - 2.0 * (a @ b.T)
    return jnp.sqrt(jnp.maximum(dist2, 0.0) + eps)


def _cfg_get(cfg, key, default):
    try:
        return cfg[key]
    except KeyError:
        return default


def _resolve_noise_dim(cfg, action_dim):
    noise_dim = int(_cfg_get(cfg, 'noise_dim', action_dim))
    if noise_dim <= 0:
        noise_dim = int(action_dim)
    return noise_dim


def _kernel_logits(dist, temp, kernel, dim_factor):
    """Kernel logits from raw pairwise distances."""
    if kernel == 'gaussian':
        scaled = dist / dim_factor
        return -(scaled ** 2) / (2.0 * temp * temp)
    # Laplace
    return -(dist / dim_factor) / temp


def _compute_tilted_drift(
        gen_a: jnp.ndarray,   # [Ngen, A] generated actions (particles)
        atoms: jnp.ndarray,   # [M, A] attractor atoms (atom 0 = dataset action)
        log_p: jnp.ndarray,   # [M] log prior over atoms (tilted + gated + anchored)
        temp: float,
        kernel: str = 'laplace',
        dim_scale: bool = True,
        mask_self_atom: bool = True,
        eps: float = 1e-12,
):
    """Drift field for one state with a tilted target measure.

    Attraction weights:  W_ij = softmax_j( kernel_logit(x_i, y_j) + log p_j )
    i.e. kernel likelihood x tilted atom prior, normalized per particle.

    Returns:
        V:          [Ngen, A] total drift = V_attract - V_repel
        V_attract:  [Ngen, A]
        V_repel:    [Ngen, A]
        W_pos:      [Ngen, M] attraction weights (for diagnostics)
    """
    Ngen, action_dim = gen_a.shape

    dist_pos = _pairwise_l2(gen_a, atoms, eps=eps)   # [Ngen, M]
    dist_neg = _pairwise_l2(gen_a, gen_a, eps=eps)   # [Ngen, Ngen]

    idx = jnp.arange(Ngen)
    # Self-mask for repulsion.
    dist_neg = dist_neg.at[idx, idx].set(1e6)
    # Self-mask for attraction: particle i must not be attracted to its own
    # stop-gradient copy (atom index i+1), otherwise attraction mass is wasted
    # on a zero displacement.
    if mask_self_atom and atoms.shape[0] == Ngen + 1:
        dist_pos = dist_pos.at[idx, idx + 1].set(1e6)

    if dim_scale:
        dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
    else:
        dim_factor = 1.0

    logit_pos = _kernel_logits(dist_pos, temp, kernel, dim_factor)  # [Ngen, M]
    logit_neg = _kernel_logits(dist_neg, temp, kernel, dim_factor)  # [Ngen, Ngen]

    W_pos = jax.nn.softmax(logit_pos + log_p[None, :], axis=-1)  # [Ngen, M]
    W_neg = jax.nn.softmax(logit_neg, axis=-1)                   # [Ngen, Ngen]

    disp_to_pos = atoms[None, :, :] - gen_a[:, None, :]   # [Ngen, M, A]
    disp_to_neg = gen_a[None, :, :] - gen_a[:, None, :]   # [Ngen, Ngen, A]

    V_attract = jnp.sum(W_pos[:, :, None] * disp_to_pos, axis=1)  # [Ngen, A]
    V_repel = jnp.sum(W_neg[:, :, None] * disp_to_neg, axis=1)    # [Ngen, A]

    V = V_attract - V_repel
    return V, V_attract, V_repel, W_pos


class QTiltedDriftQLAgent(flax.struct.PyTreeNode):
    """DriftQL where the critic tilts the drift field's target measure."""

    rng: Any
    network: Any
    step: Any
    config: Any = nonpytree_field()

    # ------------------------------------------------------------------ #
    # Critic: standard Bellman regression (unchanged from DriftQL),
    # with extra diagnostics logged.
    # ------------------------------------------------------------------ #
    def critic_loss(self, batch, grad_params, rng):
        rng, sample_rng = jax.random.split(rng)
        next_actions = self.sample_actions(batch['next_observations'], seed=sample_rng)
        next_qs = self.network.select('target_critic')(batch['next_observations'], actions=next_actions)

        if self.config['q_agg'] == 'min':
            next_q = next_qs.min(axis=0)
        else:
            next_q = next_qs.mean(axis=0)

        target_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_q
        q = self.network.select('critic')(batch['observations'], actions=batch['actions'], params=grad_params)
        critic_loss = jnp.square(q - target_q).mean()

        return critic_loss, {
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
            'q_ensemble_std': jnp.std(q, axis=0).mean(),
            'target_next_q_mean': next_q.mean(),
            'target_q_mean': target_q.mean(),
        }

    # ------------------------------------------------------------------ #
    # Actor: Q-tilted drift loss.
    # ------------------------------------------------------------------ #
    def tilted_drift_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, sub_rng = jax.random.split(rng, 3)

        # ---- Config (all static under jit) ----
        kernel = str(_cfg_get(self.config, 'kernel', 'laplace'))
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        dim_scale = bool(_cfg_get(self.config, 'dim_scale', True))
        drift_normalize = bool(self.config['drift_normalize'])
        temp = float(self.config['drift_temp'])
        eta = float(self.config['drift_eta'])
        Ngen = int(self.config['drift_ngen'])
        eps = float(self.config['drift_eps'])

        self_atoms = bool(self.config['self_atoms'])
        tilt_lambda = float(self.config['tilt_lambda'])
        tilt_q_agg = str(self.config['tilt_q_agg'])
        tilt_warmup = max(int(self.config['tilt_warmup_steps']), 1)
        rho = float(self.config['pos_anchor_floor'])
        gate_temp_mult = float(self.config['gate_temp_mult'])

        # ---- Optional subsample for drift computation ----
        drift_bs = int(self.config['drift_batch_size'])
        if drift_bs < batch_size:
            idx = jax.random.choice(sub_rng, batch_size, (drift_bs,), replace=False)
            obs = batch['observations'][idx]
            pos_actions = batch['actions'][idx]
        else:
            obs = batch['observations']
            pos_actions = batch['actions']
            drift_bs = batch_size

        pos_a = jnp.clip(pos_actions, -1.0, 1.0)  # [B, A]

        # ---- Generate Ngen particles per state from the current policy ----
        noises = jax.random.normal(noise_rng, (drift_bs * Ngen, noise_dim))
        obs_rep = jnp.repeat(obs, repeats=Ngen, axis=0)
        gen_raw = self.network.select('actor_bc_drift')(obs_rep, noises, params=grad_params)
        gen_a = jnp.clip(gen_raw.reshape(drift_bs, Ngen, action_dim), -1.0, 1.0)

        if dim_scale:
            dim_factor = jnp.sqrt(jnp.asarray(action_dim, dtype=gen_a.dtype))
        else:
            dim_factor = 1.0

        # ---- Build the atom set and its tilted prior ----
        if self_atoms:
            atoms = jnp.concatenate(
                [pos_a[:, None, :], jax.lax.stop_gradient(gen_a)], axis=1
            )  # [B, M, A], M = 1 + Ngen
            M = Ngen + 1

            # Q-values of all atoms under the *target* critic (stop-grad).
            obs_atoms = jnp.repeat(obs, repeats=M, axis=0)
            atoms_flat = atoms.reshape(drift_bs * M, action_dim)
            qs_atoms = self.network.select('target_critic')(obs_atoms, actions=atoms_flat)
            if tilt_q_agg == 'min':
                q_atoms = qs_atoms.min(axis=0)
            else:
                q_atoms = qs_atoms.mean(axis=0)
            q_atoms = jax.lax.stop_gradient(q_atoms.reshape(drift_bs, M))  # [B, M]

            # Scale-invariant tilt: lambda is in units of the batch std of Q.
            q_mu = q_atoms.mean()
            q_sd = q_atoms.std() + 1e-6
            q_norm = (q_atoms - q_mu) / q_sd

            # Warmup ramp: no tilt while the critic is still random.
            ramp = jnp.clip(self.step.astype(jnp.float32) / float(tilt_warmup), 0.0, 1.0)
            tilt_logits = ramp * q_norm[:, 1:] / tilt_lambda  # [B, Ngen]

            # Trust gate: particle atoms far from the dataset action are
            # down-weighted (Gaussian in scaled distance).
            d_gate = _l2_norm(atoms[:, 1:, :] - pos_a[:, None, :], axis=-1) / dim_factor  # [B, Ngen]
            gate_sigma = gate_temp_mult * temp
            gate_logits = -(d_gate ** 2) / (2.0 * gate_sigma * gate_sigma)

            u = jax.nn.softmax(tilt_logits + gate_logits, axis=-1)  # [B, Ngen]
            p = jnp.concatenate(
                [jnp.full((drift_bs, 1), rho, dtype=u.dtype), (1.0 - rho) * u], axis=1
            )  # [B, M]
            log_p = jnp.log(p + 1e-12)
        else:
            # Single dataset atom: reduces to the baseline DriftQL attraction.
            atoms = pos_a[:, None, :]  # [B, 1, A]
            M = 1
            q_atoms = jnp.zeros((drift_bs, 1))
            q_norm = jnp.zeros((drift_bs, 1))
            u = jnp.ones((drift_bs, 1))
            d_gate = jnp.zeros((drift_bs, 1))
            ramp = jnp.zeros(())
            log_p = jnp.zeros((drift_bs, 1))

        # ---- Tilted drift field ----
        def per_item(gen_i, atoms_i, log_p_i):
            V_i, Va_i, Vr_i, W_i = _compute_tilted_drift(
                gen_a=gen_i,
                atoms=atoms_i,
                log_p=log_p_i,
                temp=temp,
                kernel=kernel,
                dim_scale=dim_scale,
                mask_self_atom=self_atoms,
                eps=eps,
            )
            if drift_normalize:
                act_dim = gen_i.shape[-1]
                raw_sq = jnp.mean(jnp.sum(V_i * V_i, axis=-1)) / act_dim
                lam = jax.lax.stop_gradient(jnp.sqrt(raw_sq + eps))
                V_i = V_i / lam
                Va_i = Va_i / lam
                Vr_i = Vr_i / lam
            return V_i, Va_i, Vr_i, W_i

        V, V_attract, V_repel, W_pos = jax.vmap(per_item)(gen_a, atoms, log_p)

        target = jax.lax.stop_gradient(jnp.clip(gen_a + eta * V, -1.0, 1.0))
        drift_loss = jnp.mean((gen_a - target) ** 2)

        # ---- Diagnostics ----
        step_norm = _l2_norm(eta * V, axis=-1)  # [B, Ngen]
        support_dist = _l2_norm(gen_a - pos_a[:, None, :], axis=-1) / dim_factor  # [B, Ngen]
        w_entropy = -jnp.sum(W_pos * jnp.log(W_pos + 1e-12), axis=-1)  # [B, Ngen]
        u_entropy = -jnp.sum(u * jnp.log(u + 1e-12), axis=-1)  # [B]

        info = {
            'drift_loss': drift_loss,
            'drift_norm': jnp.mean(_l2_norm(V, axis=-1)),
            'drift_attract_norm': jnp.mean(_l2_norm(V_attract, axis=-1)),
            'drift_repel_norm': jnp.mean(_l2_norm(V_repel, axis=-1)),
            'target_step_rms': jnp.sqrt(jnp.mean(step_norm ** 2)),
            'target_step_p99': jnp.percentile(step_norm, 99),
            'target_step_max': step_norm.max(),
            'support_dist_mean': support_dist.mean(),
            'support_dist_min': support_dist.min(axis=-1).mean(),
            # Tilt diagnostics
            'tilt_ramp': ramp,
            'tilt_atom_q_data': q_atoms[:, 0].mean(),
            'tilt_atom_q_gen_mean': q_atoms[:, 1:].mean() if self_atoms else 0.0,
            'tilt_atom_q_gen_max': q_atoms[:, 1:].max(axis=-1).mean() if self_atoms else 0.0,
            'tilt_q_gap': (q_atoms[:, 1:].max(axis=-1) - q_atoms[:, 0]).mean() if self_atoms else 0.0,
            'tilt_q_norm_spread': q_norm.std(),
            'tilt_u_entropy': u_entropy.mean(),
            'tilt_u_eff_atoms': jnp.exp(u_entropy).mean(),
            'tilt_u_max': u.max(axis=-1).mean(),
            'tilt_gate_dist_mean': d_gate.mean() if self_atoms else 0.0,
            'attract_mass_data': W_pos[:, :, 0].mean(),
            'attract_mass_atoms': (1.0 - W_pos[:, :, 0]).mean(),
            'attract_w_entropy': w_entropy.mean(),
        }
        return drift_loss, info

    def actor_loss(self, batch, grad_params, rng):
        batch_size, action_dim = batch['actions'].shape
        rng, noise_rng, drift_rng = jax.random.split(rng, 3)

        drift_loss, drift_info = self.tilted_drift_loss(batch, grad_params, drift_rng)

        q_loss_coef = float(self.config['q_loss_coef'])
        if q_loss_coef > 0.0:
            # Optional residual additive Q force (ablation; baseline parity).
            noise_dim = _resolve_noise_dim(self.config, action_dim)
            noises = jax.random.normal(noise_rng, (batch_size, noise_dim))
            actor_raw = self.network.select('actor_bc_drift')(batch['observations'], noises, params=grad_params)
            actor_raw = jnp.clip(actor_raw, -1.0, 1.0)
            qs = self.network.select('critic')(batch['observations'], actions=actor_raw)
            q_agg = _cfg_get(self.config, 'q_agg_actor', self.config['q_agg'])
            if q_agg == 'min':
                q = qs.min(axis=0)
            else:
                q = qs.mean(axis=0)
            q_loss = -q.mean()
            if self.config['normalize_q_loss']:
                lam_q = jax.lax.stop_gradient(1.0 / (jnp.abs(q).mean() + 1e-6))
                q_loss = lam_q * q_loss
            q_mean = q.mean()
        else:
            q_loss = 0.0
            q_mean = 0.0

        actor_loss = self.config['alpha'] * drift_loss + q_loss_coef * q_loss

        info = {
            'actor_loss': actor_loss,
            'drift_loss_weighted': drift_loss * self.config['alpha'],
            'q_loss': q_loss,
            'q': q_mean,
        }
        info.update(drift_info)
        return actor_loss, info

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        info = {}
        rng = rng if rng is not None else self.rng
        rng, actor_rng, critic_rng = jax.random.split(rng, 3)

        critic_loss, critic_info = self.critic_loss(batch, grad_params, critic_rng)
        for k, v in critic_info.items():
            info[f'critic/{k}'] = v

        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = critic_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        return self.replace(network=new_network, rng=new_rng, step=self.step + 1), info

    @jax.jit
    def sample_actions(self, observations, seed=None, temperature=1.0):
        action_seed, _ = jax.random.split(seed)
        noise_dim = _resolve_noise_dim(self.config, self.config['action_dim'])
        noises = jax.random.normal(
            action_seed,
            (
                *observations.shape[: -len(self.config['ob_dims'])],
                noise_dim,
            ),
        )
        raw_actions = self.network.select('actor_bc_drift')(observations, noises)
        actions = jnp.clip(raw_actions, -1.0, 1.0)
        actions = jnp.where(jnp.isnan(actions), 0.0, actions)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config):
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ob_dims = ex_observations.shape[1:]
        action_dim = ex_actions.shape[-1]
        noise_dim = _resolve_noise_dim(config, action_dim)
        ex_noises = jnp.zeros((*ex_actions.shape[:-1], noise_dim), dtype=ex_actions.dtype)

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['actor_bc_drift'] = encoder_module()

        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )

        actor_bc_drift_def = ActorVectorField(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            encoder=encoders.get('actor_bc_drift'),
        )

        network_info = dict(
            critic=(critic_def, (ex_observations, ex_actions)),
            target_critic=(copy.deepcopy(critic_def), (ex_observations, ex_actions)),
            actor_bc_drift=(actor_bc_drift_def, (ex_observations, ex_noises)),
        )

        if encoders.get('actor_bc_drift') is not None:
            network_info['actor_bc_drift_encoder'] = (
                encoders.get('actor_bc_drift'),
                (ex_observations,),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']

        config['ob_dims'] = ob_dims
        config['action_dim'] = action_dim
        config['noise_dim'] = noise_dim

        return cls(
            rng,
            network=network,
            step=jnp.zeros((), dtype=jnp.int32),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            agent_name='qtilted_driftql',
            ob_dims=ml_collections.config_dict.placeholder(list),
            action_dim=ml_collections.config_dict.placeholder(int),
            lr=3e-4,
            batch_size=256,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            noise_dim=0,  # defaults to action_dim when <= 0
            layer_norm=True,
            actor_layer_norm=False,
            discount=0.99,
            tau=0.005,  # Polyak target update rate
            q_agg='min',  # "min" or "mean" for critic target
            q_agg_actor='mean',  # "min" or "mean" for the (optional) actor Q-loss
            alpha=10.0,  # drift loss weight
            normalize_q_loss=False,
            # Drift field (same semantics as baseline DriftQL)
            kernel='laplace',  # "laplace" or "gaussian"
            drift_temp=0.2,  # kernel temperature
            dim_scale=True,  # divide distances by sqrt(d_a) before applying temp
            drift_ngen=32,  # number of generated samples per state
            drift_eta=1.0,  # drift step size eta
            drift_normalize=False,
            drift_batch_size=256,
            drift_eps=1e-12,
            # ---- Q-tilting (new) ----
            self_atoms=True,  # promote sg(particles) to candidate attractor atoms
            tilt_lambda=2.0,  # tilt temperature, in units of batch std of atom Q
            tilt_q_agg='mean',  # "mean" or "min" over target-critic ensemble for atom Q
            tilt_warmup_steps=50000,  # linear ramp of the tilt from 0 to 1
            pos_anchor_floor=0.3,  # rho: guaranteed prior mass on the dataset atom
            gate_temp_mult=4.0,  # trust-gate sigma = gate_temp_mult * drift_temp
            q_loss_coef=0.0,  # residual additive Q force (0 = pure tilted)
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
    return config
