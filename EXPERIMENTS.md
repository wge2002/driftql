# QT-DriftQL: Q-Tilted Drift Q-Learning — Experiment Guide

## 1. Method in one paragraph

Baseline DriftQL is additive: `L_actor = alpha * L_drift + L_Q`, where `L_Q = -Q(s, f(s, eps))`
is an unconstrained gradient force. When the critic is wrong, this force drags particles off the
data manifold (OOD actions → critic overestimation → collapse). **QT-DriftQL removes the additive
Q force** (`q_loss_coef = 0` by default) and lets the critic act only through the *target measure*
of the drift field: the stop-gradient particles themselves are promoted to candidate attractor
atoms, weighted by `softmax(Q/lambda + gate)`, with the dataset action keeping a guaranteed prior
mass `rho`. Q never produces a direction in action space — it only decides **which in-support atom
to pull toward and how strongly**. A wrong critic can at worst pick the wrong in-support mode; it
cannot push the actor OOD.

Sanity reductions:

| Setting | Recovers |
|---|---|
| `self_atoms=False, q_loss_coef=1.0` | exactly baseline DriftQL |
| `tilt_lambda → ∞` (or during warmup) | drift-BC toward {a+, gated-uniform atoms} |
| `pos_anchor_floor=1.0` | baseline drift attraction (data atom only) |

## 2. New hyperparameters (`agents/qtilted_driftql.py`)

| Flag | Default | Meaning |
|---|---|---|
| `self_atoms` | True | Promote sg(particles) to candidate attractor atoms |
| `tilt_lambda` | 2.0 | Tilt temperature in units of batch std of atom Q. Smaller = sharper value-seeking |
| `tilt_q_agg` | mean | Ensemble aggregation (target critic) for atom Q |
| `tilt_warmup_steps` | 50000 | Linear ramp 0→1 of the tilt; protects against random early critic |
| `pos_anchor_floor` (rho) | 0.3 | Guaranteed prior mass on the dataset atom — BC anchor, anti self-lock |
| `gate_temp_mult` | 4.0 | Trust gate sigma = mult × drift_temp; far-from-data particles can't become attractors |
| `q_loss_coef` | 0.0 | Residual additive Q force (1.0 = DriftQL-style, for the additive ablation) |

## 3. Run layout & logs

Every run goes through `scripts/run_one.sh <TAG> <ENV> <SEED> [flags]` and writes:

```
exp/
  <TAG>/                          # e.g. base_driftql, qt_lam2, qt_lam2_ngen16, qt_lam2_qloss1
    <ENV_NAME>/
      seed<SEED>/
        stdout_<timestamp>.log    # full console output
        run_<timestamp>/
          flags.json              # complete config of the run
          train.csv               # every 5k steps: all training diagnostics below
          eval.csv                # every 100k steps: evaluation/success etc.
```

wandb runs in offline mode (local `./wandb/`, sync later if wanted); the CSVs are authoritative.

To package everything for download (excludes checkpoints):

```bash
bash scripts/collect_results.sh     # -> results_<timestamp>.tar.gz
```

## 4. Diagnostics dictionary (train.csv, prefix `training/actor/` and `training/critic/`)

**Health / stability** (same reading as our QAM-era heuristics):

- `critic/q_mean` — watch for collapse to the reward floor; sudden dives = critic collapse.
- `critic/q_ensemble_std` — ensemble disagreement; shrinking together with a q_mean dive = joint collapse.
- `critic/target_next_q_mean` — whether bootstrap targets are actually rising.
- `actor/target_step_rms`, `target_step_p99`, `target_step_max` — drift target step size; spikes precede instability.
- `actor/support_dist_mean`, `support_dist_min` — scaled distance of particles to the dataset action; rising trend = leaving support.

**Tilt mechanics** (the new signal — this is what tells us if the mechanism works):

- `tilt_ramp` — warmup ramp (0→1 by `tilt_warmup_steps`).
- `tilt_q_gap` — mean(max atom Q − dataset-action Q). **Positive and stable = the policy generates
  in-support candidates the critic prefers over the data → improvement is happening through the tilt.**
- `tilt_atom_q_data`, `tilt_atom_q_gen_mean`, `tilt_atom_q_gen_max` — raw atom Q levels.
- `tilt_u_eff_atoms` — effective number of selected atoms (exp of entropy). ≈1 = collapsed onto a single
  atom (risky if early), ≈Ngen = no selection (tilt too weak / lambda too big).
- `tilt_u_max` — max atom prior weight.
- `tilt_gate_dist_mean` — mean scaled distance of atoms to data action (gate input).
- `attract_mass_data` — total attraction weight on the dataset atom. Should start high (warmup) and
  settle; **if it goes ≈0 while support_dist rises, that's self-lock — increase rho or gate.**
- `attract_w_entropy` — entropy of per-particle attraction weights.

**Loss terms**: `drift_loss`, `drift_loss_weighted`, `drift_norm`, `drift_attract_norm`,
`drift_repel_norm`, `q_loss`, `q`.

## 5. Phase-1 experiment matrix (`scripts/phase1_commands.txt`)

Two envs sharing the same tags, 4 seeds each:

| Tag | Agent | What it tests |
|---|---|---|
| `base_driftql` | driftql | reproduction of the paper baseline |
| `qt_lam2` | qtilted | main setting (lambda=2) |
| `qt_lam05` | qtilted | sharp tilt |
| `qt_lam5` | qtilted | mild tilt |
| `qt_lam2_ngen16` | qtilted | (cube only) particle/atom coverage |
| `qt_lam2_qloss1` | qtilted | tilt + additive Q force — complement or replacement? |

Envs: `cube-double-play-singletask-task1-v0` (DriftQL's weak spot, paper: 49±1, FQL: 61) and
`antmaze-large-navigate-singletask-task1-v0` (DriftQL's strength, paper: 95±2 — regression check).

Launch on the server:

```bash
git clone git@github.com:wge2002/driftql.git && cd driftql
pip install -r requirements.txt        # same env as FQL/DriftQL; see README for mujoco210 setup
python jax_check.py                    # verify CudaDevice
GPUS="0,1,2,3" JOBS_PER_GPU=1 bash scripts/launch_all.sh scripts/phase1_commands.txt
# ... when done:
bash scripts/collect_results.sh
```

A single run (smoke test, ~5 min to first eval at step 1):

```bash
bash scripts/run_one.sh smoke cube-double-play-singletask-task1-v0 0 \
  --agent=agents/qtilted_driftql.py --agent.drift_temp=0.2 --agent.drift_ngen=4 \
  --agent.noise_dim=8 --agent.alpha=100 --agent.q_agg=mean --agent.kernel=gaussian \
  --offline_steps=2000 --eval_interval=1000 --eval_episodes=5 --log_interval=500
```

## 6. Decision criteria after Phase 1

1. **Mechanism check** (before looking at success): `tilt_q_gap` > 0 sustained, `attract_mass_data`
   stable above ~rho, no `critic/q_mean` collapse, `target_step_rms` in the same range as baseline.
2. **cube-double**: any qt tag beating `base_driftql` final success by a clear multi-seed margin
   validates the thesis (value-seeking via tilted target measure beats additive force where mode
   selection matters).
3. **antmaze-large**: qt within noise of baseline = no regression on DriftQL's home turf.
4. `qt_lam2_qloss1` vs `qt_lam2`: if pure tilt ≥ tilt+force, the "replace, don't add" story holds.

Phase 2 (after results): OGBench corruption protocol (p=0.2/0.3/0.4 random actions — expect the
gate+tilt to actively filter corrupted atoms), full OGBench suite, then D4RL.
