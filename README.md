# ReOptGuard: Lagrangian RL for Reoptimization Governance

**Budget-aware dynamic reoptimization controller for on-demand delivery using PPO-based Lagrangian Reinforcement Learning**

[![Paper Status](https://img.shields.io/badge/Status-Under%20Review-blue)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

> Release status: **README-only public preview**. Source code, configs, data, and models remain private during peer review. Target decision: **2026-08-31**; planned full release (v1.1.0) with code and weights: **2026-09-01**.

## Overview

**ReOptGuard** addresses a critical operational challenge in on-demand delivery systems: *when and how aggressively should the dispatch system invoke expensive reoptimization?*

Instead of replacing the dispatch core or tuning solver hyperparameters, ReOptGuard learns a **centralized high-level controller** that outputs three interpretable parameters:
- **Frequency** (Pulse): How often to reoptimize
- **Scope** (ScopeCap): Max reassignments per reoptimization round  
- **Threshold** (ValueGate): Minimum cost savings required to proceed

This is formulated as a **Constrained Markov Decision Process (CMDP)** solved via **Lagrangian PPO** with adaptive multipliers for managing service-quality and budget constraints simultaneously.

### Key Results

On realistic city-scale scenarios (~5000 orders, 300 couriers, one-day episodes):

| Metric | ReOptGuard | Baseline | Improvement |
|--------|-----------|----------|-------------|
| **On-time rate** | 90.25% | ~90%–91% | Within <0.3pp (parity) |
| **Reopt calls** | 42 | 50–92 | **16–54% ↓** |
| **Reassigned orders** | 110 | 142–258 | **23–57% ↓** |
| **Budget cost** | -7% to -54% | Baseline | **↓ Better** |

**Zero-shot generalization** across load regimes (light/medium/heavy) and robustness to distribution shifts (demand spikes, traffic shocks, courier dropouts, tight windows) with ≥74% constraint satisfaction.

---

## Features

### ✅ Implemented

- **CMDP-based RL framework** with Lagrangian PPO for multi-constraint optimization
- **Monotone policy** for interpretable, structurally-consistent control (load/lateness → monotone response)
- **City-scale simulator** with dynamic VRPTW, ETA models, real-world dispatch scenarios
- **Modular baseline dispatch** (nearest neighbor, smart heuristics) + local reoptimization
- **Comprehensive metrics & analysis** (completion rate, lateness, reoptimization overhead, cost savings)
- **Transfer & robustness evaluation** across scenarios, distribution shifts, and extreme conditions

### 🎯 What This Is NOT

- **Not a replacement dispatch algorithm**: We keep the existing dispatch core (SmartScheduler + Hungarian reoptimization)
- **Not a pure ML approach**: We combine RL with structured policies to enforce interpretability and monotonicity
- **Not a complete reproduction kit**: Core training data and tuned checkpoints are kept private to prevent unauthorized reproduction

---

## Release Plan & Access

- Current public repo content: README only (this file). No code, configs, checkpoints, or data are included to avoid premature reproduction during review.
- What will be released after acceptance (planned v1.1.0, 2026-09-01):
  - Full source (`src/`), configs, training/eval scripts
  - Trained checkpoints for all benchmark scenarios
  - Reproducibility scripts for paper tables/figures, plus data splits and logs
- Why limited now: protect novelty during review while providing a transparent overview of the method and release intent.
- Detailed command guides (e.g., CLI usage, reproduction steps) live in internal guides (e.g., GITHUB_UPLOAD_GUIDE_CN.md, PAPER_STATUS.md) and will ship with the full release.

---

## Project Structure (planned for post-acceptance release)

```
reoptguard/
├── src/
│   ├── bdrc/                          # BDRC RL agent & training
│   │   ├── agent.py                   # Policy + value networks
│   │   ├── trainer.py                 # On-policy PPO training loop
│   ├── scheduling/                    # Dispatch baselines
│   │   ├── smart_scheduler.py
│   │   └── nearest_neighbor.py
│   ├── reopt/                         # Local reoptimization
│   │   ├── reoptimizer.py
│   │   └── hungarian_matcher.py
│   ├── simulation/                    # Discrete-event simulator
│   │   ├── environment.py
│   │   ├── order.py
│   │   └── courier.py
│   ├── metrics/                       # Performance measurement
│   │   └── collector.py
│   └── utils/                         # Helper functions
│
├── configs/
│   ├── default.yaml                   # Base config template
│   └── scenarios/                     # Scenario definitions (lite)
│
├── paper/
│   └── main.tex                       # Springer journal submission
│
├── docs/
│   ├── METHODOLOGY.md                 # Architecture & design choices
│   ├── EVALUATION_PROTOCOL.md         # How to run benchmarks
│   └── API_REFERENCE.md               # Code documentation
│
├── requirements.txt                   # Dependencies
├── setup.py                           # Package installation
└── README.md                          # This file
```

> Not included in this preview: `src/`, `configs/`, `data/`, `models/`, `results/`, or any scripts. These arrive with the v1.1.0 release after acceptance.

---

## Installation

> The commands below describe the planned workflow once code is released. They are provided now for clarity and will become executable after acceptance.

- Python 3.8+
- PyTorch 1.10+
- NumPy, SciPy, Pandas

### Quick Setup

```bash
# Clone repository
git clone https://github.com/Luvlove-svg/ReOptGuard.git
cd ReOptGuard

# Install dependencies
pip install -r requirements.txt

# Optional: Install package in editable mode
pip install -e .
```

### Verify Installation

```bash
python -c "from src.bdrc import BDRCAgent; print('✓ ReOptGuard ready')"
```

---

## Quick Start

> The following CLI examples become available with the post-acceptance code drop. They are provided here to show intended usage and reproducibility flow.

### 1) Train ReOptGuard (monotone controller)

```bash
python scripts/train_bdrc.py \
  --data_dir data/scenario_BJ_realistic_v2 \
  --policy_type monotone \
  --episodes 50 \
  --device cuda \
  --use_parallel --num_parallel_episodes 6 \
  --exp_name bdrc_monotone_main
# Checkpoints/logs → results/bdrc_checkpoints/scenario_BJ_realistic_v2/bdrc_monotone_main/
```

### 2) Evaluate RL checkpoint vs baselines

```bash
python scripts/eval_bdrc_vs_baselines.py \
  --mode rl_policy \
  --checkpoint results/bdrc_checkpoints/scenario_BJ_realistic_v2/bdrc_monotone_main/ckpt_ep_50.pt \
  --data_dir data/scenario_BJ_realistic_v2 \
  --episodes_per_seed 20 --seeds 1,2,3 \
  --disable_timeseries --use_parallel --num_processes 6
# Outputs → results/scenario_BJ_realistic_v2/rl_seeds_1-2-3_ep60_*.csv
```

### 3) Baseline-only smoke test

```bash
python scripts/eval_bdrc_vs_baselines.py \
  --mode baseline --baseline load_based \
  --data_dir data/scenario_BJ_realistic_v2 \
  --episodes 20 \
  --disable_timeseries --use_parallel --num_processes 6
# Outputs → results/scenario_BJ_realistic_v2/baseline_load_based_*.csv
```

> Notes
> - Required data layout under `data/scenario_BJ_realistic_v2/`: `orders.json`, `riders.json`, `restaurants.json`.
> - Checkpoint path follows `results/bdrc_checkpoints/<scenario>/<exp_name>/ckpt_ep_<N>.pt` (see `scripts/train_bdrc.py --save_dir`).
> - Flags `--use_parallel/--num_parallel_episodes` (train) and `--use_parallel/--num_processes` (eval) are optional; set to CPU-friendly values if GPU not available.

---

## Methodology

### Problem Formulation: CMDP

**State** s_t: System load (pending/in-progress orders), lateness metrics, dispatcher state

**Action** a_t: Three continuous parameters
- reopt_freq_hz ∈ [0, 1/60] Hz (≥60 s interval; default target ≈90 s)
- max_reassign_per_round ∈ [0, 200] orders
- min_cost_saving_threshold ∈ [0, 1e5] cost units (typical tuned range 0–200)

**Reward** r_t: Orders completed in current step (positive RL signal)

**Costs** (constraints):
- c1_t: Service quality (lateness risk, completion rate gap vs. target ≥90%)
- c2_t: Reoptimization budget (call frequency, reassignment volume vs. limits)

**CMDP Objective** (Lagrangian):
```
max_π E[R]  subject to  E[C1] ≤ ε₁, E[C2] ≤ ε₂
```

→ Solved via **Lagrangian PPO with adaptive multipliers** for real-time constraint tracking

### Key Design Choices

1. **Monotone Policy** (optional): Policy head ensures
   - System load ↑ ⟹ reopt frequency ↑, reassignment cap ↑
   - Lateness ↑ ⟹ reopt frequency ↑, reassignment cap ↑
   - Cost threshold ↓ monotonically
   
   This structural constraint improves interpretability and cross-scenario generalization.

2. **Modular Dispatch**: Existing dispatch core (SmartScheduler + Hungarian matcher) remains untouched, allowing plug-and-play integration with production systems.

3. **Realistic Scenarios**: Scenarios based on real Beijing delivery platform data (order intensity, time windows, courier shifts) but fully synthetic to enable reproducibility.

---

## Usage & API

### Training Custom Model

```python
from src.bdrc.trainer import BDRCTrainer
from src.simulation.environment import DeliveryEnvironment
import yaml

# Load config
with open('configs/default.yaml') as f:
    config = yaml.safe_load(f)

# Create environment
env = DeliveryEnvironment(config)

# Initialize trainer
trainer = BDRCTrainer(
    env=env,
    policy_type='monotone',  # or 'mlp'
    constraint_eps=[0.01, 0.05],  # ε1, ε2
    learning_rate=3e-4
)

# Train
trainer.train(num_episodes=100, eval_interval=20)

# Save & evaluate
trainer.save_checkpoint('models/trained_bdrc.pth')
metrics = trainer.evaluate(num_episodes=10)
print(metrics)
```

### Evaluation

```python
from src.cli import evaluate_policy

results = evaluate_policy(
    model_path='models/trained_bdrc.pth',
    scenarios=['scenario_BJ_realistic_v2', 'scenario_AMAZON_train_200'],
    num_runs=5
)
# Returns: DataFrame with completion_rate, lateness, reopt_calls, reassignments, etc.
```

See [API_REFERENCE.md](docs/API_REFERENCE.md) for full documentation.

---

## Experimental Evaluation

### Scenarios Included (Lite Version)

| Scenario | Orders | Couriers | Load | Purpose |
|----------|--------|----------|------|---------|
| `scenario_BJ_realistic_v2` | ~5000 | 300 | Medium | Main benchmark (paper Fig 1–3) |
| `scenario_AMAZON_train_200` | ~3000–7000 | 200–400 | Light/Medium/Heavy | Transfer robustness test |

**Full experimental suite** (all ablations, sensitivity analyses) is available in supplementary materials upon acceptance.

### Baseline Comparison

- **Baseline-A**: No reoptimization (pure nearest-neighbor dispatch)
- **Baseline-B**: Fixed reoptimization (interval=60s, threshold=50)
- **Baseline-C**: Greedy parameter sweep (best of 36-param grid)
- **ReOptGuard**: Trained BDRC (this work)

### Evaluation Metrics

```python
metrics = {
    'completion_rate': float,        # Orders completed on-time / total
    'avg_lateness': float,           # Minutes late (for late orders)
    'reopt_calls_per_episode': int,  # Total reoptimization invocations
    'reassigned_orders': int,        # Total orders reassigned
    'reassignment_rate': float,      # Reassignments / total orders
    'cost_savings': float,           # Total cost saved vs. baseline
    'constraint_c1_cost': float,     # Service quality constraint cost
    'constraint_c2_cost': float,     # Budget constraint cost
}
```

---

## Important Disclaimers

### ⚠️ Reproducibility Note

This repository provides the **framework, architecture, and methodology**. To maintain research integrity and prevent unauthorized reproduction:

- **NOT included**: Trained model weights, full dataset files, and hyperparameter tuning scripts
- **NOT included**: Production dispatch simulator specifics or proprietary ETA models
- **Included**: Core RL algorithm, policy architecture, metric definitions, and evaluation framework

Researchers interested in reproducing results should:
1. **Contact authors** for data access or generate your own synthetic scenarios
2. **Follow methodology** in paper (Section 3–4) and [METHODOLOGY.md](docs/METHODOLOGY.md)
3. **Cite this work** if building upon it

### 📋 Citation

If you use ReOptGuard in your research, please cite:

```bibtex
@article{liu2024reoptguard,
  title={ReOptGuard: Budget-Aware Dynamic Reoptimization Governance via PPO-based Lagrangian Reinforcement Learning for On-Demand Delivery},
  author={Liu, Hongqi and Jin, Kailin},
  journal={Applied Intelligence},
  year={2026},
  note={Under Review}
}
```

---

## Contributing

We welcome contributions to improve the framework:
- Bug reports and fixes
- New scenario definitions (synthetic, anonymized data only)
- Evaluation improvements
- Documentation enhancements

Please follow the existing code style (Black formatting, type hints) and include tests.

**Contributing Guidelines**: See [CONTRIBUTING.md](CONTRIBUTING.md) (to be added upon publication)

---

## License

MIT License - see [LICENSE](LICENSE) for details

---

## Authors

**Hongqi Liu** | Northeastern University at Qinhuangdao  
📧 [202315073@stu.neuq.edu.cn](mailto:202315073@stu.neuq.edu.cn)

**Kailin Jin** | Henan Normal University  
📧 [jkl528ae@163.com](mailto:jkl528ae@163.com)

---

## FAQ

**Q: Can I use this for production?**  
A: The framework is research-focused. Production deployment requires additional components: robust ETA models, fault tolerance, A/B testing infrastructure, and compliance with platform terms of service.

**Q: What if I don't have my own data?**  
A: Start with the included synthetic scenarios. The framework is designed to be data-agnostic; you can define custom scenarios in `configs/scenarios/`.

**Q: How long does training take?**  
A: ~2–4 hours on a single GPU (A100/V100) for 100 episodes. CPU-only training is supported but slower (~8–12 hours).

**Q: How do I adapt this to my delivery platform?**  
A: See [METHODOLOGY.md](docs/METHODOLOGY.md) for detailed implementation guidance and integration points.

---

## Changelog

### v1.0.0 (Initial Release)
- Core BDRC agent & PPO training
- Monotone policy implementation
- Benchmark scenarios & evaluation suite
- Documentation & quick-start guide

---

## Acknowledgments

This work benefited from discussions with colleagues at [Institutions], access to platform data from [Platform Partners], and valuable feedback from the delivery optimization community.

---

**Last Updated**: January 2026 (submitted 2026-01-14)
**Status**: Under Review for Applied Intelligence Journal
