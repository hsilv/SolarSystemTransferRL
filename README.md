# 🚀 Interplanetary Route Optimization with Reinforcement Learning

![mga_full_animation](https://github.com/user-attachments/assets/5c0947b2-8efe-44f0-93a3-1f087cba7a21)

## 🛰 Overview

This project aims to build an **AI-based system capable of finding optimal interplanetary routes** — for example, **from Earth to Saturn** — using **Reinforcement Learning (RL)** and **multi-gravity-assist (MGA)** trajectories.

The agent learns to decide:
- **Departure date** from Earth.
- **Intermediate gravity assists** (Venus, Earth, Jupiter, etc.).
- **Time of flight (TOF)** for each leg of the journey.

It evaluates trajectories using **patched-conic approximations** (Lambert arcs) and can visualize the resulting orbit paths through the **Solar System**.

---

## 🧭 Objectives

- Implement a modular simulation environment where an RL agent can:
  - Choose planetary sequences and transfer durations.
  - Optimize for **minimum total Δv** and **reasonable total time of flight**.
  - Handle failures gracefully when trajectories are unfeasible.

- Visualize the best trajectory in 2D or 3D using `poliastro`.

- Include a **deterministic backup optimizer** (via `pygmo`) for benchmarking RL results.

- Keep the project realistic yet executable in **1–2 days**, using simplified orbital dynamics (patched conics, no full n-body physics).

---

## 🧩 Core Features

| Component | Description |
|------------|-------------|
| `baseline_lambert.py` | Basic Earth→Saturn transfer using Lambert’s problem (visual check). |
| `mga_eval.py` | Evaluates multi-gravity-assist sequences using PyKEP. |
| `sequences.py` | Defines planetary flyby sequences and TOF limits per leg. |
| `env_mga.py` | Gym environment for Reinforcement Learning. |
| `train.py` | PPO training script using Stable-Baselines3. |
| `plot_route.py` | Visualizes the best route using Poliastro. |
| `solver_backup.py` | Deterministic optimization with PyGMO (non-RL backup). |
| `README.md` | Project documentation (this file). |

---

## ⚙️ Installation

### 1️⃣ Create a Python environment

```bash
python -m venv venv
source venv/bin/activate   # (Windows: venv\Scripts\activate)
pip install --upgrade pip
