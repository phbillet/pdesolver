# 🧮 PDESolver: A Symbolic & Spectral Python Framework for PDEs

![Test Notebooks](https://github.com/phbillet/pdesolver/actions/workflows/test-notebooks.yml/badge.svg)
[![codecov](https://codecov.io/gh/phbillet/pdesolver/branch/master/graph/badge.svg)](https://codecov.io/gh/phbillet/pdesolver)

---

## ✨ Overview

**PDESolver** is a symbolic and spectral Python framework for solving **partial differential equations (PDEs)** in 1D and 2D.  
It supports:

- Time-dependent (1st and 2nd order) **and stationary** PDEs,
- Periodic and **Dirichlet boundary conditions**,
- Fully symbolic **pseudo-differential operators** via `psiOp(...)`,
- Advanced **microlocal analysis** and Hamiltonian flow simulation.

---

## 🚀 Key Features

### 📌 Symbolic PDE Parsing
- Accepts `sympy` equations with arbitrary structure.
- Separates linear, nonlinear, source, `Op(...)`, and `psiOp(...)` terms.
- Supports nonlocal, variable-coefficient and fractional operators.

### 🧠 Pseudo-Differential Operators
- 1D & 2D support via `PseudoDifferentialOperator` class.
- Symbol mode: manual symbolic definition.
- Auto mode: symbolic derivation from differential expressions.
- Asymptotic tools: principal symbol, order, adjoints, inverse composition.

### 🧮 Spatial Discretization
- Spectral methods via FFT/IFFT.
- Automatic handling of:
  - Periodic boundary conditions.
  - Dirichlet conditions via sine transforms.
- Optional **dealiasing** (e.g. 2/3 rule).

### ⏱ Time Integration Schemes
- **First-order** and **second-order** time-dependent PDEs.
- **Stationary PDEs** handled automatically (symbolic inversion if elliptic).
- Built-in schemes:
  - Exponential stepping (default for `psiOp`)
  - ETD-RK4 (1st & 2nd order)
  - Leap-Frog (energy-conserving)

### 🧭 Microlocal & Spectral Analysis
- 📈 Wavefront Set tracking
- 🔬 Symbol amplitude & phase plots
- 🎯 Characteristic & micro-support sets
- 🌀 Hamiltonian & symplectic flow visualization
- 📡 Group velocity fields

### 🔍 Ellipticity & Inversion
- Automatic symbolic inversion via **asymptotic right inverse**.
- Symbolic order analysis & homogeneity checks.
- Numerical ellipticity tests on grid.

### 📉 Energy Monitoring
- Total energy for second-order systems (optional log-scale).
- Auto-conservation check with Leap-Frog or self-adjoint operators.

### 🎞 Animation & Widgets
- Animated solution visualizations (1D/2D).
- Interactive `ipywidgets` for symbol inspection.
- Phase front overlay & singularity tracking.

---

## 📦 Installation

### Requirements

- Python ≥ 3.8
- `numpy`, `scipy`, `matplotlib`, `sympy`, `ipywidgets`

```bash
pip install numpy scipy matplotlib sympy ipywidgets
```

---

## ⚡ Quick Start

```python
from sympy import symbols, Function, Eq, diff
from PDESolver_38 import PDESolver
import numpy as np

# Define PDE
t, x = symbols('t x')
u = Function('u')(t, x)
equation = Eq(diff(u, t, t), diff(u, x, 2) - u)

# Init solver
solver = PDESolver(equation, time_scheme='LeapFrog')

# Setup domain
solver.setup(
    Lx=2*np.pi, Nx=256,
    Lt=2.0, Nt=1000,
    initial_condition=lambda x: np.sin(x),
    initial_velocity=lambda x: 0*x,
    boundary='dirichlet'  # or 'periodic'
)

# Solve & animate
solver.solve()
solver.animate(component='real')
HTML(ani.to_jshtml())
```

---

## 🧪 Test Notebooks

| Notebook                               | Description                                                        |
|----------------------------------------|--------------------------------------------------------------------|
| `PDE_symbolic_tester.ipynb`            | Verifies symbolic parsing & solutions                              |
| `PDESolver_psiOp_tester.ipynb`         | Tests `psiOp` visualization & symbolic analysis                    |
| `PDESolver_tester_1D_periodic.ipynb`   | 1D periodic PDEs: heat, wave, fractional Laplacian, etc.           |
| `PDESolver_tester_1D_Dirichlet.ipynb`  | 1D Dirichlet: Schrödinger, Airy, Hermite, Legendre, etc.           |
| `PDESolver_tester_2D_periodic.ipynb`   | 2D periodic PDEs: wave, Klein–Gordon, diffusion, etc.              |
| `PDESolver_tester_2D_Dirichlet.ipynb`  | 2D Dirichlet examples: Laplace, Helmholtz, Schrödinger             |

Use them to explore features and validate new equations.

---

## 🤝 Contributing

Pull requests welcome! Fork the repo, make a feature branch, and submit with a clear description.

---

## 📜 License

**Apache License 2.0**  
© 2025 [Philippe Billet](https://github.com/phbillet)

---

## 🙏 Acknowledgments

This project is made possible thanks to symbolic automation and research support from models like **ChatGPT**, **Qwen**, **Claude**, and **Mistral**.