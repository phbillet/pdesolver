# PDESolver: A Python Framework for Solving Partial Differential Equations

## Overview

`PDESolver` is a modular Python framework for the numerical solution and analysis of partial differential equations (PDEs) in 1D and 2D. It combines symbolic parsing, pseudo-spectral methods, and support for custom pseudo-differential operators. The framework enables advanced wave analysis, ellipticity diagnostics, and interactive visualizations for symbolic operators.

---

## Key Features

* **Symbolic Equation Parsing**

  * Automatically separates linear, nonlinear, symbolic operator (`Op`), pseudo-differential (`psiOp`), and source terms.
  * Supports both standard derivatives and custom pseudo-differential forms via `Op(expr, u)` or `psiOp(expr, u)`.

* **Pseudo-Differential Operators**

  * Define fractional, nonlocal, or space-dependent operators in Fourier space.
  * Symbol evaluation in either direct symbolic mode (`symbol`) or automatic extraction mode (`auto`).
  * Numerical symbol evaluation and visualization via the `PseudoDifferentialOperator` class (1D and 2D).

* **Spectral Discretization**

  * Uses FFT and IFFT for spatial derivatives.
  * Built-in dealiasing using configurable frequency cutoffs.

* **Temporal Integration**

  * Supports first- and second-order in time equations.
  * Multiple time schemes: exponential time stepping, ETD-RK4 (1st and 2nd order), and Leap-Frog integration.
  * Stationary PDEs (no time variable) are automatically detected and treated accordingly.

* **Visualization and Microlocal Analysis**

  * **Wavefront Set**: Visualize how singularities propagate across space and frequency.
  * **Cotangent Fiber**: Inspect symbol magnitude at fixed spatial locations.
  * **Symbol Amplitude / Phase**: View maps of operator behavior across the domain.
  * **Characteristic Set**: Identify vanishing zones of the symbol.
  * **Micro-Support & Hamiltonian Flow**: Track propagation of microlocal energy.
  * **Symplectic Vector Field & Group Velocity**: Analyze symbol-induced flows in phase space.

* **Wave Propagation Tools**

  * Compute dispersion relation, phase and group velocity.
  * Full anisotropy support in 2D systems.
  * Symbol quantization in Kohn–Nirenberg form for variable coefficients.

* **Stationary Solvers and Ellipticity Checks**

  * Automatic symbolic inversion via right asymptotic inverse when applicable.
  * Numerical ellipticity testing for `psiOp` symbols on grid.

* **Energy Monitoring**

  * Computes total system energy (especially for second-order systems).
  * Supports linear and logarithmic scale energy plots.

* **Animation and Solution Visualization**

  * Create time-evolving animations for 1D (line) and 2D (surface) solutions.
  * Interactive symbol analysis widgets for symbol inspection.
  * Options for overlaying phase fronts, wavefront tracking, and singularities.

---

## Installation

### Prerequisites

* Python 3.8 or higher
* Required libraries: `numpy`, `scipy`, `matplotlib`, `sympy`, `ipywidgets`

### Installing Dependencies

```bash
pip install numpy scipy matplotlib sympy ipywidgets
```

---

## Quick Start

1. **Define a PDE** using SymPy:

```python
from sympy import symbols, Function, Eq, diff
from PDESolver_33 import PDESolver
import numpy as np

t, x = symbols('t x')
u = Function('u')(t, x)
equation = Eq(diff(u, t), diff(u, x, 2) + u**2)
```

2. **Initialize the Solver**:

```python
solver = PDESolver(equation, time_scheme='ETD-RK4', dealiasing_ratio=2/3)
```

3. **Setup Domain and Initial Conditions**:

```python
def initial_condition(x):
    return np.sin(x)

solver.setup(
    Lx=2 * np.pi, Nx=128,
    Lt=1.0, Nt=1000,
    initial_condition=initial_condition
)
```

4. **Solve and Animate**:

```python
solver.solve()
ani = solver.animate(component='real')
```

---

## Jupyter Notebooks for Testing

Several Jupyter notebooks are provided to test and demonstrate the capabilities of the solver:

* `PDE_symbolic_tester.ipynb`: Verifies the correctness of exact solutions for 1D and 2D symbolic PDEs.
* `PDESolver_psiOp_tester.ipynb`: Tests the features of the PseudoDifferentialOperator class, including symbol evaluation and visualization.
* `PDESolver_tester_1D.ipynb`: Exercises the PDESolver class on various 1D PDEs such as the heat equation, wave equation, and equations with nonlocal or fractional operators.
* `PDESolver_tester_2D.ipynb`: Tests the PDESolver class on 2D PDEs including the heat equation, wave equation, and the Klein–Gordon equation.

These notebooks serve both as validation tools and as usage examples. They are ideal entry points for new users wishing to understand the solver's capabilities.

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes.

---

## License

PDESolver is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

You are free to use, modify, and redistribute this software under the terms of this license. Redistributions must include appropriate copyright.

### Author Attribution

If you use PDESolver in your project or derivative work, please retain the original author attribution:

**Philippe Billet (2025)**

---

## Acknowledgments

This project was built with the assistance of advanced language models, which played a key role in generating and structuring code and documentation. Thanks to **ChatGPT**, **Qwen**, **DeepSeek**, **Claude**, and **Mistral** for their contributions to automated reasoning and documentation synthesis.