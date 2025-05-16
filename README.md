# PDESolver: A Python Framework for Solving Partial Differential Equations

## Overview

`PDESolver` is a comprehensive and modular Python framework for the numerical solution and analysis of partial differential equations (PDEs) in 1D and 2D. It combines symbolic equation parsing with spectral methods to offer a flexible environment for wave propagation, dispersion analysis, and energy monitoring. New features include direct support for custom pseudo-differential operators and a suite of advanced visualization tools.

---

## Key Features

* **Symbolic Parsing**

  * Automatically separates linear, nonlinear, symbolic operator, pseudo-differential, and source terms from user-defined equations.
  * Supports both classical derivatives and custom operator wrappers (`Op` and `psiOp`).

* **Pseudo-Differential Operators**

  * Define fractional or nonlocal operators directly in Fourier space using `Op(symbol, u)` and `psiOp(symbol, u)` wrappers.
  * Automatic extraction of operator symbols from symbolic expressions when needed.
  * Numerical evaluation via the `PseudoDifferentialOperator` class for both 1D and 2D domains.

* **Spectral Methods**

  * Fast Fourier Transform (FFT) based spatial discretization for high accuracy.
  * Dealiasing strategies to mitigate spectral aliasing errors.

* **Time Integration Schemes**

  * Exponential time stepping and ETD-RK4 methods for first-order and second-order time derivatives.
  * Leap-frog and Runge-Kutta schemes for robust temporal evolution.

* **Visualization and Analysis**

  * **Wavefront Set**: Visualize the magnitude of operator symbols across position and frequency domains.
  * **Cotangent Fiber Structure**: Contour plots showing symbol values over the phase space.
  * **Symbol Amplitude and Phase Portraits**: Color maps of absolute values and phase of symbols.
  * **Characteristic Set**: Contours indicating where the symbol vanishes.
  * **Dynamic Wavefront**: Time-evolving wavefront visualization in 1D and 2D.
  * Automated integration of symbol and solution visualizations during setup and solving.

* **Wave Propagation Analysis**

  * Compute and plot dispersion relations, phase velocity, and group velocity.
  * Support for anisotropy analysis in 2D systems.

* **Energy Monitoring**

  * Track total energy over time for second-order systems.
  * Plot energy evolution with options for linear and logarithmic scales.

* **Animation Tools**

  * Generate animations of solution evolution in 1D (line plots) and 2D (surface plots).
  * Options for overlays, contour lines, or front gradients.

---

## Installation

### Prerequisites

* Python 3.8 or higher
* Required libraries: `numpy`, `scipy`, `matplotlib`, `sympy`

### Installing Dependencies

Install the necessary packages via pip:

```bash
pip install numpy scipy matplotlib sympy
```

### Downloading the Code

Clone the repository and navigate into its directory:

```bash
git clone https://github.com/phbillet/pdesolver.git
cd pdesolver
```

---

## Quick Start

1. **Define a PDE** using SymPy:

   ```python
   from sympy import symbols, Function, Eq, diff
   from pdesolver import PDESolver

   t, x = symbols('t x')
   u = Function('u')(t, x)
   equation = Eq(diff(u, t, t) - diff(u, x, 2), 0)
   ```

2. **Initialize the Solver**:

   ```python
   solver = PDESolver(equation)
   ```

3. **Setup Domain and Initial Conditions**:

   ```python
   def initial_condition(x):
       return np.exp(-x**2)

   solver.setup(
       Lx=10, Nx=256,
       Lt=5, Nt=1000,
       initial_condition=initial_condition
   )
   ```

4. **Solve the PDE**:

   ```python
   solver.solve()
   ```

5. **Visualize and Animate**:

   ```python
   ani = solver.animate(component='abs')
   solver.plot_energy(log=True)
   ```

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with a clear description of your changes.

---


## **License**

PDESolver is distributed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

You are free to use, modify, and redistribute this software under the terms of this license.
Redistributions must include appropriate copyright notices.

### Author attribution

If you use PDESolver in your project or derivative work, please retain the original author attribution:

**Philippe Billet (2025)**  

---

## **Acknowledgments**

The code and documentation presented here were generated with the assistance of large language models (LLMs), utilizing their free-tier capabilities. This project would not have been possible without the invaluable support of these advanced AI tools. We extend our gratitude to **ChatGPT**, **Qwen**, **DeepSeek**, **Claude**, and the **Mistral chat** for their contributions in generating, refining, and structuring both the code and its accompanying documentation. Their ability to assist in complex problem-solving and technical writing has been instrumental in bringing this project to life. Thank you!
