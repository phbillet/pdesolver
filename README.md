# PDESolver: A Python Framework for Solving Partial Differential Equations

## Overview

`PDESolver` is a flexible and modular Python framework designed to solve partial differential equations (PDEs) numerically. It supports both 1D and 2D problems, and provides tools for analyzing wave propagation, stability, and energy conservation. The code leverages advanced numerical techniques such as spectral methods (via FFT), exponential time differencing (ETD), and Runge-Kutta schemes for temporal integration.

This framework is particularly useful for researchers, engineers, and students working on physical simulations involving PDEs, including wave equations, diffusion processes, and nonlinear dynamics.

---

## Key Features

- **Symbolic Parsing**: Automatically parses symbolic equations provided by the user to separate linear, nonlinear, and source terms.
- **Spectral Methods**: Uses Fast Fourier Transform (FFT) for spatial discretization, ensuring high accuracy and efficiency.
- **Dealiasing**: Implements spectral dealiasing to reduce aliasing errors in nonlinear terms.
- **Time Integration Schemes**:
  - Supports first-order and second-order temporal derivatives.
  - Includes default exponential integration and optional ETD-RK4 schemes for improved stability.
- **Wave Analysis**:
  - Computes dispersion relations, phase velocities, and group velocities.
  - Analyzes anisotropy in 2D systems.
- **Energy Conservation**: Computes total energy over time for second-order systems, aiding in stability analysis.
- **Visualization Tools**:
  - Animates solutions in 1D (line plots) and 2D (surface plots).
  - Provides overlays for contour lines or gradient fronts in 2D animations.
- **Testing and Validation**: Compares numerical solutions with exact solutions to ensure accuracy.

---

## Installation

### Prerequisites

- Python 3.8 or higher
- Required libraries: `numpy`, `scipy`, `matplotlib`, `sympy`

### Installing Dependencies

You can install the required libraries using `pip`:

```bash
pip install numpy scipy matplotlib sympy
```

### Downloading the Code

Clone this repository to your local machine:

```bash
git clone https://github.com/phbillet/pdesolver.git
cd PDESolver
```

---

## Usage

### Step 1: Define Your PDE

The solver accepts symbolic equations defined using SymPy. For example, to define a wave equation:

```python
from pdesolver import PDESolver

# Define symbols
t, x = symbols('t x')
u = Function('u')(t, x)

# Define the PDE
equation = Eq(diff(u, t, t) - diff(u, x, x), 0)
```

### Step 2: Initialize the Solver

Pass the equation to the `PDESolver` class:

```python
solver = PDESolver(equation)
```

### Step 3: Set Up the Domain and Initial Conditions

Configure the computational domain, grid resolution, and initial conditions:

```python
def initial_condition(x):
    return np.exp(-x**2)  # Example: Gaussian pulse

solver.setup(
    Lx=10, Nx=256, Lt=5, Nt=1000,
    initial_condition=initial_condition
)
```

For second-order systems, you may also need to provide an initial velocity:

```python
def initial_velocity(x):
    return np.zeros_like(x)

solver.setup(
    Lx=10, Nx=256, Lt=5, Nt=1000,
    initial_condition=initial_condition,
    initial_velocity=initial_velocity
)
```

### Step 4: Solve the PDE

Run the solver to compute the solution:

```python
solver.solve()
```

### Step 5: Visualize Results

Animate the solution to observe its evolution over time:

```python
ani = solver.animate(component='abs')
```

You can also plot the energy evolution for second-order systems:

```python
solver.plot_energy(log=True)
```

---

## Testing
Currently, testing is quite well covered by two dedicated notebooks: PDE_symbolic_tester.ipynb ensures that symbolic solutions and initial conditions are valid, while PDESolver_tester.ipynb (with 27 tests, that can be used as examples) uses these initial conditions to validate the solver's performance. 

The framework includes a testing utility to compare numerical solutions with exact solutions:

```python
def exact_solution(x, t):
    return np.exp(-(x - t)**2)  # Example: Traveling Gaussian wave

solver.test(exact_solution, t_eval=2.5, norm='relative', plot=True)
```

---
## Formulation of Differential Operators: Derivatives and Fourier-Space Symbols

In `PDESolver`, linear operators can be formulated in two equivalent ways: either using classical derivatives (`diff`) or using symbolic Fourier-space expressions with `Op(symbol, u)`. Both approaches are internally unified by the solver during the setup phase.

When derivatives are used, such as `diff(u, x, 2)` or `diff(u, t, t)`, the solver automatically translates them into their Fourier counterparts. Spatial derivatives (`x`, `y`) are mapped to powers of `i kx` and `i ky`, while time derivatives (`t`) introduce powers of `-i ω`. This allows the solver to infer the full symbolic dispersion relation without requiring the user to manually define it.

Alternatively, users can directly specify pseudo-differential operators using `Op(symbol, u)`, where `symbol` is an explicit function of the Fourier variables `kx`, `ky`, or the temporal frequency `omega`. This is useful for fractional derivatives, nonlocal operators, or custom dispersions. 

Both methods are fully compatible and can be mixed within the same PDE. Care must be taken to express spatial dependencies in terms of `kx`, `ky`, and temporal dependencies using `omega`, to ensure correct symbolic treatment.

Choosing between derivatives and symbolic expressions offers flexibility: use derivatives for standard PDEs, and `Op` when precise control over the Fourier symbol is needed.

---

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed description of your changes.

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
