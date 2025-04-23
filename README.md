# PDESolver: A Flexible Python Framework for Solving Partial Differential Equations (PDEs)

**PDESolver** is a powerful and modular Python library designed to numerically solve partial differential equations (PDEs) in 1D and 2D. Whether you're working on physics, engineering, or mathematical modeling problems, this tool provides the flexibility and precision needed to tackle complex PDEs efficiently.

### **Author's note:**
- This project was started at the beginning of April 2025 and both the code and this documentation have been entirely and iteratively generated using LLM. It is not bug-free and should not be used for professional purposes.
- From a theoretical point of view, Chebyshev's pseudo-spectral method should be implemented for non-periodic boundary conditions, as indicated by some LLMs. A volume penalization has been added to alleviate this problem.
- In spite of this, any contributions aimed at perfecting this code are welcome.

---

**Ready to solve your PDEs? Download PDESolver now and unlock the power of numerical PDE solving!**

---

## **Key Features**

### **1. Versatile Equation Parsing**
- Automatically parse and classify terms of your PDE into **linear**, **nonlinear**, and **symbolic** components.
- Support for symbolic operators (`Op`) in Fourier space, enabling the use of custom pseudo-differential operators.
- Handles equations with derivatives of any order in time (`t`) and space (`x`, `y`).

### **2. Advanced Linear Operator Computation**
- Compute the **spectral symbol** of the linear operator directly in Fourier space.
- Automatically derive the **dispersion relation** and handle both first-order (`∂ₜu = Lu + N(u)`) and second-order (`∂ₜₜu = Lu + N(u)`) temporal derivatives.
- Supports fully symbolic operators for problems where explicit differentiation is not feasible.

### **3. Robust Numerical Methods**
- **Exponential Time Differencing (ETD)** schemes:
  - Default exponential integration (`e^{LΔt}`).
  - High-order **ETD-RK4** for improved accuracy.
- **Dealiasing** to prevent aliasing errors in nonlinear terms.
- Stability checks, including **CFL condition verification** and spectral analysis of the linear operator.

### **4. Flexible Boundary Conditions**
- Handle a wide range of boundary conditions:
  - **Dirichlet**, **Neumann**, **Robin**, and **periodic**.
  - Curvilinear domains defined by characteristic functions (`f(x, y) ≤ 0`).
- Smooth interpolation near boundaries for enhanced accuracy.
- Improved handling of Neumann conditions using calculated normals for curvilinear domains.

### **5. Visualization Tools**
- Animate solutions in real-time:
  - 1D: Line plots showing evolution over time.
  - 2D: Surface plots with optional overlays (e.g., contours or fronts).
- Built-in tools for testing against exact solutions and visualizing error distributions.
- Enhanced wave propagation analysis for second-order temporal derivatives.

---

## **Equation Examples**

### **Supported Equations**
PDESolver can handle a wide variety of PDEs, including but not limited to:
- **Heat equation**:  
   ∂ₜu = ∂ₓₓu 
- **Wave equation**:  
   ∂ₜₜu = ∂ₓₓu 
- **Nonlinear Schrödinger equation**:  
   i∂ₜu = -∂ₓₓu + |u|²u 
- Custom equations with pseudo-differential operators:
   ∂ₜu = Op(k²)u + N(u) 

### **Symbolic Operators**
For equations involving non-standard operators (e.g., fractional Laplacians), you can define them symbolically using the `Op` class:
```python
from sympy import symbols, Function
k = symbols('k')
u = Function('u')(t, x)
equation = Eq(diff(u, t), Op(k**2, u))
```

---

## **Why Choose PDESolver?**

- **Ease of Use**: Set up and solve complex PDEs with minimal code.
- **Modularity**: Each component (parsing, boundary conditions, integration) is encapsulated for clarity and extensibility.
- **Performance**: Optimized FFT-based computations ensure fast and accurate results.
- **Flexibility**: From simple rectangular domains to curvilinear geometries, PDESolver adapts to your problem's needs.

---

## **Quick Start**

1. **Install Dependencies**:
   ```bash
   pip install numpy scipy matplotlib sympy
   ```

2. **Define Your PDE**:
   ```python
   from sympy import symbols, Function, diff, Eq
   from PDESolver import PDESolver

   t, x = symbols('t x')
   u = Function('u')(t, x)
   equation = Eq(diff(u, t), diff(u, x, x))  # Heat equation
   solver = PDESolver(equation, boundary_condition='dirichlet')
   ```

3. **Set Up and Solve**:
   ```python
   def initial_condition(x):
       return np.exp(-x**2)

   solver.setup(Lx=10, Nx=256, Lt=1.0, Nt=100, initial_condition=initial_condition)
   solver.solve()
   ```

4. **Visualize**:
   ```python
   ani = solver.animate(component='abs')
   ```

---

## **New Features and Improvements**

### **1. Enhanced Boundary Handling**
- Improved **Neumann boundary conditions** for curvilinear domains using calculated normals.
- Added support for **smooth interpolation** at boundaries to reduce artifacts.

### **2. Wave Propagation Analysis**
- New method `analyze_wave_propagation()` to analyze:
  - **Dispersion relation**: $ \omega(k) $
  - **Phase velocity**: $ v_p(k) = \omega / |k| $
  - **Group velocity**: $ v_g(k) = \nabla_k \omega(k) $
  - **Anisotropy** in 2D.

### **3. Symbol Plotting**
- Visualize the spectral symbol (`L_symbolic`) in 1D and 2D using `plot_symbol()`.
- Options to plot the real part, imaginary part, or magnitude of the symbol.

### **4. Testing and Validation**
- Added `test()` method to compare numerical solutions against exact solutions.
- Comprehensive error analysis with options for relative and absolute norms.

### **5. Performance Optimizations**
- Improved dealiasing logic to enhance stability for high-frequency components.
- Optimized FFT workers for faster computations.

---

## **Contribute and Support**

- **Download**: Clone the repository and start solving your PDEs today!
- **Issues**: Found a bug or have a feature request? Open an issue on GitHub.
- **Contributions**: We welcome contributions! Whether it's improving documentation, adding new features, or optimizing performance, your input is valuable.  
Currently, testing is partially covered by two dedicated notebooks: `PDE_symbolic_tester.ipynb` ensures that symbolic solutions and initial conditions are valid, while `PDESolver_tester.ipynb` uses these initial conditions to validate the solver's performance. However, there is still significant work to be done to expand the test coverage and improve robustness. Any contributions to enhance the testing framework or add new test cases are highly appreciated!

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