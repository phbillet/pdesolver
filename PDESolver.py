# Copyright 2025 Philippe Billet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft, fftfreq
from sympy import (
    symbols, Function, 
    solve, pprint, Mul,
    lambdify, expand, Eq, simplify, trigsimp, N,
    Lambda, Piecewise, Basic, degree, Pow, preorder_traversal,
    sqrt, 
    I,  pi,
    re, im, arg, Abs, conjugate, 
    sin, cos, tan, cot, sec, csc, sinc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    exp, ln, 
    diff, Derivative, integrate, 
    fourier_transform, inverse_fourier_transform,
)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from functools import partial
from misc import * 

plt.rcParams['text.usetex'] = False

class Op(Function):
    """Custom symbolic wrapper for pseudo-differential operators in Fourier space.
    Usage: Op(symbol_expr, u)
    """
    nargs = 2


class psiOp(Function):
    """Symbolic wrapper for PseudoDifferentialOperator"""
    nargs = 2   # (expr, u)

class PseudoDifferentialOperator:
    """
    Pseudo-differential operator with dynamic symbol evaluation on spatial grids.
    Supports both 1D and 2D operators, and can be defined explicitly (symbol mode)
    or extracted automatically from symbolic equations (auto mode).

    Parameters
    ----------
    expr : sympy expression
        Symbolic expression representing the pseudo-differential symbol.
    vars_x : list of sympy symbols
        Spatial variables (e.g., [x] for 1D, [x, y] for 2D).
    var_u : sympy function, optional
        Function u(x, t) used in auto mode to extract the operator symbol.
    mode : str, {'symbol', 'auto'}
        - 'symbol': directly uses expr as the operator symbol.
        - 'auto': computes the symbol automatically by applying expr to exp(i x xi).

    Notes
    -----
    - Supports 1D and 2D operators.
    - Uses numpy for numerical evaluation and scipy.fft for FFTs.

    Examples
    --------
    >>> # Example 1: 1D Laplacian operator (symbol mode)
    >>> from sympy import symbols
    >>> x, xi = symbols('x xi', real=True)
    >>> op = PseudoDifferentialOperator(expr=xi**2, vars_x=[x], mode='symbol')

    >>> # Example 2: 1D transport operator (auto mode)
    >>> from sympy import Function
    >>> u = Function('u')
    >>> expr = u(x).diff(x)
    >>> op = PseudoDifferentialOperator(expr=expr, vars_x=[x], var_u=u(x), mode='auto')

    """

    def __init__(self, expr, vars_x, var_u=None, mode='symbol'):
        self.dim = len(vars_x)
        self.mode = mode
        self.fft_workers = 4
        self.symbol_cached = None
        self.expr = expr

        if self.dim == 1:
            x, = vars_x
            xi_internal = symbols('xi', real=True)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            self.fft = partial(fft, workers=self.fft_workers)
            self.ifft = partial(ifft, workers=self.fft_workers)

            if mode == 'symbol':
                self.p_func = lambdify((x, xi_internal), expr, 'numpy')
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * x * xi_internal)
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                self.p_func = lambdify((x, xi_internal), symbol, 'numpy')
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        elif self.dim == 2:
            x, y = vars_x
            xi_internal, eta_internal = symbols('xi eta', real=True)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            expr = expr.subs(symbols('eta', real=True), eta_internal)
            self.fft = partial(fft2, workers=self.fft_workers)
            self.ifft = partial(ifft2, workers=self.fft_workers)

            if mode == 'symbol':
                self.p_func = lambdify((x, y, xi_internal, eta_internal), expr, 'numpy')
            elif mode == 'auto':
                if var_u is None:
                    raise ValueError("var_u must be provided in mode='auto'")
                exp_i = exp(I * (x * xi_internal + y * eta_internal))
                P_ei = expr.subs(var_u, exp_i)
                symbol = simplify(P_ei / exp_i)
                self.p_func = lambdify((x, y, xi_internal, eta_internal), symbol, 'numpy')
            else:
                raise ValueError("mode must be 'auto' or 'symbol'")

        else:
            raise NotImplementedError("Only 1D and 2D supported")

        print("\nsymbol = ")
        pprint(expr)

    def evaluate(self, X, Y, KX, KY, cache=True):
        if cache and self.symbol_cached is not None:
            return self.symbol_cached

        if self.dim == 1:
            symbol = self.p_func(X, KX)
        elif self.dim == 2:
            symbol = self.p_func(X, Y, KX, KY)
        else:
            raise NotImplementedError("Only 1D and 2D supported")

        if cache:
            self.symbol_cached = symbol

        return symbol

    def clear_cache(self):
        self.symbol_cached = None

    def visualize_wavefront(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        if self.dim == 1:
            symbol_vals = self.p_func(x_grid[:, None], xi_grid[None, :])
            plt.imshow(np.abs(symbol_vals), extent=[xi_grid.min(), xi_grid.max(), x_grid.min(), x_grid.max()], aspect='auto', origin='lower')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('ξ (frequency)')
            plt.ylabel('x (position)')
            plt.title('Wavefront Set (|Symbol(x, ξ)|)')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.imshow(np.abs(symbol_vals), extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],aspect='auto', origin='lower')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Wavefront Set at ξ={xi0}, η={eta0}')
            plt.show()

    def visualize_fiber(self, x_grid, xi_grid, y0=0.0, x0=0.0):
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI)
            plt.contourf(X, XI, np.abs(symbol_vals), levels=50, cmap='viridis')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x (position)')
            plt.ylabel('ξ (frequency)')
            plt.title('Cotangent Fiber Structure')
            plt.show()
        elif self.dim == 2:
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, xi_grid)
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            plt.contourf(xi_grid, xi_grid, np.abs(symbol_vals), levels=50, cmap='viridis')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Cotangent Fiber at x={x0}, y={y0}')
            plt.show()

    def visualize_symbol_amplitude(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.pcolormesh(X, XI, np.abs(symbol_vals), shading='auto')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Symbol Amplitude |p(x, ξ)|')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.pcolormesh(X, Y, np.abs(symbol_vals), shading='auto')
            plt.colorbar(label='|Symbol|')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Symbol Amplitude at ξ={xi0}, η={eta0}')
            plt.show()

    def visualize_phase(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.pcolormesh(X, XI, np.angle(symbol_vals), shading='auto', cmap='twilight')
            plt.colorbar(label='arg(Symbol) [rad]')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Phase Portrait (arg p(x, ξ))')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.pcolormesh(X, Y, np.angle(symbol_vals), shading='auto', cmap='twilight')
            plt.colorbar(label='arg(Symbol) [rad]')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Phase Portrait at ξ={xi0}, η={eta0}')
            plt.show()

    def visualize_characteristic_set(self, x_grid, xi_grid, y0=0.0, x0=0.0):
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.contour(X, XI, np.abs(symbol_vals), levels=[1e-5], colors='red')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Characteristic Set (p(x, ξ) ≈ 0)')
            plt.show()
        elif self.dim == 2:
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, xi_grid)
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            plt.contour(xi_grid, xi_grid, np.abs(symbol_vals), levels=[1e-5], colors='red')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Characteristic Set at x={x0}, y={y0}')
            plt.show()

    def visualize_dynamic_wavefront(self, x_grid, t_grid, y_grid=None, xi0=5.0, eta0=0.0):
        if self.dim == 1:
            X, T = np.meshgrid(x_grid, t_grid)
            U = np.cos(xi0 * X - xi0 * T)
            plt.imshow(U, extent=[t_grid.min(), t_grid.max(), x_grid.min(), x_grid.max()], aspect='auto', origin='lower', cmap='seismic')
            plt.colorbar(label='u(x, t)')
            plt.xlabel('t (time)')
            plt.ylabel('x (position)')
            plt.title('Dynamic Wavefront u(x, t)')
            plt.show()
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid)
            U = np.cos(xi0 * X + eta0 * Y - np.sqrt(xi0**2 + eta0**2) * t_grid[0])
            plt.imshow(U, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], aspect='auto', origin='lower', cmap='seismic')
            plt.colorbar(label='u(x, y)')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Dynamic Wavefront at t={t_grid[0]}')
            plt.show()

class PDESolver:
    """
    A PDE solver based on spectral methods using Fourier transforms.

    Features:
        - Handles symbolic PDEs via sympy
        - Supports 1D and 2D problems
        - Temporal integration schemes: default exponential time stepping and ETD-RK4
        - Nonlinear terms handled via pseudo-spectral method
        - Visualization and analysis tools included

    Example usage:
    >>> from sympy import Function, diff, symbols
    >>> u = Function('u')
    >>> t, x = symbols('t x')
    >>> eq = Eq(diff(u(t,x), t), diff(u(t,x), x, 2) + u(t,x)**2)
    >>> def initial(x): return np.sin(x)
    >>> solver = PDESolver(eq)
    >>> solver.setup(Lx=2*np.pi, Nx=128, Lt=1.0, Nt=1000, initial_condition=initial)
    >>> solver.solve()
    >>> ani = solver.animate()
    >>> HTML(ani.to_jshtml())
    """
    def __init__(self, equation, time_scheme='default', dealiasing_ratio=2/3):
        """
        Initialize the PDE solver with a given equation.
    
        Args:
            equation (sympy.Eq): The PDE to solve.
            time_scheme (str): 'default' or 'ETD-RK4'
            dealiasing_ratio (float): Ratio for dealiasing mask (e.g., 2/3)
        """
        self.time_scheme = time_scheme # 'default'  or 'ETD-RK4'
        self.dealiasing_ratio = dealiasing_ratio
        
        print("\n*********************************")
        print("* Partial differential equation *")
        print("*********************************\n")
        pprint(equation)
        
        # Extract symbols and function from the equation
        functions = equation.atoms(Function)
        candidate_functions = [f for f in functions if any(str(arg) == 't' for arg in f.args)]
        
        if len(candidate_functions) != 1:
            raise ValueError("The equation must contain exactly one unknown function")
        
        self.u = candidate_functions[0]
        args = self.u.args
        if len(args) < 2 or len(args) > 3:
            raise ValueError("The function must depend on t and at least one spatial variable (x [, y])")
    
        self.t = args[0]
        self.spatial_vars = args[1:]
        self.dim = len(self.spatial_vars)
        if self.dim == 1:
            self.x = self.spatial_vars[0]
            self.y = None
        elif self.dim == 2:
            self.x, self.y = self.spatial_vars
    
        self.fft_workers = 4
        
        if self.dim == 1:
            self.fft = partial(fft, workers=self.fft_workers)
            self.ifft = partial(ifft, workers=self.fft_workers)
        else:
            self.fft = partial(fft2, workers=self.fft_workers)
            self.ifft = partial(ifft2, workers=self.fft_workers)
        # Parse the equation
        self.linear_terms = {}
        self.nonlinear_terms = []
        self.symbol_terms = []
        self.source_terms = []
        self.pseudo_terms = []
        self.temporal_order = 0  # Order of the temporal derivative
        self.linear_terms, self.nonlinear_terms, self.symbol_terms, self.source_terms, self.pseudo_terms = self.parse_equation(equation)
        # flag : pseudo‑differential operator present ?
        self.has_psi = bool(self.pseudo_terms)
        if self.has_psi:
            print("⚠️  Pseudo‑differential operator detected: all other linear terms have been rejected.")
    
        if self.dim == 1:
            self.kx = symbols('kx')
        elif self.dim == 2:
            self.kx, self.ky = symbols('kx ky')
    
        # Compute linear operator
        self.compute_linear_operator()

    def parse_equation(self, equation):
        """
        Parse the PDE to separate linear and nonlinear terms.
        Args:
            equation (sympy.Eq): Partial Differential Equation to parse.
        Returns:
            tuple: Dictionary of linear terms, list of nonlinear terms, list of symbolic operator terms (Op),
                   list of source terms, and list of pseudo-differential operator terms (psiOp).
        """
        def is_nonlinear_term(term, u_func):
            if any(arg.has(u_func) for arg in term.args if isinstance(arg, Function) and arg.func != u_func.func):
                return True
            if any(isinstance(arg, Pow) and arg.base == u_func and (arg.exp != 1) for arg in term.args):
                return True
            if term.func == Mul:
                factors = term.args
                has_u = any(f == u_func for f in factors)
                has_derivative = any(isinstance(f, Derivative) and f.expr.func == u_func.func for f in factors)
                if has_u and has_derivative:
                    return True
            if term.has(u_func) and isinstance(term, Function) and term.func != u_func.func:
                return True
            return False
    
        print("\n********************")
        print("* Equation parsing *")
        print("********************\n")
    
        if isinstance(equation, Eq):
            lhs = equation.lhs - equation.rhs
        else:
            lhs = equation
    
        print(f"\nEquation rewritten in standard form: {lhs}")
        lhs_expanded = expand(lhs)
        print(f"\nExpanded equation: {lhs_expanded}")
    
        linear_terms = {}
        nonlinear_terms = []
        symbol_terms = []
        source_terms = []
        pseudo_terms = []
    
        for term in lhs_expanded.as_ordered_terms():
            print(f"Analyzing term: {term}")
    
            if term.has(psiOp):
                psiops = term.atoms(psiOp)
                for psi in psiops:
                    coeff = term / psi
                    expr = psi.args[0]
                    pseudo_terms.append((coeff, expr))
                    print("  --> Classified as pseudo linear term (psiOp)")
                continue
    
            if term.has(Op):
                ops = term.atoms(Op)
                for op in ops:
                    coeff = term / op
                    expr = op.args[0]
                    symbol_terms.append((coeff, expr))
                    print("  --> Classified as symbolic linear term (Op)")
                continue
    
            if is_nonlinear_term(term, self.u):
                nonlinear_terms.append(term)
                print("  --> Classified as nonlinear")
                continue
    
            derivs = term.atoms(Derivative)
            if derivs:
                deriv = derivs.pop()
                coeff = term / deriv
                linear_terms[deriv] = linear_terms.get(deriv, 0) + coeff
                print(f"  Derivative found: {deriv}")
                print("  --> Classified as linear")
            elif self.u in term.atoms(Function):
                coeff = term.as_coefficients_dict().get(self.u, 1)
                linear_terms[self.u] = linear_terms.get(self.u, 0) + coeff
                print("  --> Classified as linear")
            else:
                source_terms.append(term)
                print("  --> Classified as source term")
    
        print(f"Final linear terms: {linear_terms}")
        print(f"Final nonlinear terms: {nonlinear_terms}")
        print(f"Symbol terms: {symbol_terms}")
        print(f"Pseudo terms: {pseudo_terms}")
        print(f"Source terms: {source_terms}")
    
        if pseudo_terms:
            time_terms = {term: coeff for term, coeff in linear_terms.items()
                          if any(isinstance(arg, Derivative) or 'Derivative' in repr(term) for arg in [term])}
            other_linear = {t: c for t, c in linear_terms.items() if t not in time_terms}
            print("other_linear =", other_linear)
            print("symbol_terms =", symbol_terms)
    
            if other_linear or symbol_terms:
                raise ValueError(
                    "When using psiOp, only the time derivative term and the non-linear and source terms are permitted alongside the pseudo operator."
                )
    
        return linear_terms, nonlinear_terms, symbol_terms, source_terms, pseudo_terms

    def compute_linear_operator(self):
        """
        Compute the linear operator L(k) by applying each derivative to a plane wave.
        Automatically handles any derivative structure without hardcoding.
        """
        print("\n*******************************")
        print("* Linear operator computation *")
        print("*******************************\n")
    
        # --- Step 1: symbolic variables ---
        omega = symbols("omega")
        if self.dim == 1:
            kvars = [symbols("kx")]
            space_vars = [self.x]
        elif self.dim == 2:
            kvars = symbols("kx ky")
            space_vars = [self.x, self.y]
        else:
            raise ValueError("Only 1D and 2D are supported.")
    
        kdict = dict(zip(space_vars, kvars))
        self.k_symbols = kvars
    
        # Plane wave expression
        phase = sum(k * x for k, x in zip(kvars, space_vars)) - omega * self.t
        plane_wave = exp(I * phase)
    
        # --- Step 2: build lhs expression from linear terms ---
        lhs = 0
        for deriv, coeff in self.linear_terms.items():
            if isinstance(deriv, Derivative):
                total_factor = 1
                for var, n in deriv.variable_count:
                    if var == self.t:
                        total_factor *= (-I * omega)**n
                    elif var in kdict:
                        total_factor *= (I * kdict[var])**n
                    else:
                        raise ValueError(f"Unknown variable {var} in derivative")
                lhs += coeff * total_factor * plane_wave
            elif deriv == self.u:
                lhs += coeff * plane_wave
            else:
                raise ValueError(f"Unsupported linear term: {deriv}")
    
        # --- Step 3: dispersion relation ---
        equation = simplify(lhs / plane_wave)
        print("\nCharacteristic equation before symbol treatment:")
        pprint(equation)

        print("\n--- Symbolic symbol analysis ---")
        symb_omega = 0
        symb_k = 0
        
        for coeff, symbol in self.symbol_terms:
            if symbol.has(omega):
                # Ajouter directement les termes dépendant de omega
                symb_omega += coeff * symbol
            elif any(symbol.has(k) for k in self.k_symbols):
                 symb_k += coeff * symbol.subs(dict(zip(symbol.free_symbols, self.k_symbols)))

        print(f"symb_omega: {symb_omega}")
        print(f"symb_k: {symb_k}")
        
        equation = equation + symb_omega + symb_k         

        print("\nRaw characteristic equation:")
        pprint(equation)

        # Temporal derivative order detection
        try:
            poly_eq = Eq(equation, 0)
            poly = poly_eq.lhs.as_poly(omega)
            self.temporal_order = poly.degree() if poly else 0
        except:
            self.temporal_order = 0
        print(f"Temporal order from dispersion relation: {self.temporal_order}")
        print('self.pseudo_terms = ', self.pseudo_terms)
        if self.pseudo_terms:
            # on détecte l’ordre temporel comme avant
            # puis on instancie pour chaque terme :
            self.psi_ops = []
            for coeff, sym_expr in self.pseudo_terms:
                # expr est le Sympy expr. différentiel, var_x la liste [x] ou [x,y]
                psi = PseudoDifferentialOperator(sym_expr, self.spatial_vars, self.u, mode='symbol')
                
                self.psi_ops.append((coeff, psi))
        else:
            dispersion = solve(Eq(equation, 0), omega)
            if not dispersion:
                raise ValueError("No solution found for omega")
            print("\n--- Solutions found ---")
            pprint(dispersion)
        
            if self.temporal_order == 2:
                omega_expr = simplify(sqrt(dispersion[0]**2))
                self.omega_symbolic = omega_expr
                self.omega = lambdify(self.k_symbols, omega_expr, "numpy")
                self.L_symbolic = -omega_expr**2
            else:
                self.L_symbolic = -I * dispersion[0]
        
        
            self.L = lambdify(self.k_symbols, self.L_symbolic, "numpy")
  
            print("\n--- Final linear operator ---")
            pprint(self.L_symbolic)   

    def linear_rhs(self, u, is_v=False):
        """
        Apply the linear operator (in Fourier space) to the field u or v.
        Used to compute initial acceleration when no psiOp is used.
        """
        if self.dim == 1:
            self.symbol_u = np.array(self.L(self.KX), dtype=np.complex128)
            self.symbol_v = self.symbol_u  # même opérateur pour u et v
        elif self.dim == 2:
            self.symbol_u = np.array(self.L(self.KX, self.KY), dtype=np.complex128)
            self.symbol_v = self.symbol_u
        u_hat = self.fft(u)
        u_hat *= self.symbol_v if is_v else self.symbol_u
        u_hat *= self.dealiasing_mask
        return self.ifft(u_hat)


    def setup(self, Lx, Ly=None, Nx=None, Ny=None, Lt=1.0, Nt=100,
              initial_condition=None, initial_velocity=None, n_frames=100):
        """
        Set up the computational grid, initial conditions, and (optionally) the curvilinear domain.
        """

        # time stepping parameters
        self.Lt, self.Nt = Lt, Nt
        self.dt = Lt / Nt
        self.n_frames = n_frames
        self.frames = []

        # check spatial dimension requirements
        if self.dim == 1:
            if Nx is None:
                raise ValueError("Nx must be specified in 1D.")
        else:
            if None in (Ly, Ny):
                raise ValueError("Both Ly and Ny must be specified in 2D.")

        # 1D grid
        if self.dim == 1:
            self.Lx, self.Nx = Lx, Nx
            self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
            self.X = self.x_grid
            self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
            self.KX = self.kx

            # dealiasing
            k_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
            self.dealiasing_mask = (np.abs(self.KX) <= k_max)

            if self.temporal_order == 2 and not self.has_psi:
                omega_val = self.omega(self.KX)
                self.omega_val = omega_val
                self.cos_omega_dt = np.cos(omega_val * self.dt)
                self.sin_omega_dt = np.sin(omega_val * self.dt)
                self.inv_omega = np.zeros_like(omega_val)
                nonzero = omega_val != 0
                self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]

            self.u_prev = initial_condition(self.X)

            if self.has_psi:
                self.prepare_symbol_tables()
            
            if self.temporal_order == 2:
                self.v_prev = initial_velocity(self.X) if initial_velocity is not None else np.zeros_like(self.X)
            
                if self.has_psi:
                    acc0 = self.psiOp_fast(self.u_prev)
                else:
                    acc0 = self.linear_rhs(self.u_prev, is_v=False)
            
                self.u_prev2 = self.u_prev + self.dt * self.v_prev + 0.5 * self.dt**2 * acc0

        # 2D grid
        else:
            self.Lx, self.Ly = Lx, Ly
            self.Nx, self.Ny = Nx, Ny
            self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
            self.y_grid = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
            self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
            self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
            self.ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)
            self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
            kx_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
            ky_max = self.dealiasing_ratio * np.max(np.abs(self.ky))
            self.dealiasing_mask = (np.abs(self.KX) <= kx_max) & (np.abs(self.KY) <= ky_max)

            if self.temporal_order == 2 and not self.has_psi:
                omega_val = self.omega(self.KX, self.KY)
                self.omega_val = omega_val
                self.cos_omega_dt = np.cos(omega_val * self.dt)
                self.sin_omega_dt = np.sin(omega_val * self.dt)
                self.inv_omega = np.zeros_like(omega_val)
                nonzero = omega_val != 0
                self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]

        # If no psiOp, compute linear operator L and its exponential
        if not self.has_psi:
            if self.dim == 1:
                L_vals = np.array(self.L(self.KX), dtype=np.complex128)
                self.exp_L = np.exp(L_vals * self.dt)
            else:
                L_vals = self.L(self.KX, self.KY)
                self.exp_L = np.exp(L_vals * self.dt)

        # initial condition for u
        if self.dim == 1:
            self.u_prev = initial_condition(self.X)
        else:
            self.u_prev = initial_condition(self.X, self.Y)
        self.apply_boundary(self.u_prev)

        # for second order in time, set initial velocity v_prev
        if self.temporal_order == 2:
            if initial_velocity is None:
                raise ValueError("Initial velocity must be provided for second-order temporal derivatives")
            if self.dim == 1:
                self.v_prev = initial_velocity(self.X)
            else:
                self.v_prev = initial_velocity(self.X, self.Y)

        if self.has_psi:
            self.prepare_symbol_tables()

        if self.temporal_order == 2:
            if not hasattr(self, 'u_prev2'):
                # Compute initial acceleration a0 = L[u0] + nonlinear + source
                if self.has_psi:
                    acc0 = self.psiOp_fast(self.u_prev)
                else:
                    acc0 = self.linear_rhs(self.u_prev, is_v=False)
        
                rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                acc0 += rhs_nl
        
                if hasattr(self, 'source_terms') and self.source_terms:
                    # Evaluate source at t=0 similarly
                    source_contribution = 0  # (Add source evaluation here if needed)
                    acc0 += source_contribution
        
                # Initialize u_prev2 by Taylor expansion
                self.u_prev2 = self.u_prev + self.dt * self.v_prev + 0.5 * self.dt**2 * acc0

        self.frames = [self.u_prev.copy()]
        
        if self.has_psi:
            self.visualize_total_psi_symbol()
        else:
            self.check_cfl_condition()
    
            self.check_symbol_conditions()
    
            self.plot_symbol()
    
            if self.temporal_order == 2:
                self.analyze_wave_propagation()

    def visualize_total_psi_symbol(self, xi0=1.0, eta0=0.0):
        if not self.has_psi or not hasattr(self, 'psi_ops'):
            print("No pseudo-differential operator detected.")
            return
    
        # Créer un opérateur factice avec la somme des symboles
        from sympy import symbols
        x = self.spatial_vars[0]
        y = self.spatial_vars[1] if self.dim == 2 else None
        xi = symbols('xi', real=True)
        eta = symbols('eta', real=True) if self.dim == 2 else None
    
        total_expr = 0
        for coeff, psi in self.psi_ops:
            total_expr += coeff * psi.expr  # Accumule l'expression du symbole
    
        if self.dim == 1:
            fused = PseudoDifferentialOperator(total_expr, [x], mode='symbol')
            xg = self.X
            kg = self.KX
            fused.visualize_wavefront(xg, kg)
            fused.visualize_fiber(xg, kg)
            fused.visualize_symbol_amplitude(xg, kg)
            fused.visualize_phase(xg, kg)
            fused.visualize_characteristic_set(xg, kg)
            fused.visualize_dynamic_wavefront(xg, np.linspace(0, self.Lt, 100), xi0=xi0)
    
        elif self.dim == 2:
            fused = PseudoDifferentialOperator(total_expr, [x, y], mode='symbol')
            xg = self.x_grid
            yg = self.y_grid
            kx = self.kx
            fused.visualize_wavefront(xg, kx, yg, self.ky, xi0=xi0, eta0=eta0)
            fused.visualize_fiber(xg, kx, x0=0.0, y0=0.0)
            fused.visualize_symbol_amplitude(xg, kx, yg, self.ky, xi0=xi0, eta0=eta0)
            fused.visualize_phase(xg, kx, yg, self.ky, xi0=xi0, eta0=eta0)
            fused.visualize_characteristic_set(xg, kx, x0=0.0, y0=0.0)
            fused.visualize_dynamic_wavefront(xg, np.linspace(0, self.Lt, 100), yg, xi0=xi0, eta0=eta0)

            
    def apply_boundary(self, u):
        """Apply boundary conditions for a rectangular domain."""
        if self.dim == 1:
            u[0] = u[-2]
            u[-1] = u[1]
        elif self.dim == 2:
            u[0, :] = u[-2, :]
            u[-1, :] = u[1, :]
            u[:, 0] = u[:, -2]
            u[:, -1] = u[:, 1]

    def apply_nonlinear(self, u, is_v=False):
        """
        Apply nonlinear terms to the solution with dealiasing (spectral differentiation).
        Args:
            u (numpy.ndarray): Current solution grid.
            is_v (bool): Whether to compute nonlinear terms for v.
        Returns:
            numpy.ndarray: Contribution from nonlinear terms.
        """
        if not self.nonlinear_terms:
            return np.zeros_like(u, dtype=np.complex128)
        
        nonlinear_term = np.zeros_like(u, dtype=np.complex128)
    
        if self.dim == 1:
            u_hat = self.fft(u)
            u_hat *= self.dealiasing_mask
            u = self.ifft(u_hat)
    
            u_x_hat = (1j * self.KX) * u_hat
            u_x = self.ifft(u_x_hat)
    
            for term in self.nonlinear_terms:
                term_replaced = term
                if term.has(Derivative):
                    for deriv in term.atoms(Derivative):
                        if deriv.args[1][0] == self.x:
                            term_replaced = term_replaced.subs(deriv, symbols('u_x'))
                term_func = lambdify((self.t, self.x, self.u, 'u_x'), term_replaced, 'numpy')
                if is_v:
                    nonlinear_term += term_func(0, self.X, self.v_prev, u_x)
                else:
                    nonlinear_term += term_func(0, self.X, u, u_x)
    
        elif self.dim == 2:
            u_hat = self.fft(u)
            u_hat *= self.dealiasing_mask
            u = self.ifft(u_hat)
    
            u_x_hat = (1j * self.KX) * u_hat
            u_y_hat = (1j * self.KY) * u_hat
            u_x = self.ifft(u_x_hat)
            u_y = self.ifft(u_y_hat)
    
            for term in self.nonlinear_terms:
                term_replaced = term
                if term.has(Derivative):
                    for deriv in term.atoms(Derivative):
                        if deriv.args[1][0] == self.x:
                            term_replaced = term_replaced.subs(deriv, symbols('u_x'))
                        elif deriv.args[1][0] == self.y:
                            term_replaced = term_replaced.subs(deriv, symbols('u_y'))
                term_func = lambdify((self.t, self.x, self.y, self.u, 'u_x', 'u_y'), term_replaced, 'numpy')
                if is_v:
                    nonlinear_term += term_func(0, self.X, self.Y, self.v_prev, u_x, u_y)
                else:
                    nonlinear_term += term_func(0, self.X, self.Y, u, u_x, u_y)
        else:
            raise ValueError("Unsupported spatial dimension.")
        
        return nonlinear_term * self.dt

    def prepare_symbol_tables(self):
        """Precompute all psiOp symbols as arrays (real or complex)."""
        self.precomputed_symbols = []
        for coeff, psi in self.psi_ops:
            # Evaluate the symbol (can be 1D or 2D)
            if self.dim == 1:
                raw = psi.evaluate(self.X, None, self.KX, None)
            elif self.dim == 2:
                raw = psi.evaluate(self.X, self.Y, self.KX, self.KY)
            else:
                raise ValueError("Unsupported spatial dimension.")
    
            # Robust conversion: handle both 1D and 2D arrays
            raw_flat = raw.flatten()
            converted = np.array([complex(N(val)) for val in raw_flat], dtype=np.complex128)
            raw_eval = converted.reshape(raw.shape)
    
            self.precomputed_symbols.append((coeff, raw_eval))

    def psiOp_fast(self, u):
        u_hat = self.fft(u)
        combined_symbol = np.zeros_like(u_hat, dtype=np.complex128)
    
        for coeff, precomputed_symbol in self.precomputed_symbols:
            # Convert coeff to complex128 if not already
            coeff = np.complex128(coeff)
    
            # Ensure symbol is complex128
            symbol = np.array(precomputed_symbol, dtype=np.complex128)
    
            combined_symbol += coeff * symbol
    
        # Spectral update
        u_hat *= np.exp(-self.dt * combined_symbol)
        u_hat *= self.dealiasing_mask
    
        return self.ifft(u_hat)

    def solve(self):
        """
        Solve the PDE with the chosen time integration scheme.
        Handles both first-order and second-order in time equations.
        Supports:
            - Default exponential time-stepping (linear propagation + nonlinear correction)
            - ETD-RK4 (Exponential Time Differencing Runge-Kutta of 4th order)
        """
        print("\n*******************")
        print("* Solving the PDE *")
        print("*******************\n")
        
        save_interval = max(1, self.Nt // self.n_frames)
        self.energy_history = []
    
        for step in range(self.Nt):
            # Evaluate source term
            if hasattr(self, 'source_terms') and self.source_terms:
                source_contribution = np.zeros_like(self.X, dtype=np.float64)
                for term in self.source_terms:
                    try:
                        if self.dim == 1:
                            source_func = lambdify((self.t, self.x), term, 'numpy')
                            source_contribution += source_func(step * self.dt, self.X)
                        elif self.dim == 2:
                            source_func = lambdify((self.t, self.x, self.y), term, 'numpy')
                            source_contribution += source_func(step * self.dt, self.X, self.Y)
                    except Exception as e:
                        print(f"Error evaluating source term {term}: {e}")
            else:
                source_contribution = 0
    
            # First-order in time
            if self.temporal_order == 1:
                if self.has_psi:
                    u_sym = self.psiOp_fast(self.u_prev)
                    u_nl = self.apply_nonlinear(u_sym)
                    u_new = u_sym + u_nl
                else:
                    if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                        u_new = self.step_ETD_RK4(self.u_prev)
                    else:
                        u_hat = self.fft(self.u_prev)
                        u_hat *= self.exp_L
                        u_hat *= self.dealiasing_mask
                        u_lin = self.ifft(u_hat)
                        u_nl = self.apply_nonlinear(u_lin)
                        u_new = u_lin + u_nl
    
                u_new = u_new + source_contribution
                self.apply_boundary(u_new)
                self.u_prev = u_new
    
            # Second-order in time
            elif self.temporal_order == 2:
                if self.has_psi:
                    # === LEAP-FROG (explicit 2nd-order centered scheme) ===
            
                    # Compute spectral multiplier symbol_vals on first call
                    if step == 0:
                        self.symbol_vals = self.compute_combined_symbol()
            
                    # 1. FFT of u_prev (u^n)
                    u_hat = self.fft(self.u_prev)
            
                    # 2. Apply spectral operator
                    Lu_hat = -self.symbol_vals * u_hat
                    Lu_prev = self.ifft(Lu_hat)
            
                    # 3. Leap-Frog update: u^{n+1} = 2u^n - u^{n-1} + dt² L(u^n)
                    u_new = 2 * self.u_prev - self.u_prev2 + self.dt**2 * Lu_prev
            
                    # 4. Add optional nonlinear and source terms
                    rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                    u_new += self.dt**2 * (rhs_nl + source_contribution)
            
                    # 5. Enforce boundary conditions and update
                    self.apply_boundary(u_new)
                    self.u_prev2 = self.u_prev
                    self.u_prev = u_new

                else:
                    if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                        u_new, v_new = self.step_ETD_RK4_order2(self.u_prev, self.v_prev)
                    else:
                        u_hat = self.fft(self.u_prev)
                        v_hat = self.fft(self.v_prev)
    
                        u_new_hat = (self.cos_omega_dt * u_hat +
                                     self.sin_omega_dt * self.inv_omega * v_hat)
                        v_new_hat = (-self.omega_val * self.sin_omega_dt * u_hat +
                                      self.cos_omega_dt * v_hat)
    
                        u_new = self.ifft(u_new_hat)
                        v_new = self.ifft(v_new_hat)
    
                        u_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                        v_nl = self.apply_nonlinear(self.v_prev, is_v=True)
    
                        u_new += (u_nl + source_contribution) * (self.dt**2) / 2
                        v_new += (u_nl + source_contribution) * self.dt
    
                    self.apply_boundary(u_new)
                    self.apply_boundary(v_new)
                    self.u_prev = u_new
                    self.v_prev = v_new
    
            # Save current state
            if step % save_interval == 0:
                self.frames.append(self.u_prev.copy())
    
            # Energy monitoring only in linear case without psiOp
            if self.temporal_order == 2 and not self.has_psi:
                E = self.compute_energy()
                self.energy_history.append(E)

        
    def step_ETD_RK4(self, u):
        """
        Perform one ETD-RK4 time step for first-order time PDEs.
        
        Args:
            u (np.ndarray): Current solution in real space
        
        Returns:
            np.ndarray: Updated solution in real space
        """
        dt = self.dt
        L_fft = self.L(self.KX) if self.dim == 1 else self.L(self.KX, self.KY)
    
        E  = np.exp(dt * L_fft)
        E2 = np.exp(dt * L_fft / 2)
    
        def phi1(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1) / z, 1.0)
    
        def phi2(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1 - z) / z**2, 0.5)
    
        phi1_dtL = phi1(dt * L_fft)
        phi2_dtL = phi2(dt * L_fft)
    
        fft = self.fft
        ifft = self.ifft
    
        u_hat = fft(u)
        N1 = fft(self.apply_nonlinear(u))
    
        a = ifft(E2 * (u_hat + 0.5 * dt * N1 * phi1_dtL))
        N2 = fft(self.apply_nonlinear(a))
    
        b = ifft(E2 * (u_hat + 0.5 * dt * N2 * phi1_dtL))
        N3 = fft(self.apply_nonlinear(b))
    
        c = ifft(E * (u_hat + dt * N3 * phi1_dtL))
        N4 = fft(self.apply_nonlinear(c))
    
        u_new_hat = E * u_hat + dt * (
            N1 * phi1_dtL + 2 * (N2 + N3) * phi2_dtL + N4 * phi1_dtL
        ) / 6
    
        return ifft(u_new_hat)


    def step_ETD_RK4_order2(self, u, v):
        """
        Perform one ETD-RK4 time step for second-order time PDEs.
    
        Args:
            u (np.ndarray): Current solution in real space
            v (np.ndarray): Current derivative in real space
    
        Returns:
            tuple: Updated (u_new, v_new)
        """
        dt = self.dt
    
        L_fft = self.L(self.KX) if self.dim == 1 else self.L(self.KX, self.KY)
        fft = self.fft
        ifft = self.ifft
    
        def phi1(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1) / z, 1.0)
    
        def phi2(z):
            return np.where(np.abs(z) > 1e-12, (np.exp(z) - 1 - z) / z**2, 0.5)
    
        phi1_dtL = phi1(dt * L_fft)
        phi2_dtL = phi2(dt * L_fft)
    
        def rhs(u_val):
            return ifft(L_fft * fft(u_val)) + self.apply_nonlinear(u_val, is_v=False)
    
        # Stage A
        A = rhs(u)
        ua = u + 0.5 * dt * v
        va = v + 0.5 * dt * A
    
        # Stage B
        B = rhs(ua)
        ub = u + 0.5 * dt * va
        vb = v + 0.5 * dt * B
    
        # Stage C
        C = rhs(ub)
        uc = u + dt * vb
        vc = v + dt * C
    
        # Stage D
        D = rhs(uc)
    
        # Final update
        u_new = u + dt * v + (dt**2 / 6.0) * (A + 2*B + 2*C + D)
        v_new = v + (dt / 6.0) * (A + 2*B + 2*C + D)
    
        return u_new, v_new

    def compute_combined_symbol(self):
        """
        Evaluates the weighted sum of pseudo-differential symbols on the grid.
        Returns a numpy array (complex128) of the same shape as the spectral grid.
        """
        from sympy import N
    
        if not hasattr(self, 'psi_ops'):
            raise AttributeError("psi_ops not defined")
    
        shape = self.KX.shape if self.dim == 2 else self.KX.shape
        symbol_vals = np.zeros(shape, dtype=np.complex128)
    
        for coeff_sym, psi in self.psi_ops:
            coeff = complex(N(coeff_sym))
            raw = psi.evaluate(
                self.X,
                self.Y if self.dim == 2 else None,
                self.KX,
                self.KY if self.dim == 2 else None
            )
    
            flat = list(raw.flat)
            values = [complex(N(v)) for v in flat]
            sym_np = np.array(values, dtype=np.complex128).reshape(raw.shape)
    
            symbol_vals += coeff * sym_np
    
        return symbol_vals

    def check_cfl_condition(self):
        """
        Check the CFL condition based on group velocity for second-order PDEs.
        """
        print("\n*****************")
        print("* CFL condition *")
        print("*****************\n")

        cfl_factor = 0.5  # Safety factor
        
        if self.dim == 1:
            if self.temporal_order == 2 and hasattr(self, 'omega'):
                k_vals = self.kx
                omega_vals = np.real(self.omega(k_vals))
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_group = np.gradient(omega_vals, k_vals)
                max_speed = np.max(np.abs(v_group))
            else:
                max_speed = np.max(np.abs(np.imag(self.L(self.kx))))
            
            dx = self.Lx / self.Nx
            cfl_limit = cfl_factor * dx / max_speed if max_speed != 0 else np.inf
            
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")
    
        elif self.dim == 2:
            if self.temporal_order == 2 and hasattr(self, 'omega'):
                k_vals = self.kx
                omega_x = np.real(self.omega(k_vals, 0))
                omega_y = np.real(self.omega(0, k_vals))
                with np.errstate(divide='ignore', invalid='ignore'):
                    v_group_x = np.gradient(omega_x, k_vals)
                    v_group_y = np.gradient(omega_y, k_vals)
                max_speed_x = np.max(np.abs(v_group_x))
                max_speed_y = np.max(np.abs(v_group_y))
            else:
                max_speed_x = np.max(np.abs(np.imag(self.L(self.kx, 0))))
                max_speed_y = np.max(np.abs(np.imag(self.L(0, self.ky))))
            
            dx = self.Lx / self.Nx
            dy = self.Ly / self.Ny
            cfl_limit = cfl_factor / (max_speed_x / dx + max_speed_y / dy) if (max_speed_x + max_speed_y) != 0 else np.inf
            
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")
    
        else:
            raise NotImplementedError("Only 1D and 2D problems are supported.")


    def check_symbol_conditions(self, k_range=None, verbose=True):
        """
        Check strict conditions on self.L_symbolic:
            - Stability: Re(a(k)) ≤ 0
            - Dissipation: Re(a(k)) ≤ -δ |k|^p
            - Growth: |a(k)| ≤ C (1 + |k|)^m
    
        Works for both 1D and 2D cases.
        """
        import numpy as np
        from sympy import lambdify, symbols

        print("\n********************")
        print("* Symbol condition *")
        print("********************\n")

    
        if self.dim == 1:    
            if k_range is None:
                k_vals = np.linspace(-10, 10, 500)
            else:
                k_min, k_max, N = k_range
                k_vals = np.linspace(k_min, k_max, N)
    
            L_vals = self.L(k_vals)
            k_abs = np.abs(k_vals)
    
        elif self.dim == 2:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 100)
            else:
                k_min, k_max, N = k_range
                k_vals = np.linspace(k_min, k_max, N)
    
            KX, KY = np.meshgrid(k_vals, k_vals)
            L_vals = self.L(KX, KY)
            k_abs = np.sqrt(KX**2 + KY**2)
    
        else:
            raise ValueError("Only 1D and 2D dimensions are supported.")
    
        re_vals = np.real(L_vals)
        im_vals = np.imag(L_vals)
        abs_vals = np.abs(L_vals)
    
        # === Condition 1: Stability
        if np.any(re_vals > 1e-12):
            max_pos = np.max(re_vals)
            if verbose:
                print(f"❌ Stability violated: max Re(a(k)) = {max_pos}")
            print("Unstable symbol: Re(a(k)) > 0")
        elif verbose:
            print("✅ Spectral stability satisfied: Re(a(k)) ≤ 0")
    
        # === Condition 2: Dissipation
        mask = k_abs > 2
        if np.any(mask):
            re_decay = re_vals[mask]
            expected_decay = -0.01 * k_abs[mask]**2
            if np.any(re_decay > expected_decay + 1e-6):
                if verbose:
                    print("⚠️ Insufficient high-frequency dissipation")
            else:
                if verbose:
                    print("✅ Proper high-frequency dissipation")
    
        # === Condition 3: Growth
        growth_ratio = abs_vals / (1 + k_abs)**4
        if np.max(growth_ratio) > 100:
            if verbose:
                print(f"⚠️ Symbol grows rapidly: |a(k)| ≳ |k|^4")
        else:
            if verbose:
                print("✅ Reasonable spectral growth")
    
        if verbose:
            print("✔ Symbol analysis completed.")

    def analyze_wave_propagation(self):
        """
        Analyze wave propagation properties:
        - Dispersion relation ω(k)
        - Phase velocity v_p(k) = ω/|k|
        - Group velocity v_g(k) = ∇ₖ ω(k)
        - Anisotropy (in 2D)
        """
        print("\n*****************************")
        print("* Wave propagation analysis *")
        print("*****************************\n")
        if not hasattr(self, 'omega_symbolic'):
            print("❌ omega_symbolic not defined. Only available for 2nd order in time.")
            return
    
        import matplotlib.pyplot as plt
        from sympy import lambdify
        import numpy as np
    
        if self.dim == 1:
            k = self.k_symbols[0]
            omega_func = lambdify(k, self.omega_symbolic, 'numpy')
    
            k_vals = np.linspace(-10, 10, 1000)
            omega_vals = omega_func(k_vals)
    
            with np.errstate(divide='ignore', invalid='ignore'):
                v_phase = np.where(k_vals != 0, omega_vals / k_vals, 0.0)
    
            dk = k_vals[1] - k_vals[0]
            v_group = np.gradient(omega_vals, dk)
    
            plt.figure(figsize=(10, 6))
            plt.plot(k_vals, omega_vals, label=r'$\omega(k)$')
            plt.plot(k_vals, v_phase, label=r'$v_p(k)$')
            plt.plot(k_vals, v_group, label=r'$v_g(k)$')
            plt.title("1D Wave Propagation Analysis")
            plt.xlabel("k")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        elif self.dim == 2:
            kx, ky = self.k_symbols
            omega_func = lambdify((kx, ky), self.omega_symbolic, 'numpy')
    
            k_vals = np.linspace(-10, 10, 200)
            KX, KY = np.meshgrid(k_vals, k_vals)
            K_mag = np.sqrt(KX**2 + KY**2)
            K_mag[K_mag == 0] = 1e-8  # Avoid division by 0
    
            omega_vals = omega_func(KX, KY)
            v_phase = np.real(omega_vals) / K_mag
    
            dk = k_vals[1] - k_vals[0]
            domega_dx = np.gradient(omega_vals, dk, axis=0)
            domega_dy = np.gradient(omega_vals, dk, axis=1)
            v_group_norm = np.sqrt(np.abs(domega_dx)**2 + np.abs(domega_dy)**2)
    
            fig, axs = plt.subplots(1, 3, figsize=(18, 5))
            im0 = axs[0].imshow(np.real(omega_vals), extent=[-10, 10, -10, 10],
                                origin='lower', cmap='viridis')
            axs[0].set_title(r'$\omega(k_x, k_y)$')
            plt.colorbar(im0, ax=axs[0])
    
            im1 = axs[1].imshow(v_phase, extent=[-10, 10, -10, 10],
                                origin='lower', cmap='plasma')
            axs[1].set_title(r'$v_p(k_x, k_y)$')
            plt.colorbar(im1, ax=axs[1])
    
            im2 = axs[2].imshow(v_group_norm, extent=[-10, 10, -10, 10],
                                origin='lower', cmap='inferno')
            axs[2].set_title(r'$|v_g(k_x, k_y)|$')
            plt.colorbar(im2, ax=axs[2])
    
            for ax in axs:
                ax.set_xlabel(r'$k_x$')
                ax.set_ylabel(r'$k_y$')
                ax.set_aspect('equal')
    
            plt.tight_layout()
            plt.show()
    
        else:
            print("❌ Only 1D and 2D wave analysis supported.")
        
    def plot_symbol(self, component="abs", k_range=None, cmap="viridis"):
        """
        Visualise le symbole L_symbolic en 1D ou 2D.
    
        Args:
            component: 'abs', 're', ou 'im'
            k_range: (kmin, kmax, N), optionnel
            cmap: colormap matplotlib (2D)
        """
        print("\n*******************")
        print("* Symbol plotting *")
        print("*******************\n")
        
        assert component in ("abs", "re", "im"), "component must be 'abs', 're' or 'im'"
        
    
        if self.dim == 1:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 1000)
            else:
                kmin, kmax, N = k_range
                k_vals = np.linspace(kmin, kmax, N)
            L_vals = self.L(k_vals)
    
            if component == "re":
                vals = np.real(L_vals)
                label = "Re[a(k)]"
            elif component == "im":
                vals = np.imag(L_vals)
                label = "Im[a(k)]"
            else:
                vals = np.abs(L_vals)
                label = "|a(k)|"
    
            plt.plot(k_vals, vals)
            plt.xlabel("k")
            plt.ylabel(label)
            plt.title(f"Spectral symbol: {label}")
            plt.grid(True)
            plt.show()
    
        elif self.dim == 2:
            if k_range is None:
                k_vals = np.linspace(-10, 10, 300)
            else:
                kmin, kmax, N = k_range
                k_vals = np.linspace(kmin, kmax, N)
    
            KX, KY = np.meshgrid(k_vals, k_vals)
            L_vals = self.L(KX, KY)
    
            if component == "re":
                Z = np.real(L_vals)
                title = "Re[a(kx, ky)]"
            elif component == "im":
                Z = np.imag(L_vals)
                title = "Im[a(kx, ky)]"
            else:
                Z = np.abs(L_vals)
                title = "|a(kx, ky)|"
    
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
        
            surf = ax.plot_surface(KX, KY, Z, cmap=cmap, edgecolor='none', antialiased=True)
            fig.colorbar(surf, ax=ax, shrink=0.6)
        
            ax.set_xlabel("kx")
            ax.set_ylabel("ky")
            ax.set_zlabel(title)
            ax.set_title(f"2D spectral symbol: {title}")
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError("Only 1D and 2D supported.")

    def compute_energy(self):
        """
        Compute total energy of the wave equation:
            E(t) = 1/2 ∫ [ (∂_t u)^2 + |L^{1/2} u|^2 ] dx
        Supports 1D and 2D cases. Only meaningful if temporal_order == 2.
        """
        if self.temporal_order != 2 or self.v_prev is None:
            return None
    
        u = self.u_prev
        v = self.v_prev
    
        # Fourier transform of u
        u_hat = self.fft(u)
    
        if self.dim == 1:
            # 1D case
            L_vals = self.L(self.KX)
            sqrt_L = np.sqrt(np.abs(L_vals))
            Lu_hat = sqrt_L * u_hat  # Apply sqrt(|L(k)|) in Fourier space
            Lu = self.ifft(Lu_hat)
    
            dx = self.Lx / self.Nx
            energy_density = 0.5 * (np.abs(v)**2 + np.abs(Lu)**2)
            total_energy = np.sum(energy_density) * dx
    
        elif self.dim == 2:
            # 2D case
            L_vals = self.L(self.KX, self.KY)
            sqrt_L = np.sqrt(np.abs(L_vals))
            Lu_hat = sqrt_L * u_hat
            Lu = self.ifft(Lu_hat)
    
            dx = self.Lx / self.Nx
            dy = self.Ly / self.Ny
            energy_density = 0.5 * (np.abs(v)**2 + np.abs(Lu)**2)
            total_energy = np.sum(energy_density) * dx * dy
    
        else:
            raise ValueError("Unsupported dimension for u.")
    
        return total_energy

    def plot_energy(self, log=False):
        """
        Plot the evolution of energy over time.
        Supports both 1D and 2D wave simulations (requires temporal_order=2).
        
        Args:
            log (bool): if True, plot energy on a logarithmic scale.
        """
        if not hasattr(self, 'energy_history') or not self.energy_history:
            print("No energy data recorded. Call compute_energy() within solve().")
            return
    
        import matplotlib.pyplot as plt
    
        # Time vector for plotting
        t = np.linspace(0, self.Lt, len(self.energy_history))
    
        # Create the figure
        plt.figure(figsize=(6, 4))
        if log:
            plt.semilogy(t, self.energy_history, label="Energy (log scale)")
        else:
            plt.plot(t, self.energy_history, label="Energy")
    
        # Axis labels and title
        plt.xlabel("Time")
        plt.ylabel("Total energy")
        plt.title("Energy evolution ({}D)".format(self.dim))
    
        # Display options
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    
    def animate(self, component='abs', overlay='contour'):
        """
        Create an animated plot of the solution evolution.
        Args:
            component (str): 'real', 'imag', 'abs', or 'angle'.
            overlay (str): Only used in 2D: 'contour' or 'front'.
        """
        def get_component(u):
            if component == 'real':
                return np.real(u)
            elif component == 'imag':
                return np.imag(u)
            elif component == 'abs':
                return np.abs(u)
            elif component == 'angle':
                return np.angle(u)
            else:
                raise ValueError("Invalid component")

        print("\n*********************")
        print("* Solution plotting *")
        print("*********************\n")
        
        # === Calculate time vector of stored frames ===
        save_interval = max(1, self.Nt // self.n_frames)
        frame_times = np.arange(0, self.Lt + self.dt, save_interval * self.dt)
        
        # === Target times for animation ===
        target_times = np.linspace(0, self.Lt, self.n_frames)
        
        # Map target times to nearest frame indices
        frame_indices = [np.argmin(np.abs(frame_times - t)) for t in target_times]
    
        if self.dim == 1:
            fig, ax = plt.subplots()
            line, = ax.plot(self.X, get_component(self.frames[0]))
            ax.set_ylim(np.min(self.frames[0]), np.max(self.frames[0]))
            ax.set_xlabel('x')
            ax.set_ylabel(f'{component} of u')
            ax.set_title('Initial condition')
            plt.tight_layout()
            plt.show()
    
            def update(frame_number):
                frame = frame_indices[frame_number]
                ydata = get_component(self.frames[frame])
                ydata_real = np.real(ydata) if np.iscomplexobj(ydata) else ydata
                line.set_ydata(ydata_real)
                ax.set_ylim(np.min(ydata_real), np.max(ydata_real))
                current_time = target_times[frame_number]
                ax.set_title(f't = {current_time:.2f}')
                return line,
    
            ani = FuncAnimation(fig, update, frames=len(target_times), interval=50)
            return ani
    
        else:  # dim == 2
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'{component.title()} of u')
            ax.set_title('Initial condition')
    
            data0 = get_component(self.frames[0])
            surf = [ax.plot_surface(self.X, self.Y, data0, cmap='viridis')]
            plt.tight_layout()
            plt.show()
    
            def update(frame_number):
                frame = frame_indices[frame_number]
                current_data = get_component(self.frames[frame])
                z_offset = np.max(current_data) + 0.05 * (np.max(current_data) - np.min(current_data))
    
                ax.clear()
                surf[0] = ax.plot_surface(self.X, self.Y, current_data,
                                          cmap='viridis', vmin=-1, vmax=1 if component != 'angle' else np.pi)
    
                if overlay == 'contour':
                    ax.contour(self.X, self.Y, current_data, levels=10, cmap='cool', offset=z_offset)
    
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_zlabel(f'{component.title()} of u')
                current_time = target_times[frame_number]
                ax.set_title(f'Solution at t = {current_time:.2f}')
                return surf
    
            ani = FuncAnimation(fig, update, frames=len(target_times), interval=50)
            return ani

    def test(self, u_exact, t_eval=None, norm='relative', threshold=1e-2, plot=True, component='real'):
        """
        Test the solver by comparing the numerical solution to an exact solution.
        
        Args:
            u_exact (callable): Exact solution function u(x[, y], t).
            t_eval (float, optional): Time at which to compare (default: final time).
            norm (str): 'relative' or 'absolute'.
            threshold (float): Threshold for the error.
            plot (bool): Whether to display plots.
            component (str): 'real', 'imag', or 'abs' for comparison.
        """
        if t_eval is None:
            t_eval = self.Lt
    
        # Find the closest frame index corresponding to time t_eval
        save_interval = max(1, self.Nt // self.n_frames)
        frame_times = np.arange(0, self.Lt + self.dt, save_interval * self.dt)  # All possible times
        frame_index = np.argmin(np.abs(frame_times - t_eval))  # Closest index
        actual_t = frame_times[frame_index]
        print(f"Closest available time to t_eval={t_eval}: {actual_t}")
    
        if frame_index >= len(self.frames):
            raise ValueError(f"Time t = {t_eval} exceeds simulation duration.")
        
        u_num = self.frames[frame_index]
    
        # Compute the exact solution at the actual time
        if self.dim == 1:
            u_ex = u_exact(self.X, actual_t)
        elif self.dim == 2:
            u_ex = u_exact(self.X, self.Y, actual_t)
        else:
            raise ValueError("Unsupported dimension.")
    
        # Select component for comparison
        if component == 'real':
            diff = np.real(u_num) - np.real(u_ex)
            ref = np.real(u_ex)
        elif component == 'imag':
            diff = np.imag(u_num) - np.imag(u_ex)
            ref = np.imag(u_ex)
        elif component == 'abs':
            diff = np.abs(u_num) - np.abs(u_ex)
            ref = np.abs(u_ex)
        else:
            raise ValueError("Invalid component.")
    
        # Compute error
        if norm == 'relative':
            error = np.linalg.norm(diff) / np.linalg.norm(ref)
        elif norm == 'absolute':
            error = np.linalg.norm(diff)
        else:
            raise ValueError("Unknown norm type.")
    
        print(f"Test error at t = {actual_t}: {error:.3e}")
        assert error < threshold, f"Error too large at t = {actual_t}: {error:.3e}"
    
        # Optional plots
        if plot:
            if self.dim == 1:
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(self.X, np.real(u_num), label='Numerical')
                plt.plot(self.X, np.real(u_ex), label='Exact', linestyle='--')
                plt.legend()
                plt.title(f'Solution at t = {actual_t}, error = {error:.2e}')
                plt.grid()
    
                plt.subplot(2, 1, 2)
                plt.plot(self.X, np.abs(diff), color='red')
                plt.title('Absolute Error')
                plt.grid()
                plt.tight_layout()
                plt.show()
            else:
                plt.figure(figsize=(15, 5))
                plt.subplot(1, 3, 1)
                plt.title("Numerical Solution")
                plt.imshow(np.abs(u_num), origin='lower', extent=[0, self.Lx, 0, self.Ly], cmap='viridis')
                plt.colorbar()
    
                plt.subplot(1, 3, 2)
                plt.title("Exact Solution")
                plt.imshow(np.abs(u_ex), origin='lower', extent=[0, self.Lx, 0, self.Ly], cmap='viridis')
                plt.colorbar()
    
                plt.subplot(1, 3, 3)
                plt.title(f"Error (Norm = {error:.2e})")
                plt.imshow(np.abs(diff), origin='lower', extent=[0, self.Lx, 0, self.Ly], cmap='inferno')
                plt.colorbar()
                plt.tight_layout()
                plt.show()
