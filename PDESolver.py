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
    radsimp, ratsimp, cancel,
    Lambda, Piecewise, Basic, degree, Pow, preorder_traversal,
    sqrt, I,  pi, series, oo, 
    re, im, arg, Abs, conjugate, 
    sin, cos, tan, cot, sec, csc, sinc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
    asinh, acosh, atanh, acoth, asech, acsch,
    exp, ln, factorial, 
    diff, Derivative, integrate, 
    fourier_transform, inverse_fourier_transform,
)
from sympy.core.function import AppliedUndef
from IPython.display import display
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from functools import partial
from misc import * 
from scipy.integrate import solve_ivp
from IPython.display import display
from ipywidgets import interact, FloatSlider, Dropdown


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
        self.vars_x = vars_x

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
        """
        Evaluate the symbol on a spatial-frequency grid.

        Parameters
        ----------
        X, Y : np.ndarray
            Spatial grid coordinates (Y is ignored in 1D).
        KX, KY : np.ndarray
            Frequency grid coordinates (KY is ignored in 1D).
        cache : bool, default=True
            Whether to use/cached computed values.

        Returns
        -------
        np.ndarray
            Evaluated symbol values on the grid.
        """
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
        """
        Clear cached symbol evaluations.
        """        
        self.symbol_cached = None

    def principal_symbol(self, order=1):
        """
        Return the homogeneous principal symbol of the operator.
        
        Parameters
        ----------
        order : int
            Degree of homogeneity in |ξ| (or (ξ, η)).
        
        Returns
        -------
        sympy expression
            Leading homogeneous part of the symbol.
        """
        p = self.expr
        if self.dim == 1:
            xi = symbols('xi', real=True)
            return simplify(series(p, xi, oo, n=order).removeO())
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True)
            # Expansion radiale homogène : on fixe (ξ, η) = ρ (cosθ, sinθ)
            rho, theta = symbols('rho theta', real=True)
            p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
            expansion = series(p_rho, rho, oo, n=order).removeO()
            # Revenir à (ξ, η)
            expansion_cart = expansion.subs({rho: sqrt(xi**2 + eta**2),
                                             cos(theta): xi / sqrt(xi**2 + eta**2),
                                             sin(theta): eta / sqrt(xi**2 + eta**2)})
            return simplify(expansion_cart)
            
    def symbol_order(self, max_order=10, tol=1e-3):
        """
        Estimate the order (degree of homogeneity) of the pseudo-differential symbol.
    
        Parameters
        ----------
        max_order : int
            Maximum order to test.
        tol : float
            Tolerance for considering a term non-zero.
    
        Returns
        -------
        int or None
            Estimated order (homogeneity degree), or None if not determined.
        """
        from sympy import symbols, simplify, series, oo, sqrt, cos, sin, expand
    
        p = self.expr
    
        if self.dim == 1:
            xi = symbols('xi', real=True)
            try:
                s = simplify(series(p, xi, oo, n=max_order).removeO())
                terms = s.as_ordered_terms()
                for term in reversed(terms):
                    poly = term.as_poly(xi)
                    if poly is None:
                        continue
                    degree = poly.degree()
                    coeff = poly.coeff_monomial(xi**degree)
                    if coeff.free_symbols:
                        continue  # dépend encore de x, on ignore
                    if abs(float(coeff.evalf())) > tol:
                        return degree
            except Exception as e:
                print(f"Order estimation failed: {e}")
            return None
    
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True)
            rho, theta = symbols('rho theta', real=True)
            try:
                p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
                s = simplify(series(p_rho, rho, oo, n=max_order).removeO())
                terms = s.as_ordered_terms()
                for term in reversed(terms):
                    poly = term.as_poly(rho)
                    if poly is None:
                        continue
                    degree = poly.degree()
                    coeff = poly.coeff_monomial(rho**degree)
                    if coeff.free_symbols:
                        continue
                    if abs(float(coeff.evalf())) > tol:
                        return degree
            except Exception as e:
                print(f"2D Order estimation failed: {e}")
            return None
    
        else:
            raise NotImplementedError("Only 1D and 2D are supported.")

    def asymptotic_expansion(self, order=3):
        """
        Asymptotic expansion of the symbol in |ξ| → ∞.
    
        Parameters
        ----------
        order : int
            Order up to which the expansion is computed.
    
        Returns
        -------
        sympy expression
            Expansion up to order `order` in 1/|ξ|.
        """
        p = self.expr
    
        if self.dim == 1:
            xi = symbols('xi', real=True)
    
            try:
                # Cas exp(f(x, xi))
                if p.func == exp and len(p.args) == 1:
                    arg = p.args[0]
                    arg_series = series(arg, xi, oo, n=order).removeO()
                    # Développer exp(arg_series)
                    expanded = series(expand(exp(arg_series)), xi, oo, n=order).removeO()
                    return simplify(expanded)
                else:
                    return simplify(series(p, xi, oo, n=order).removeO())
    
            except Exception as e:
                print(f"Warning: expansion failed: {e}")
                return p
    
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True)
            rho, theta = symbols('rho theta', real=True)
            from sympy import cos, sin, sqrt
    
            # Passer en coordonnées polaires
            p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
    
            try:
                if p_rho.func == exp and len(p_rho.args) == 1:
                    arg = p_rho.args[0]
                    arg_series = series(arg, rho, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), rho, oo, n=order).removeO()
                else:
                    expanded = series(p_rho, rho, oo, n=order).removeO()
    
                # Revenir à (xi, eta)
                norm = sqrt(xi**2 + eta**2)
                expansion_cart = expanded.subs({
                    rho: norm,
                    cos(theta): xi / norm,
                    sin(theta): eta / norm
                })
    
                return simplify(expansion_cart)
    
            except Exception as e:
                print(f"Warning: 2D expansion failed: {e}")
                return p

    def compose_asymptotic(self, other, order=1):
        """
        Compose self with another PseudoDifferentialOperator via asymptotic expansion.
        """
        assert self.dim == other.dim, "Operator dimensions must match"
        p, q = self.expr, other.expr
    
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            result = 0
            for n in range(order + 1):
                term = (1 / factorial(n)) * diff(p, xi, n) * diff(q, x, n) * (1j)**(-n)
                result += term
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            result = 0
            for n in range(order + 1):
                for i in range(n + 1):
                    j = n - i
                    term = (1 / (factorial(i) * factorial(j))) * \
                           diff(p, xi, i, eta, j) * diff(q, x, i, y, j) * (1j)**(-n)
                    result += term
    
        return result

    def right_inverse_asymptotic(self, order=1):
        """
        Construct right formal inverse R such that P \circ R = I + O(xi^{-order})
        """
        p = self.expr
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            r = 1 / p.subs(xi, xi)  # r0
            R = r
            for n in range(1, order + 1):
                term = 0
                for k in range(1, n + 1):
                    coeff = (1j)**(-k) / factorial(k)
                    inner = diff(p, xi, k) * diff(R, x, k)
                    term += coeff * inner
                R = R - r * term
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            r = 1 / p.subs({xi: xi, eta: eta})
            R = r
            for n in range(1, order + 1):
                term = 0
                for k1 in range(n + 1):
                    for k2 in range(n + 1 - k1):
                        if k1 + k2 == 0: continue
                        coeff = (1j)**(-(k1 + k2)) / (factorial(k1) * factorial(k2))
                        dp = diff(p, xi, k1, eta, k2)
                        dR = diff(R, x, k1, y, k2)
                        term += coeff * dp * dR
                R = R - r * term
        return R

    def left_inverse_asymptotic(self, order=1):
        """
        Construct left formal inverse L such that L \circ P = I + O(xi^{-order})
        """
        p = self.expr
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True)
            l = 1 / p.subs(xi, xi)
            L = l
            for n in range(1, order + 1):
                term = 0
                for k in range(1, n + 1):
                    coeff = (1j)**(-k) / factorial(k)
                    inner = diff(L, xi, k) * diff(p, x, k)
                    term += coeff * inner
                L = L - term * l
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            l = 1 / p.subs({xi: xi, eta: eta})
            L = l
            for n in range(1, order + 1):
                term = 0
                for k1 in range(n + 1):
                    for k2 in range(n + 1 - k1):
                        if k1 + k2 == 0: continue
                        coeff = (1j)**(-(k1 + k2)) / (factorial(k1) * factorial(k2))
                        dp = diff(p, x, k1, y, k2)
                        dL = diff(L, xi, k1, eta, k2)
                        term += coeff * dL * dp
                L = L - term * l
        return L

    def formal_adjoint(self):
        """
        Compute the formal adjoint of the pseudo-differential operator.
        
        Returns
        -------
        sympy expression
            Symbol of the adjoint operator P^*.
        """
        p = self.expr
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
            p_star = conjugate(p)
            p_star = simplify(series(p_star, xi, oo, n=6).removeO())
            return p_star
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
            p_star = conjugate(p)
            p_star = simplify(series(p_star, sqrt(xi**2 + eta**2), oo, n=6).removeO())
            return p_star

    def symplectic_flow(self):
        """
        Compute the Hamiltonian vector field of the symbol.
    
        Returns
        -------
        dict
            Dictionary with 'dx/dt', 'dξ/dt' (and optionally y, η).
        """
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi')
            return {
                'dx/dt': diff(self.expr, xi),
                'dxi/dt': -diff(self.expr, x)
            }
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta')
            return {
                'dx/dt': diff(self.expr, xi),
                'dy/dt': diff(self.expr, eta),
                'dxi/dt': -diff(self.expr, x),
                'deta/dt': -diff(self.expr, y)
            }

    def is_elliptic_numerically(self, x_grid, xi_grid, threshold=1e-8):
        """
        Check if the symbol is elliptic on a grid of (x, xi) or (x, y, xi, eta),
        with resampling to avoid memory explosion in 2D.
    
        Parameters
        ----------
        x_grid : ndarray
            1D or 2D spatial grid (x or (x, y)).
        xi_grid : ndarray
            1D or 2D frequency grid (xi or (xi, eta)).
        threshold : float
            Minimum allowed magnitude of the symbol.
    
        Returns
        -------
        bool
            True if elliptic on grid, False otherwise.
        """
        RESAMPLE_SIZE = 32  # Taille réduite pour éviter l'explosion mémoire
    
        if self.dim == 1:
            x_vals = x_grid
            xi_vals = xi_grid
            # Rééchantillonnage si nécessaire
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
    
            X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
            symbol_vals = self.p_func(X, XI)
    
        elif self.dim == 2:
            x_vals, y_vals = x_grid
            xi_vals, eta_vals = xi_grid
    
            # Rééchantillonnage spatial
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(y_vals) > RESAMPLE_SIZE:
                y_vals = np.linspace(y_vals.min(), y_vals.max(), RESAMPLE_SIZE)
    
            # Rééchantillonnage fréquentiel
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
            if len(eta_vals) > RESAMPLE_SIZE:
                eta_vals = np.linspace(eta_vals.min(), eta_vals.max(), RESAMPLE_SIZE)
    
            X, Y, XI, ETA = np.meshgrid(x_vals, y_vals, xi_vals, eta_vals, indexing='ij')
            symbol_vals = self.p_func(X, Y, XI, ETA)
    
        else:
            raise NotImplementedError("Only 1D and 2D supported")
    
        min_abs_val = np.min(np.abs(symbol_vals))
        return min_abs_val > threshold

    def is_self_adjoint(self, tol=1e-10):
        """
        Check whether the operator is formally self-adjoint.
    
        Returns
        -------
        bool
        """
        p = self.expr
        p_star = self.formal_adjoint()
        return simplify(p - p_star).equals(0)

    def is_homogeneous(self, degree):
        """
        Check whether the symbol is homogeneous of a given degree in (ξ, η).
    
        Parameters
        ----------
        degree : int or float
    
        Returns
        -------
        bool
        """
        if self.dim == 1:
            xi = symbols('xi', real=True)
            scaling = self.expr.subs(xi, symbols('λ') * xi)
            return simplify(scaling / self.expr - symbols('λ')**degree).equals(0)
        else:
            xi, eta = symbols('xi eta', real=True)
            lam = symbols('λ')
            scaled = self.expr.subs({xi: lam * xi, eta: lam * eta})
            return simplify(scaled / self.expr - lam**degree).equals(0)

    def visualize_wavefront(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """
        Visualize the wavefront set of the symbol.

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            Spatial grids.
        xi_grid, eta_grid : np.ndarray
            Frequency grids.
        xi0, eta0 : float
            Fixed frequency values for visualization.
        """
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
        """
        Visualize the fiber structure of the symbol.

        Parameters
        ----------
        x_grid, xi_grid : np.ndarray
            Spatial and frequency grids.
        x0, y0 : float
            Spatial position where to visualize the fiber.
        """
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
        """
        Plot the amplitude of the symbol.

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            Spatial grids.
        xi_grid, eta_grid : np.ndarray
            Frequency grids.
        xi0, eta0 : float
            Fixed frequency values for visualization.
        """
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
        """
        Plot the phase of the symbol.

        Parameters
        ----------
        x_grid, y_grid : np.ndarray
            Spatial grids.
        xi_grid, eta_grid : np.ndarray
            Frequency grids.
        xi0, eta0 : float
            Fixed frequency values for visualization.
        """
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
        """
        Plot the characteristic set of the symbol.

        Parameters
        ----------
        x_grid, xi_grid : np.ndarray
            Spatial and frequency grids.
        x0, y0 : float
            Spatial position where to analyze the characteristic set.
        """
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
        """
        Visualize dynamic wave propagation over time.

        Parameters
        ----------
        x_grid, t_grid : np.ndarray
            Spatial and temporal grids.
        y_grid : np.ndarray, optional
            Second spatial dimension (for 2D).
        xi0, eta0 : float
            Initial frequency values for wave propagation.
        """
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

    def plot_hamiltonian_flow(self, x0=0.0, xi0=5.0, y0=0.0, eta0=0.0, tmax=1.0, n_steps=100):
        """
        Integrate and plot the Hamiltonian flow (bicharacteristics of the symbol).
    
        Parameters
        ----------
        x0, xi0 : float
            Initial spatial position and frequency (1D).
        y0, eta0 : float
            (2D only) Initial y and η.
        tmax : float
            Final integration time.
        n_steps : int
            Number of time steps.
        """
        from scipy.integrate import solve_ivp
        import matplotlib.pyplot as plt
        from sympy import simplify, symbols, lambdify, im
    
        def make_real(expr):
            """Return the real part of an expression (if complex)."""
            return simplify(expr.as_real_imag()[0])
    
        H = self.symplectic_flow()
    
        if any(im(H[k]) != 0 for k in H):
            print("⚠️ The Hamiltonian field is complex. Only the real part is used for integration.")
    
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
    
            dxdt_expr = make_real(H['dx/dt'])
            dxidt_expr = make_real(H['dxi/dt'])
    
            dxdt = lambdify((x, xi), dxdt_expr, 'numpy')
            dxidt = lambdify((x, xi), dxidt_expr, 'numpy')
    
            def hamilton(t, Y):
                x, xi = Y
                return [dxdt(x, xi), dxidt(x, xi)]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, xi0], t_eval=np.linspace(0, tmax, n_steps))
            x_vals, xi_vals = sol.y
    
            plt.plot(x_vals, xi_vals)
            plt.xlabel("x")
            plt.ylabel("ξ")
            plt.title("Hamiltonian Flow in Phase Space (1D)")
            plt.grid(True)
            plt.show()
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
    
            dxdt = lambdify((x, y, xi, eta), make_real(H['dx/dt']), 'numpy')
            dydt = lambdify((x, y, xi, eta), make_real(H['dy/dt']), 'numpy')
            dxidt = lambdify((x, y, xi, eta), make_real(H['dxi/dt']), 'numpy')
            detadt = lambdify((x, y, xi, eta), make_real(H['deta/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, y, xi, eta = Y
                return [
                    dxdt(x, y, xi, eta),
                    dydt(x, y, xi, eta),
                    dxidt(x, y, xi, eta),
                    detadt(x, y, xi, eta)
                ]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, y0, xi0, eta0], t_eval=np.linspace(0, tmax, n_steps))
            x_vals, y_vals, xi_vals, eta_vals = sol.y
    
            plt.plot(x_vals, y_vals, label='Position')
            plt.quiver(x_vals, y_vals, xi_vals, eta_vals, scale=20, width=0.003, alpha=0.5, color='r')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Hamiltonian Flow in Phase Space (2D)")
            plt.legend()
            plt.grid(True)
            plt.axis('equal')
            plt.show()


    def plot_symplectic_vector_field(self, xlim=(-2, 2), klim=(-5, 5), density=30):
        """
        Plot the symplectic (Hamiltonian) vector field (dx/dt, dxi/dt) for the symbol.
        """
        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')

        if self.dim != 1:
            raise NotImplementedError("Only 1D version implemented.")

        x, = self.vars_x
        xi = symbols('xi', real=True)
        H = self.symplectic_flow()
        dxdt = lambdify((x, xi), simplify(H['dx/dt']), 'numpy')
        dxidt = lambdify((x, xi), simplify(H['dxi/dt']), 'numpy')

        U = dxdt(X, XI)
        V = dxidt(X, XI)

        plt.quiver(X, XI, U, V, scale=10, width=0.005)
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Symplectic Vector Field (1D)")
        plt.grid(True)
        plt.show()

    def visualize_micro_support(self, xlim=(-2, 2), klim=(-10, 10), threshold=1e-3, density=300):
        """
        Visualize the micro-support: region in (x, ξ) where |symbol(x, ξ)| is small.
        """
        if self.dim != 1:
            raise NotImplementedError("Only 1D micro-support visualization implemented.")

        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
        Z = np.abs(self.p_func(X, XI))

        plt.contourf(X, XI, 1 / (Z + 1e-10), levels=100, cmap='inferno')
        plt.colorbar(label=r'$1/|p(x,\xi)|$')
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Micro-Support Estimate (1/|Symbol|)")
        plt.show()

    def group_velocity_field(self, xlim=(-2, 2), klim=(-10, 10), density=30):
        """
        Plot group velocity vector field \nabla_ξ p(x, ξ).
        """
        if self.dim != 1:
            raise NotImplementedError("Only 1D group velocity visualization implemented.")

        x, = self.vars_x
        xi = symbols('xi', real=True)
        dp_dxi = diff(self.expr, xi)
        grad_func = lambdify((x, xi), dp_dxi, 'numpy')

        x_vals = np.linspace(*xlim, density)
        xi_vals = np.linspace(*klim, density)
        X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
        V = grad_func(X, XI)

        plt.quiver(X, XI, np.ones_like(V), V, scale=10, width=0.004)
        plt.xlabel('x')
        plt.ylabel(r'$\xi$')
        plt.title("Group Velocity Field (1D)")
        plt.grid(True)
        plt.show()

    def animate_singularity(self, xi0=5.0, eta0=0.0, tmax=4.0, n_frames=20, projection=None):
        """
        Animate the motion of a singularity under the Hamiltonian flow.
    
        Parameters
        ----------
        xi0, eta0 : float
            Initial frequency values (eta0 ignored in 1D)
        tmax : float
            Total time of animation
        n_frames : int
            Number of frames in animation
        projection : str or None
            'position'   → show (x, y)
            'frequency'  → show (xi, eta)
            'phase'      → show (x, xi) or (x, eta)
            None         → default: 'phase' in 1D, 'position' in 2D
    
        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object for inline display
        """
        from scipy.integrate import solve_ivp
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        from matplotlib import rc
        from sympy import simplify, symbols, lambdify, im
    
        rc('animation', html='jshtml')
    
        def make_real(expr):
            return simplify(expr.as_real_imag()[0])
    
        H = self.symplectic_flow()
    
        if any(im(H[k]) != 0 for k in H):
            print("⚠️  The Hamiltonian field is complex. Only the real part is used for integration.")
    
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
    
            dxdt = lambdify((x, xi), make_real(H['dx/dt']), 'numpy')
            dxidt = lambdify((x, xi), make_real(H['dxi/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, xi = Y
                return [dxdt(x, xi), dxidt(x, xi)]
    
            sol = solve_ivp(hamilton, [0, tmax], [0.0, xi0], t_eval=np.linspace(0, tmax, n_frames))
            x_vals, xi_vals = sol.y
    
            if projection is None:
                projection = 'phase'
    
            fig, ax = plt.subplots()
            point, = ax.plot([], [], 'ro')
            traj, = ax.plot([], [], 'b--', lw=1, alpha=0.5)
    
            if projection == 'phase':
                ax.set_xlabel('x')
                ax.set_ylabel(r'$\xi$')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [xi_vals[i]])
                    traj.set_data(x_vals[:i+1], xi_vals[:i+1])
                    return point, traj
    
            elif projection == 'position':
                ax.set_xlabel('x')
                ax.set_ylabel('x')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(x_vals) - 1, np.max(x_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [x_vals[i]])
                    traj.set_data(x_vals[:i+1], x_vals[:i+1])
                    return point, traj
    
            elif projection == 'frequency':
                ax.set_xlabel(r'$\xi$')
                ax.set_ylabel(r'$\xi$')
                ax.set_xlim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
                ax.set_ylim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
    
                def update(i):
                    point.set_data([xi_vals[i]], [xi_vals[i]])
                    traj.set_data(xi_vals[:i+1], xi_vals[:i+1])
                    return point, traj
    
            else:
                raise ValueError("Invalid projection mode")
    
            ax.set_title(f"1D Singularity Flow ({projection})")
            ax.grid(True)
            ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
            plt.close(fig)
            return ani
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True)
    
            dxdt = lambdify((x, y, xi, eta), make_real(H['dx/dt']), 'numpy')
            dydt = lambdify((x, y, xi, eta), make_real(H['dy/dt']), 'numpy')
            dxidt = lambdify((x, y, xi, eta), make_real(H['dxi/dt']), 'numpy')
            detadt = lambdify((x, y, xi, eta), make_real(H['deta/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, y, xi, eta = Y
                return [
                    dxdt(x, y, xi, eta),
                    dydt(x, y, xi, eta),
                    dxidt(x, y, xi, eta),
                    detadt(x, y, xi, eta)
                ]
    
            sol = solve_ivp(hamilton, [0, tmax], [0.0, 0.0, xi0, eta0], t_eval=np.linspace(0, tmax, n_frames))
            x_vals, y_vals, xi_vals, eta_vals = sol.y
    
            if projection is None:
                projection = 'position'
    
            fig, ax = plt.subplots()
            point, = ax.plot([], [], 'ro')
            traj, = ax.plot([], [], 'b--', lw=1, alpha=0.5)
    
            if projection == 'position':
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(y_vals) - 1, np.max(y_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [y_vals[i]])
                    traj.set_data(x_vals[:i+1], y_vals[:i+1])
                    return point, traj
    
            elif projection == 'frequency':
                ax.set_xlabel(r'$\xi$')
                ax.set_ylabel(r'$\eta$')
                ax.set_xlim(np.min(xi_vals) - 1, np.max(xi_vals) + 1)
                ax.set_ylim(np.min(eta_vals) - 1, np.max(eta_vals) + 1)
    
                def update(i):
                    point.set_data([xi_vals[i]], [eta_vals[i]])
                    traj.set_data(xi_vals[:i+1], eta_vals[:i+1])
                    return point, traj
    
            elif projection == 'phase':
                ax.set_xlabel('x')
                ax.set_ylabel(r'$\eta$')
                ax.set_xlim(np.min(x_vals) - 1, np.max(x_vals) + 1)
                ax.set_ylim(np.min(eta_vals) - 1, np.max(eta_vals) + 1)
    
                def update(i):
                    point.set_data([x_vals[i]], [eta_vals[i]])
                    traj.set_data(x_vals[:i+1], eta_vals[:i+1])
                    return point, traj
    
            else:
                raise ValueError("Invalid projection mode")
    
            ax.set_title(f"2D Singularity Flow ({projection})")
            ax.grid(True)
            ax.axis('equal')
            ani = animation.FuncAnimation(fig, update, frames=n_frames, interval=50)
            plt.close(fig)
            return ani

    def interactive_symbol_analysis(pseudo_op,
                                    xlim=(-2, 2), ylim=(-2, 2),
                                    xi_range=(0.1, 5), eta_range=(-5, 5),
                                    density=100):
        dim = pseudo_op.dim
        expr = pseudo_op.expr
        vars_x = pseudo_op.vars_x
    
        mode_selector = Dropdown(
            options=[
                'Group Velocity Field',
                'Micro-Support (1/|p|)',
                'Symplectic Vector Field',
                'Symbol Amplitude',
                'Symbol Phase',
                'Cotangent Fiber',
                'Characteristic Set',
                'Wavefront Set',
                'Hamiltonian Flow',
            ],
            value='Group Velocity Field',
            description='Mode:'
        )
    
        x_vals = np.linspace(*xlim, density)
        if dim == 2:
            y_vals = np.linspace(*ylim, density)
    
        if dim == 1:
            x, = vars_x
            xi = symbols('xi', real=True)
            grad_func = lambdify((x, xi), diff(expr, xi), 'numpy')
            symplectic_func = lambdify((x, xi), [diff(expr, xi), -diff(expr, x)], 'numpy')
            symbol_func = lambdify((x, xi), expr, 'numpy')
    
            def plot_1d(mode, xi0, x0):
                X = x_vals[:, None]
    
                if mode == 'Group Velocity Field':
                    V = grad_func(X, xi0)
                    plt.quiver(X, V, np.ones_like(V), V, scale=10, width=0.004)
                    plt.title(f'Group Velocity Field at ξ={xi0:.2f}')
    
                elif mode == 'Micro-Support (1/|p|)':
                    Z = 1 / (np.abs(symbol_func(X, xi0)) + 1e-10)
                    plt.plot(x_vals, Z)
                    plt.title(f'Micro-Support (1/|p|) at ξ={xi0:.2f}')
    
                elif mode == 'Symplectic Vector Field':
                    U, V = symplectic_func(X, xi0)
                    plt.quiver(X, V, U, V, scale=10, width=0.004)
                    plt.title(f'Symplectic Field at ξ={xi0:.2f}')
    
                elif mode == 'Symbol Amplitude':
                    Z = np.abs(symbol_func(X, xi0))
                    plt.plot(x_vals, Z)
                    plt.title(f'Symbol Amplitude |p(x,ξ)| at ξ={xi0:.2f}')
    
                elif mode == 'Symbol Phase':
                    Z = np.angle(symbol_func(X, xi0))
                    plt.plot(x_vals, Z)
                    plt.title(f'Symbol Phase arg(p(x,ξ)) at ξ={xi0:.2f}')
    
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(x_vals, np.linspace(*xi_range, density), x0=x0)
    
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(x_vals, np.linspace(*xi_range, density), x0=x0)
    
                elif mode == 'Wavefront Set':
                    pseudo_op.visualize_wavefront(x_vals, np.linspace(*xi_range, density), xi0=xi0)
    
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, xi0=xi0)
    
            interact(plot_1d,
                     mode=mode_selector,
                     xi0=FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀'),
                     x0=FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀'))
    
        elif dim == 2:
            x, y = vars_x
            xi, eta = symbols('xi eta', real=True)
            grad_func = lambdify((x, y, xi, eta), [diff(expr, xi), diff(expr, eta)], 'numpy')
            symplectic_func = lambdify((x, y, xi, eta), [diff(expr, xi), diff(expr, eta)], 'numpy')
            symbol_func = lambdify((x, y, xi, eta), expr, 'numpy')
    
            def plot_2d(mode, xi0, eta0, x0, y0):
                X, Y = np.meshgrid(x_vals, y_vals, indexing='ij')
    
                if mode == 'Group Velocity Field':
                    U, V = grad_func(X, Y, xi0, eta0)
                    plt.quiver(X, Y, U, V, scale=10, width=0.004)
                    plt.title(f'Group Velocity Field at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Micro-Support (1/|p|)':
                    Z = 1 / (np.abs(symbol_func(X, Y, xi0, eta0)) + 1e-10)
                    plt.pcolormesh(X, Y, Z, shading='auto', cmap='inferno')
                    plt.colorbar(label='1/|p|')
                    plt.title(f'Micro-Support at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symplectic Vector Field':
                    U, V = symplectic_func(X, Y, xi0, eta0)
                    plt.quiver(X, Y, U, V, scale=10, width=0.004)
                    plt.title(f'Symplectic Field at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symbol Amplitude':
                    Z = np.abs(symbol_func(X, Y, xi0, eta0))
                    plt.pcolormesh(X, Y, Z, shading='auto')
                    plt.colorbar(label='|p(x,y,ξ,η)|')
                    plt.title(f'Symbol Amplitude at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Symbol Phase':
                    Z = np.angle(symbol_func(X, Y, xi0, eta0))
                    plt.pcolormesh(X, Y, Z, shading='auto', cmap='twilight')
                    plt.colorbar(label='arg(p)')
                    plt.title(f'Symbol Phase at ξ={xi0:.2f}, η={eta0:.2f}')
    
                elif mode == 'Cotangent Fiber':
                    pseudo_op.visualize_fiber(np.linspace(*xi_range, density), np.linspace(*eta_range, density),
                                              x0=x0, y0=y0)
    
                elif mode == 'Characteristic Set':
                    pseudo_op.visualize_characteristic_set(np.linspace(*xi_range, density),
                                                           np.linspace(*eta_range, density),
                                                           x0=x0, y0=y0)
    
                elif mode == 'Wavefront Set':
                    pseudo_op.visualize_wavefront(x_vals, np.linspace(*xi_range, density),
                                                  y_grid=y_vals, xi0=xi0, eta0=eta0)
    
                elif mode == 'Hamiltonian Flow':
                    pseudo_op.plot_hamiltonian_flow(x0=x0, y0=y0, xi0=xi0, eta0=eta0)
                    
            interact(plot_2d,
                     mode=mode_selector,
                     xi0=FloatSlider(min=xi_range[0], max=xi_range[1], step=0.1, value=1.0, description='ξ₀'),
                     eta0=FloatSlider(min=eta_range[0], max=eta_range[1], step=0.1, value=1.0, description='η₀'),
                     x0=FloatSlider(min=xlim[0], max=xlim[1], step=0.1, value=0.0, description='x₀'),
                     y0=FloatSlider(min=ylim[0], max=ylim[1], step=0.1, value=0.0, description='y₀'))

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
        
        # On ignore les wrappers psiOp et Op
        excluded_wrappers = {'psiOp', 'Op'}
        
        # Extraction des fonctions candidates (hors wrappers)
        candidate_functions = [
            f for f in functions 
            if f.func.__name__ not in excluded_wrappers
        ]
        
        # Keep only user functions (u(x), u(x, t), etc.)
        candidate_functions = [
            f for f in functions
            if isinstance(f, AppliedUndef)
        ]
        
        # Stationary detection: no dependence on t
        self.is_stationary = all(
            not any(str(arg) == 't' for arg in f.args)
            for f in candidate_functions
        )
        
        if len(candidate_functions) != 1:
            print("candidate_functions :", candidate_functions)
            raise ValueError("The equation must contain exactly one unknown function")
        
        self.u = candidate_functions[0]


        args = self.u.args
        
        if self.is_stationary:
            if len(args) not in (1, 2):
                raise ValueError("Stationary problems must depend on 1 or 2 spatial variables")
            self.spatial_vars = args
        else:
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
        else:
            raise ValueError("Only 1D and 2D problems are supported.")

    
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
        if not self.is_stationary:
            self.compute_linear_operator()
        else:
            self.psi_ops = []
            for coeff, sym_expr in self.pseudo_terms:
                psi = PseudoDifferentialOperator(sym_expr, self.spatial_vars, self.u, mode='symbol')
                self.psi_ops.append((coeff, psi))

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
        if lhs.has(psiOp):
            print("⚠️ psiOp detected: skipping expansion for safety")
            lhs_expanded = lhs
        else:
            lhs_expanded = expand(lhs)
        
        print(f"\nExpanded equation: {lhs_expanded}")
    
        linear_terms = {}
        nonlinear_terms = []
        symbol_terms = []
        source_terms = []
        pseudo_terms = []
    
        for term in lhs_expanded.as_ordered_terms():
            print(f"Analyzing term: {term}")
    
            if isinstance(term, psiOp):
                expr = term.args[0]
                pseudo_terms.append((1, expr))
                print("  --> Classified as pseudo linear term (psiOp)")
                continue
            
            # Sinon, cherche psiOp à l’intérieur (cas général)
            if term.has(psiOp):
                psiops = term.atoms(psiOp)
                for psi in psiops:
                    try:
                        coeff = simplify(term / psi)
                        expr = psi.args[0]
                        pseudo_terms.append((coeff, expr))
                        print("  --> Classified as pseudo linear term (psiOp)")
                    except Exception as e:
                        print(f"  ⚠️  Failed to extract psiOp coefficient in term: {term}")
                        print(f"     Reason: {e}")
                        nonlinear_terms.append(term)
                        print("  --> Fallback: classified as nonlinear")
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
            # Vérifie si une dérivée temporelle est présente parmi les termes linéaires
            has_time_derivative = any(
                isinstance(term, Derivative) and self.t in [v for v, _ in term.variable_count]
                for term in linear_terms
            )
            # Extrait les termes linéaires non temporels
            invalid_linear_terms = {
                term: coeff for term, coeff in linear_terms.items()
                if not (
                    isinstance(term, Derivative)
                    and self.t in [v for v, _ in term.variable_count]
                )
                and term != self.u  # exclusion du terme u simple (sans dérivée)
            }

            if invalid_linear_terms or symbol_terms:
                raise ValueError(
                    "Lorsque psiOp est utilisé, seuls les termes non-linéaires, les termes source, "
                    "et éventuellement une dérivée temporelle sont autorisés. "
                    "Les autres termes linéaires et les Op sont interdits."
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

        Parameters
        ----------
        u : np.ndarray
            Input solution array.
        is_v : bool
            Whether to apply the operator to v instead of u.

        Returns
        -------
        np.ndarray
            Result of applying the linear operator.
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
        Set up the computational grid, initial conditions, and parameters.

        Parameters
        ----------
        Lx, Ly : float
            Domain size in x and y directions.
        Nx, Ny : int
            Number of spatial points in x and y directions.
        Lt : float
            Total simulation time.
        Nt : int
            Number of time steps.
        initial_condition : callable
            Function returning initial condition.
        initial_velocity : callable, optional
            Function returning initial velocity (for second-order equations).
        n_frames : int
            Number of frames to store during simulation.
        """

        # time stepping parameters
        self.Lt, self.Nt = Lt, Nt
        self.dt = Lt / Nt
        self.n_frames = n_frames
        self.frames = []
        self.initial_condition = initial_condition

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

            if self.has_psi:
                self.prepare_symbol_tables()
            
            if not self.is_stationary:
                self.u_prev = initial_condition(self.X)

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

        if self.has_psi:
            self.prepare_symbol_tables()

        if not self.is_stationary:
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
            print("For psiOp, please use the interactive_symbol_analysis method separately")
        else:
            self.check_cfl_condition()
    
            self.check_symbol_conditions()
    
            self.plot_symbol()
    
            if self.temporal_order == 2:
                self.analyze_wave_propagation()
            
    def apply_boundary(self, u):
        """
        Apply periodic boundary conditions.

        Parameters
        ----------
        u : np.ndarray
            Solution array.
        """
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
        """
        Apply pseudo-differential operators via precomputed symbols.
        Automatically switches to Kohn-Nirenberg quantization when symbol depends on spatial variables.
        Parameters
        ----------
        u : np.ndarray
            Input solution array.
        Returns
        -------
        np.ndarray
            Updated solution after applying the operator.
        """
        # Check if any symbol depends on spatial variables using symbolic expressions
        use_kohn_nirenberg = False
        for coeff, expr in self.pseudo_terms:
            if expr.has(self.x) or (self.dim == 2 and expr.has(self.y)):
                use_kohn_nirenberg = True
                break
    
        if not use_kohn_nirenberg:
            # Fast path: pure spectral multiplier (no x/y dependence)
            u_hat = self.fft(u)
            combined_symbol = np.zeros_like(u_hat, dtype=np.complex128)
            for coeff, precomputed_symbol in self.precomputed_symbols:
                coeff = np.complex128(coeff)
                symbol = np.array(precomputed_symbol, dtype=np.complex128)
                combined_symbol += coeff * symbol
            u_hat *= np.exp(-self.dt * combined_symbol)
            u_hat *= self.dealiasing_mask
            return self.ifft(u_hat)
    
        else:
            # Slow but accurate path: apply Kohn-Nirenberg quantization
            def build_symbol_func(symbol_expr):
                if self.dim == 1:
                    x, xi = symbols('x xi', real=True)
                    return lambdify((x, xi), symbol_expr, 'numpy')
                else:
                    x, y, xi, eta = symbols('x y xi eta', real=True)
                    return lambdify((x, y, xi, eta), symbol_expr, 'numpy')
    
            total_symbol = 0
            for coeff, expr in self.pseudo_terms:
                total_symbol += coeff * expr
            symbol_func = build_symbol_func(total_symbol)
            return self.kohn_nirenberg_fft(f_vals=u, symbol_func=symbol_func)


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
    
    def solve_stationary_psiOp(self, order=3):
        """
        Solve P[u] = f(x) or f(x,y) for stationary pseudo-differential problems using asymptotic inversion.
    
        Parameters
        ----------
        order : int
            Order of the asymptotic inverse expansion.
        method : str
            'diagonal' for fast approximate inverse (default), 'full' for pointwise exact inverse (slower).
    
        Returns
        -------
        ndarray
            The solution u(x) or u(x, y)
        """
        if not self.has_psi:
            raise ValueError("Only supports problems with psiOp.")
    
        if self.linear_terms or self.nonlinear_terms:
            raise ValueError("Stationary psiOp problems must be linear and purely pseudo-differential.")
    
        if self.dim == 1:
            x = self.x
            xi = symbols('xi', real=True)
            spatial_vars = (x,)
            freq_vars = (xi,)
            X, KX = self.X, self.KX
        elif self.dim == 2:
            x, y = self.x, self.y
            xi, eta = symbols('xi eta', real=True)
            spatial_vars = (x, y)
            freq_vars = (xi, eta)
            X, Y, KX, KY = self.X, self.Y, self.KX, self.KY
        else:
            raise ValueError("Unsupported spatial dimension.")
    
        total_symbol = sum(coeff * psi.expr for coeff, psi in self.psi_ops)
        psi_total = PseudoDifferentialOperator(total_symbol, spatial_vars, mode='symbol')
    
        # Check ellipticity
        if self.dim == 1:
            is_elliptic = psi_total.is_elliptic_numerically(X, KX)
        else:
            is_elliptic = psi_total.is_elliptic_numerically((X[:, 0], Y[0, :]), (KX[:, 0], KY[0, :]))
        if not is_elliptic:
            raise ValueError("❌ The pseudo-differential symbol is not numerically elliptic on the grid.")
        print("✅ Elliptic pseudo-differential symbol: inversion allowed.")
    
        R_symbol = psi_total.right_inverse_asymptotic(order=order)
        print("Right inverse asymptotic symbol:")
        pprint(R_symbol)

        if self.dim == 1:
            if R_symbol.has(x):
                R_func = lambdify((x, xi), R_symbol, modules='numpy')
            else:
                R_func = lambdify((xi,), R_symbol, modules='numpy')
        else:
            if R_symbol.has(x) or R_symbol.has(y):
                R_func = lambdify((x, y, xi, eta), R_symbol, modules='numpy')
            else:
                R_func = lambdify((xi, eta), R_symbol, modules='numpy')
    
        # Build rhs
        if self.source_terms:
            f_expr = sum(self.source_terms)
            used_vars = [v for v in spatial_vars if f_expr.has(v)]
            f_func = lambdify(used_vars, -f_expr, modules='numpy')
            if self.dim == 1:
                rhs = f_func(self.x_grid) if used_vars else np.zeros_like(self.x_grid)
            else:
                rhs = f_func(self.X, self.Y) if used_vars else np.zeros_like(self.X)
        elif self.initial_condition:
            raise ValueError("Initial condition should be None for stationnary equation.")
        else:
            raise ValueError("No source term provided to construct the right-hand side.")
    
        f_hat = self.fft(rhs)
    
        if self.dim == 1:
            Nx = self.Nx
            if not R_symbol.has(x):
                print("⚡ Optimisation : symbole indépendant de x — produit direct en Fourier.")
                R_vals = R_func(self.KX)
                u_hat = R_vals * f_hat
                u = np.fft.ifft(u_hat)
            else:
                print("⚙️  Quantification de Kohn-Nirenberg 1D")
                x, xi = symbols('x xi', real=True)
                R_func = lambdify((x, xi), R_symbol, 'numpy')  # Still 2 args for uniformity
                u = self.kohn_nirenberg_fft(f_vals=rhs, symbol_func=R_func)
                
        elif self.dim == 2:
            Nx, Ny = self.Nx, self.Ny
            if not R_symbol.has(x) and not R_symbol.has(y):
                print("⚡ Optimisation : symbole indépendant de x et y — produit direct en Fourier 2D.")
                R_vals = np.vectorize(R_func)(self.KX, self.KY)
                u_hat = R_vals * f_hat
                u = np.fft.ifft2(u_hat)
            else:
                print("⚙️  Quantification de Kohn-Nirenberg 2D")
                x, xi, y, eta = symbols('x xi y eta', real=True)
                R_func = lambdify((x, y, xi, eta), R_symbol, 'numpy')  # Still 2 args for uniformity
                u = self.kohn_nirenberg_fft(f_vals=rhs, symbol_func=R_func)
        self.u = u
        return u

    def kohn_nirenberg_fft(self, f_vals, symbol_func,
                           freq_window='gaussian', clamp=1e6,
                           space_window=False):
        """
        Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator.
    
        Parameters
        ----------
        f_vals : np.ndarray
            Spatial samples of the input function f(x) or f(x, y).
        symbol_func : callable
            Symbol function p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
            Must be a NumPy-compatible function (e.g., via lambdify).
        freq_window : {'gaussian', 'hann', None}, optional
            Type of frequency-domain window to apply to smooth out oscillations and suppress aliasing.
        clamp : float, optional
            Maximum absolute value allowed for the symbol. Helps prevent numerical blow-up.
        space_window : bool, optional
            Whether to apply a spatial Gaussian window to regularize edge behavior.
    
        Returns
        -------
        np.ndarray
            Output array corresponding to Op(p)[f], same shape as f_vals.
        """
    
        # === Common setup ===
        xg = self.x_grid
        dx = xg[1] - xg[0]
    
        if self.dim == 1:
            # === 1D case ===
    
            # Frequency grid (shifted to center zero)
            Nx = self.Nx
            k = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
            dk = k[1] - k[0]
    
            # Centered FFT of input
            f_shift = np.fft.ifftshift(f_vals)
            f_hat = self.fft(f_shift) * dx
            f_hat = np.fft.fftshift(f_hat)
    
            # Build meshgrid for (x, ξ)
            X, K = np.meshgrid(xg, k, indexing='ij')
    
            # Evaluate the symbol p(x, ξ)
            P = symbol_func(X, K)
    
            # Optional: clamp extreme values
            P = np.clip(P, -clamp, clamp)
    
            # === Frequency-domain window ===
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(k))
                W = np.exp(-(K / sigma) ** 4)
                P *= W
            elif freq_window == 'hann':
                W = 0.5 * (1 + np.cos(np.pi * K / np.max(np.abs(K))))
                P *= W * (np.abs(K) < np.max(np.abs(K)))
    
            # === Optional spatial window ===
            if space_window:
                x0 = (xg[0] + xg[-1]) / 2
                L = (xg[-1] - xg[0]) / 2
                S = np.exp(-((X - x0) / L) ** 2)
                P *= S
    
            # === Oscillatory kernel and integration ===
            kernel = np.exp(1j * X * K)
            integrand = P * f_hat[None, :] * kernel
    
            # Approximate inverse Fourier integral
            u = np.sum(integrand, axis=1) * dk / (2 * np.pi)
            return u
    
        else:
            # === 2D case ===
    
            yg = self.y_grid
            dy = yg[1] - yg[0]
            Nx, Ny = self.Nx, self.Ny
    
            # Frequency grids
            kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx, d=dx))
            ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny, d=dy))
            dkx = kx[1] - kx[0]
            dky = ky[1] - ky[0]
    
            # 2D FFT of f(x, y)
            f_hat = np.fft.fftshift(self.fft(f_vals)) * dx * dy
    
            # Create 4D grids for broadcasting
            X, Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
            KX, KY = np.meshgrid(kx, ky, indexing='ij')
            Xb = X[:, :, None, None]
            Yb = Y[:, :, None, None]
            KXb = KX[None, None, :, :]
            KYb = KY[None, None, :, :]
    
            # Evaluate p(x, y, ξ, η)
            P_vals = symbol_func(Xb, Yb, KXb, KYb)
            P_vals = np.clip(P_vals, -clamp, clamp)
    
            # === Frequency windowing ===
            if freq_window == 'gaussian':
                sigma_kx = 0.8 * np.max(np.abs(kx))
                sigma_ky = 0.8 * np.max(np.abs(ky))
                W_kx = np.exp(-(KXb / sigma_kx) ** 4)
                W_ky = np.exp(-(KYb / sigma_ky) ** 4)
                P_vals *= W_kx * W_ky
            elif freq_window == 'hann':
                Wx = 0.5 * (1 + np.cos(np.pi * KXb / np.max(np.abs(kx))))
                Wy = 0.5 * (1 + np.cos(np.pi * KYb / np.max(np.abs(ky))))
                mask_x = np.abs(KXb) < np.max(np.abs(kx))
                mask_y = np.abs(KYb) < np.max(np.abs(ky))
                P_vals *= Wx * Wy * mask_x * mask_y
    
            # === Optional spatial tapering ===
            if space_window:
                x0 = (self.x_grid[0] + self.x_grid[-1]) / 2
                y0 = (self.y_grid[0] + self.y_grid[-1]) / 2
                Lx = (self.x_grid[-1] - self.x_grid[0]) / 2
                Ly = (self.y_grid[-1] - self.y_grid[0]) / 2
                S = np.exp(-((Xb - x0) / Lx) ** 2 - ((Yb - y0) / Ly) ** 2)
                P_vals *= S
    
            # === Oscillatory kernel and integration ===
            phase = np.exp(1j * (Xb * KXb + Yb * KYb))
            integrand = P_vals * phase * f_hat[None, None, :, :]
    
            # 2D Fourier inversion (numerical integration)
            u = np.sum(integrand, axis=(2, 3)) * dkx * dky / (2 * np.pi) ** 2
            return u
           
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
        Evaluate the weighted sum of pseudo-differential symbols on the grid.

        Returns
        -------
        np.ndarray
            Combined symbol values as a complex numpy array.
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

    def show_stationary_solution(self, u=None, component=r'abs', cmap='viridis'):
        """
        Display the stationary solution computed by solve_stationary_psiOp.
        
        Parameters
        ----------
        u : ndarray, optional
            Precomputed solution array. If None, calls solve_stationary_psiOp().
        cmap : str, optional
            Colormap to use for 2D display (default: 'viridis')
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
                
        if u is None:
            u = self.solve_stationary_psiOp()

        if self.dim == 1:
            # Plot the solution in 1D
            plt.figure(figsize=(8, 4))
            plt.plot(self.x_grid, get_component(u), label=f'{component} of u')
            plt.xlabel('x')
            plt.ylabel(f'{component} of u')
            plt.title('Stationary solution (1D)')
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()
    
        elif self.dim == 2:
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel(f'{component.title()} of u')
            ax.set_title('Initial condition')
    
            data0 = get_component(u)
            surf = [ax.plot_surface(self.X, self.Y, data0, cmap='viridis')]
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError("Only 1D and 2D display are supported.")

    
    def animate(self, component='abs', overlay='contour'):
        """
        Create an animated plot of the solution evolution.

        Parameters
        ----------
        component : str {'real', 'imag', 'abs', 'angle'}
            Component of the solution to animate.
        overlay : str {'contour', 'front'}, optional
            Overlay type in 2D animations.

        Returns
        -------
        FuncAnimation
            Animation object.
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
        Test the solver against an exact solution.
    
        Parameters
        ----------
        u_exact : callable
            Exact solution function.
        t_eval : float, optional
            Time at which to compare (ignored if stationary).
        norm : str {'relative', 'absolute'}
            Type of error norm.
        threshold : float
            Acceptable error threshold.
        plot : bool
            Whether to display plots.
        component : str {'real', 'imag', 'abs'}
            Component to compare.
        """
        if self.is_stationary:
            print("Testing a stationary solution.")
            u_num = self.u
    
            # Compute exact solution
            if self.dim == 1:
                u_ex = u_exact(self.X)
            elif self.dim == 2:
                u_ex = u_exact(self.X, self.Y)
            else:
                raise ValueError("Unsupported dimension.")
            actual_t = None
        else:
            if t_eval is None:
                t_eval = self.Lt
    
            save_interval = max(1, self.Nt // self.n_frames)
            frame_times = np.arange(0, self.Lt + self.dt, save_interval * self.dt)
            frame_index = np.argmin(np.abs(frame_times - t_eval))
            actual_t = frame_times[frame_index]
            print(f"Closest available time to t_eval={t_eval}: {actual_t}")
    
            if frame_index >= len(self.frames):
                raise ValueError(f"Time t = {t_eval} exceeds simulation duration.")
    
            u_num = self.frames[frame_index]
    
            # Compute exact solution at the actual time
            if self.dim == 1:
                u_ex = u_exact(self.X, actual_t)
            elif self.dim == 2:
                u_ex = u_exact(self.X, self.Y, actual_t)
            else:
                raise ValueError("Unsupported dimension.")
    
        # Select component
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
    
        label_time = f"t = {actual_t}" if actual_t is not None else ""
        print(f"Test error {label_time}: {error:.3e}")
        assert error < threshold, f"Error too large {label_time}: {error:.3e}"
    
        # Plot
        if plot:
            if self.dim == 1:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(self.X, np.real(u_num), label='Numerical')
                plt.plot(self.X, np.real(u_ex), '--', label='Exact')
                plt.title(f'Solution {label_time}, error = {error:.2e}')
                plt.legend()
                plt.grid()
    
                plt.subplot(2, 1, 2)
                plt.plot(self.X, np.abs(diff), color='red')
                plt.title('Absolute Error')
                plt.grid()
                plt.tight_layout()
                plt.show()
            else:
                import matplotlib.pyplot as plt
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
