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
"""
PDESolver — A Spectral Method PDE Solver with Symbolic Capabilities

Overview
--------
This module provides a flexible and symbolic-based solver for partial differential equations (PDEs)
using spectral methods. It supports:
- 1D and 2D problems
- First- and second-order time evolution
- Linear and nonlinear PDEs
- Symbolic parsing via SymPy
- Exponential time integration and ETD-RK4 schemes
- Advanced pseudo-differential operator analysis
- Interactive visualization using IPython widgets

Symbolic Workflow
-----------------
The solver accepts PDEs defined symbolically using SymPy syntax. For example:
>>> from sympy import Function, diff, Eq
>>> u = Function('u')
>>> t, x = symbols('t x')
>>> eq = Eq(diff(u(t,x), t), diff(u(t,x), x, 2) + u(t,x)**2)

It automatically extracts:
- The linear operator L(k)
- Dispersion relation ω(k)
- Nonlinear terms
- Pseudo-differential operators (psiOp)

Numerical Methods
-----------------
- Fourier-based spectral differentiation
- Dealiasing for nonlinear terms
- Temporal integrators:
    - Default exponential stepping
    - ETD-RK4 (Exponential Time Differencing Runge-Kutta of 4th order)

Interactive Analysis
--------------------
Use `interactive_symbol_analysis(pseudo_op)` to explore:
- Group velocity fields
- Symbol amplitude/phase
- Hamiltonian flows
- Characteristic sets
- Wavefront propagation

Example Usage
-------------
>>> from sympy import sin, pi
>>> def initial(x): return sin(2 * pi * x)
>>> solver = PDESolver(eq)
>>> solver.setup(Lx=1.0, Nx=256, Lt=1.0, Nt=1000, initial_condition=initial)
>>> solver.solve()
>>> ani = solver.animate()
>>> HTML(ani.to_jshtml())
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft, fftfreq, fftshift, ifftshift
from scipy.signal.windows import hann
from scipy.integrate import solve_ivp
from scipy.ndimage import maximum_filter
from sympy import (
    symbols, Function, 
    solve, pprint, Mul,
    lambdify, expand, Eq, simplify, trigsimp, N,
    radsimp, ratsimp, cancel,
    Lambda, Piecewise, Basic, degree, Pow, preorder_traversal, 
    powdenest, expand, 
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
from scipy.special import legendre, eval_hermite, airy, eval_genlaguerre
from matplotlib import cm
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from matplotlib import rc
from IPython.display import HTML
from functools import partial
from misc import * 
from IPython.display import display
from ipywidgets import interact, FloatSlider, Dropdown
from itertools import product

    
plt.rcParams['text.usetex'] = False
FFT_WORKERS = 4

class Op(Function):
    """Custom symbolic wrapper for pseudo-differential operators in Fourier space.
    Usage: Op(symbol_expr, u)
    """
    nargs = 2


class psiOp(Function):
    """Symbolic wrapper for PseudoDifferentialOperator.
    Usage: psiOp(symbol_expr, u)
    """
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
        - 'auto': computes the symbol automatically by applying expr to exp(i x ξ).

    Attributes
    ----------
    dim : int
        Spatial dimension (1 or 2).
    fft, ifft : callable
        Fast Fourier transform and inverse (scipy.fft or scipy.fft2).
    p_func : callable
        Evaluated symbol function ready for numerical use.

    Notes
    -----
    - In 'symbol' mode, `expr` should be expressed in terms of spatial variables and frequency variables (ξ, η).
    - In 'auto' mode, the symbol is derived by applying the differential expression to a complex exponential.
    - Frequency variables are internally named 'xi' and 'eta' for consistency.
    - Uses numpy for numerical evaluation and scipy.fft for FFT operations.

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
        self.symbol_cached = None
        self.expr = expr
        self.vars_x = vars_x

        if self.dim == 1:
            x, = vars_x
            xi_internal = symbols('xi', real=True)
            expr = expr.subs(symbols('xi', real=True), xi_internal)
            self.fft = partial(fft, workers=FFT_WORKERS)
            self.ifft = partial(ifft, workers=FFT_WORKERS)

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
            self.fft = partial(fft2, workers=FFT_WORKERS)
            self.ifft = partial(ifft2, workers=FFT_WORKERS)

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
        pprint(expr, num_columns=150)
        
    def evaluate(self, X, Y, KX, KY, cache=True):
        """
        Evaluate the pseudo-differential operator's symbol on a grid of spatial and frequency coordinates.

        The method dynamically selects between 1D and 2D evaluation based on the spatial dimension.
        If caching is enabled and a cached symbol exists, it returns the cached result to avoid recomputation.

        Parameters
        ----------
        X, Y : ndarray
            Spatial grid coordinates. In 1D, Y is ignored.
        KX, KY : ndarray
            Frequency grid coordinates. In 1D, KY is ignored.
        cache : bool, default=True
            If True, stores the computed symbol for reuse in subsequent calls to avoid redundant computation.

        Returns
        -------
        ndarray
            Evaluated symbol values over the input grid. Shape matches the input spatial/frequency grids.

        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.
        """
        if cache and self.symbol_cached is not None:
            return self.symbol_cached

        if self.dim == 1:
            symbol = self.p_func(X, KX)
        elif self.dim == 2:
            symbol = self.p_func(X, Y, KX, KY)

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
        Compute the leading homogeneous component of the pseudo-differential symbol.

        This method extracts the principal part of the symbol, which is the dominant 
        term under high-frequency asymptotics (|ξ| → ∞). The expansion is performed 
        in polar coordinates for 2D symbols to maintain rotational symmetry, then 
        converted back to Cartesian form.

        Parameters
        ----------
        order : int
            Order of the asymptotic expansion in powers of 1/ρ, where ρ = |ξ| in 1D 
            or ρ = sqrt(ξ² + η²) in 2D. Only the leading-order term is returned.

        Returns
        -------
        sympy.Expr
            The principal symbol component, homogeneous of degree `m - order`, where 
            `m` is the original symbol's order.

        Notes:
        - In 1D, uses direct series expansion in ξ.
        - In 2D, expands in radial variable ρ while preserving angular dependence.
        - Useful for microlocal analysis and constructing parametrices.
        """

        p = self.expr
        if self.dim == 1:
            xi = symbols('xi', real=True, positive=True)
            return simplify(series(p, xi, oo, n=order).removeO())
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True, positive=True)
            # Homogeneous radial expansion: we set (ξ, η) = ρ (cosθ, sinθ)
            rho, theta = symbols('rho theta', real=True, positive=True)
            p_rho = p.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
            expansion = series(p_rho, rho, oo, n=order).removeO()
            # Revert back to (ξ, η)
            expansion_cart = expansion.subs({rho: sqrt(xi**2 + eta**2),
                                             cos(theta): xi / sqrt(xi**2 + eta**2),
                                             sin(theta): eta / sqrt(xi**2 + eta**2)})
            return simplify(powdenest(expansion_cart, force=True))
                       
    def is_homogeneous(self, tol=1e-10):
        """
        Check whether the symbol is homogeneous in the frequency variables.
    
        Returns
        -------
        (bool, Rational or float or None)
            Tuple (is_homogeneous, degree) where:
            - is_homogeneous: True if the symbol satisfies p(λξ, λη) = λ^m * p(ξ, η)
            - degree: the detected degree m if homogeneous, or None
        """
        from sympy import symbols, simplify, expand, Eq
        from sympy.abc import l
    
        if self.dim == 1:
            xi = symbols('xi', real=True, positive=True)
            l = symbols('l', real=True, positive=True)
            p = self.expr
            p_scaled = p.subs(xi, l * xi)
            ratio = simplify(p_scaled / p)
            if ratio.has(xi):
                return False, None
            try:
                deg = simplify(ratio).as_base_exp()[1]
                return True, deg
            except Exception:
                return False, None
    
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True, positive=True)
            l = symbols('l', real=True, positive=True)
            p = self.expr
            p_scaled = p.subs({xi: l * xi, eta: l * eta})
            ratio = simplify(p_scaled / p)
            # If ratio == l**m with no (xi, eta) left, it's homogeneous
            if ratio.has(xi, eta):
                return False, None
            try:
                base, exp = ratio.as_base_exp()
                if base == l:
                    return True, exp
            except Exception:
                pass
            return False, None

    def symbol_order(self, max_order=10, tol=1e-3):
        """
        Estimate the homogeneity order of the pseudo-differential symbol in high-frequency asymptotics.
    
        This method attempts to determine the leading-order behavior of the symbol p(x, ξ) or p(x, y, ξ, η)
        as |ξ| → ∞ (in 1D) or |(ξ, η)| → ∞ (in 2D). The returned value represents the asymptotic growth or decay rate,
        which is essential for understanding the regularity and mapping properties of the corresponding operator.
    
        The function uses symbolic preprocessing to ensure proper factorization of frequency variables,
        especially in sqrt and power expressions, to avoid erroneous order detection (e.g., due to hidden scaling).
    
        Parameters
        ----------
        max_order : int, optional
            Maximum number of terms to consider in the series expansion. Default is 10.
        tol : float, optional
            Tolerance threshold for evaluating the coefficient magnitude. If the coefficient is too small,
            the detected order may be discarded. Default is 1e-3.
    
        Returns
        -------
        float or None
            - If the symbol is homogeneous, returns its exact homogeneity degree as a float.
            - Otherwise, estimates the dominant asymptotic order from leading terms in the expansion.
            - Returns None if no valid order could be determined.
    
        Notes
        -----
        - In 1D:
            Two strategies are used:
                1. Expand directly in xi at infinity.
                2. Substitute xi = 1/z and expand around z = 0.
    
        - In 2D:
            - Transform the symbol into polar coordinates: (xi, eta) = rho*(cos(theta), sin(theta)).
            - Expand in rho at infinity, then extract the leading term's power.
            - An alternative substitution using 1/z is also tried if the first method fails.
    
        - Preprocessing steps:
            - Sqrt expressions involving frequencies are rewritten to isolate the leading variable.
            - Power expressions are factored explicitly to ensure correct symbolic scaling.
    
        - If the symbol is not homogeneous, a warning is issued, and the result should be interpreted with care.
        
        - For non-homogeneous symbols, only the principal asymptotic term is considered.
    
        Raises
        ------
        NotImplementedError
            If the spatial dimension is neither 1 nor 2.
        """
        from sympy import symbols, series, simplify, sqrt, cos, sin, oo, powdenest, radsimp, expand, expand_power_base
    
        def preprocess_sqrt(expr, freq):
            return expr.replace(
                lambda e: e.func == sqrt and freq in e.free_symbols,
                lambda e: freq * sqrt(1 + (e.args[0] - freq**2) / freq**2)
            )
    
        def preprocess_power(expr, freq):
            """Force factorization of the leading frequency variable in power expressions."""
            return expr.replace(
                lambda e: e.is_Pow and freq in e.free_symbols,
                lambda e: freq**e.exp * (1 + e.base/freq**e.base.as_powers_dict().get(freq, 0))**e.exp
            )

        # Check if the symbol is homogeneous
        is_homog, degree = self.is_homogeneous()
        if is_homog:
            return float(degree)
        else:
            print("⚠️ The symbol is not homogeneous. The asymptotic order is not well defined.")
    
        if self.dim == 1:
            x = self.vars_x[0]
            xi = symbols('xi', real=True, positive=True)
            try:
                print("1D symbol_order - method 1")
                expr = preprocess_sqrt(self.expr, xi)
                s = series(expr, xi, oo, n=max_order).removeO()
                lead = s.as_leading_term(xi)
                lead = simplify(powdenest(lead, force=True))
                power = lead.as_powers_dict().get(xi, None)
                coeff = lead / xi**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
    
                if power is not None and (not coeff.free_symbols or x not in coeff.free_symbols):
                    if abs(float(coeff.evalf())) > tol:
                        return int(power) if power == int(power) else float(power)
            except Exception:
                pass
    
            try:
                print("1D symbol_order - method 2")
                z = symbols('z', real=True, positive=True)
                expr_z = self.expr.subs(xi, 1/z)
                expr_z = preprocess_sqrt(expr_z, 1/z)
                s = series(expr_z, z, 0, n=max_order).removeO()
                lead = s.as_leading_term(z)
                lead = simplify(powdenest(lead, force=True))
                power = lead.as_powers_dict().get(z, None)
                print("lead =", lead)
                print("power =", power)
    
                if power is not None:
                    return -float(power)
            except Exception as e:
                print(f"⚠️ fallback z failed: {e}")
            return None
    
        elif self.dim == 2:
            x, y = self.vars_x
            xi, eta = symbols('xi eta', real=True, positive=True)
            rho, theta = symbols('rho theta', real=True, positive=True)
    
            try:
                print("2D symbol_order - method 1")
                p_rho = self.expr.subs({xi: rho * cos(theta), eta: rho * sin(theta)})
                p_rho = preprocess_sqrt(p_rho, rho)
                p_rho = preprocess_power(p_rho, rho)  
                p_rho = simplify(p_rho)
                s = series(p_rho, rho, oo, n=max_order).removeO()
                s = expand(s)
                lead = s.as_leading_term(rho)
                lead = simplify(powdenest(lead, force=True))
                lead = radsimp(lead)
                power = lead.as_powers_dict().get(rho, None)
                coeff = lead / rho**power if power is not None else 0
                print("lead =", lead)
                print("power =", power)
                print("coeff =", coeff)
    
                if power is not None and not any(v in coeff.free_symbols for v in (x, y)):
                    if abs(float(coeff.evalf())) > tol or power > -1:
                        return int(power) if power == int(power) else float(power)
            except Exception as e:
                print(f"⚠️ polar expansion failed: {e}")
                pass
    
            try:
                print("2D symbol_order - method 2")
                z = symbols('z', real=True, positive=True)
                p_rho = self.expr.subs({xi: (1/z) * cos(theta), eta: (1/z) * sin(theta)})
                p_rho = preprocess_sqrt(p_rho, 1/z)
                p_rho = simplify(p_rho)
                s = series(p_rho, z, 0, n=max_order).removeO()
                lead = s.as_leading_term(z)
                lead = simplify(powdenest(lead, force=True))
                lead = radsimp(lead)
                power = lead.as_powers_dict().get(z, None)
                print("lead =", lead)
                print("power =", power)
    
                if power is not None:
                    order = -power
                    return int(order) if order == int(order) else float(order)
            except Exception as e:
                print(f"⚠️ fallback z (2D) failed: {e}")
            return None
    
        else:
            raise NotImplementedError("Only 1D and 2D supported.")
        
    def asymptotic_expansion(self, order=3):
        """
        Compute the asymptotic expansion of the symbol as |ξ| → ∞ (high-frequency regime).
    
        This method expands the pseudo-differential symbol in inverse powers of the 
        frequency variable(s), either in 1D or 2D. It handles both polynomial and 
        exponential symbols by performing a series expansion in 1/|ξ| up to the specified order.
    
        The expansion is performed directly in Cartesian coordinates for 1D symbols.
        For 2D symbols, the method uses polar coordinates (ρ, θ) to perform the expansion 
        at infinity in ρ, then converts the result back to Cartesian coordinates.
    
        Parameters
        ----------
        order : int, optional
            Maximum order of the asymptotic expansion. Default is 3.
    
        Returns
        -------
        sympy.Expr
            The asymptotic expansion of the symbol up to the given order, expressed in Cartesian coordinates.
            If expansion fails, returns the original unexpanded symbol.
    
        Notes:
        - In 1D: expansion is performed directly in terms of ξ.
        - In 2D: the symbol is first rewritten in polar coordinates (ρ,θ), expanded asymptotically 
          in ρ → ∞, then converted back to Cartesian coordinates (ξ,η).
        - Handles special case when the symbol is an exponential function by expanding its argument.
        - Symbolic normalization is applied early (via `simplify`) for 2D expressions to improve convergence.
        - Robust to failures: catches exceptions and issues warnings instead of raising errors.
        - Final expression is simplified using `powdenest` and `expand` for improved readability.
        """
        p = self.expr
    
        if self.dim == 1:
            xi = symbols('xi', real=True, positive=True)
    
            try:
                # Case: exponential function
                if p.func == exp and len(p.args) == 1:
                    arg = p.args[0]
                    arg_series = series(arg, xi, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
                else:
                    expanded = series(p, xi, oo, n=order).removeO()
                    return simplify(powdenest(expanded, force=True))
    
            except Exception as e:
                print(f"Warning: 1D expansion failed: {e}")
                return p
    
        elif self.dim == 2:
            xi, eta = symbols('xi eta', real=True, positive=True)
            rho, theta = symbols('rho theta', real=True, positive=True)
    
            # Normalize before substitution
            p = simplify(p)
    
            # Substitute polar coordinates
            p_polar = p.subs({
                xi: rho * cos(theta),
                eta: rho * sin(theta)
            })
    
            try:
                # Handle exponentials
                if p_polar.func == exp and len(p_polar.args) == 1:
                    arg = p_polar.args[0]
                    arg_series = series(arg, rho, oo, n=order).removeO()
                    expanded = series(exp(expand(arg_series)), rho, oo, n=order).removeO()
                else:
                    expanded = series(p_polar, rho, oo, n=order).removeO()
    
                # Convert back to Cartesian
                norm = sqrt(xi**2 + eta**2)
                expansion_cart = expanded.subs({
                    rho: norm,
                    cos(theta): xi / norm,
                    sin(theta): eta / norm
                })
    
                # Final simplifications
                result = simplify(powdenest(expansion_cart, force=True))
                result = expand(result)
                return result
    
            except Exception as e:
                print(f"Warning: 2D expansion failed: {e}")
                return p  
            
    def compose_asymptotic(self, other, order=1):
        """
        Compose this pseudo-differential operator with another using formal asymptotic expansion.

        This method computes the composition symbol via an asymptotic expansion in powers of 
        derivatives, following the symbolic calculus of pseudo-differential operators. The 
        composition is performed up to the specified order and respects the dimensionality 
        (1D or 2D) of the operators.

        Parameters
        ----------
        other : PseudoDifferentialOperator
            The pseudo-differential operator to compose with this one.
        order : int, default=1
            Maximum order of the asymptotic expansion. Higher values include more terms in the 
            symbolic composition, increasing accuracy at the cost of complexity.

        Returns
        -------
        sympy.Expr
            Symbolic expression representing the asymptotic expansion of the composed operator.

        Notes
        -----
        - In 1D, the composition uses the formula:
          (p ∘ q)(x, ξ) ~ Σₙ (1/n!) ∂_ξⁿ p(x, ξ) ∂_xⁿ q(x, ξ) (i)^{-n}
        - In 2D, the multi-index generalization is used:
          (p ∘ q)(x, y, ξ, η) ~ Σₙ Σᵢ (1/(i! j!)) ∂_ξⁱ∂_ηʲ p ∂_xⁱ∂_yʲ q (i)^{-n}, where n = i + j.
        - This expansion is valid for symbols admitting an asymptotic series representation.
        - Operators must be defined on the same spatial domain (same dimension).
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
        Construct a formal right inverse R of the pseudo-differential operator P such that 
        the composition P ∘ R equals the identity plus a smoothing operator of order -order.
    
        This method computes an asymptotic expansion for the right inverse using recursive 
        corrections based on derivatives of the symbol p(x, ξ) and lower-order terms of R.
    
        Parameters
        ----------
        order : int
            Number of terms to include in the asymptotic expansion. Higher values improve 
            approximation at the cost of complexity and computational effort.
    
        Returns
        -------
        sympy.Expr
            The symbolic expression representing the formal right inverse R(x, ξ), which satisfies:
            P ∘ R = Id + O(⟨ξ⟩^{-order}), where ⟨ξ⟩ = (1 + |ξ|²)^{1/2}.
    
        Notes
        -----
        - In 1D: The recursion involves spatial derivatives of R and derivatives of p with respect to ξ.
        - In 2D: The multi-index generalization is used with mixed derivatives in ξ and η.
        - The construction relies on the non-vanishing of the principal symbol p to ensure invertibility.
        - Each term in the expansion corresponds to higher-order corrections involving commutators 
          between the operator P and the current approximation of R.
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
        Construct a formal left inverse L such that the composition L ∘ P equals the identity 
        operator up to terms of order ξ^{-order}. This expansion is performed asymptotically 
        at infinity in the frequency variable(s).
    
        The left inverse is built iteratively using symbolic differentiation and the 
        method of asymptotic expansions for pseudo-differential operators. It ensures that:
        
            L(P(x,ξ),x,D) ∘ P(x,D) = Id + smoothing operator of order -order
    
        Parameters
        ----------
        order : int, optional
            Maximum number of terms in the asymptotic expansion (default is 1). Higher values 
            yield more accurate inverses at the cost of increased computational complexity.
    
        Returns
        -------
        sympy.Expr
            Symbolic expression representing the principal symbol of the formal left inverse 
            operator L(x,ξ). This expression depends on spatial variables and frequencies, 
            and includes correction terms up to the specified order.
    
        Notes
        -----
        - In 1D: Uses recursive application of the Leibniz formula for symbols.
        - In 2D: Generalizes to multi-indices for mixed derivatives in (x,y) and (ξ,η).
        - Each term involves combinations of derivatives of the original symbol p(x,ξ) and 
          previously computed terms of the inverse.
        - Coefficients include powers of 1j (i) and factorial normalization for derivative terms.
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
        Compute the formal adjoint symbol P* of the pseudo-differential operator.

        The adjoint is defined such that for any test functions u and v,
        ⟨P u, v⟩ = ⟨u, P* v⟩ holds in the distributional sense. This is obtained by 
        taking the complex conjugate of the symbol and expanding it asymptotically 
        at infinity to ensure proper behavior under integration by parts.

        Returns
        -------
        sympy.Expr
            The adjoint symbol P*(x, ξ) in 1D or P*(x, y, ξ, η) in 2D.
        
        Notes:
        - In 1D, the expansion is performed in powers of 1/|ξ|.
        - In 2D, the expansion is radial in |ξ| = sqrt(ξ² + η²).
        - This method ensures symbolic simplifications for readability and efficiency.
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
        Compute the Hamiltonian vector field associated with the principal symbol.

        This method derives the canonical equations of motion for the phase space variables 
        (x, ξ) in 1D or (x, y, ξ, η) in 2D, based on the Hamiltonian formalism. These describe 
        how position and frequency variables evolve under the flow generated by the symbol.

        Returns
        -------
        dict
            A dictionary containing the components of the Hamiltonian vector field:
            - In 1D: keys are 'dx/dt' and 'dxi/dt', corresponding to dx/dt = ∂p/∂ξ and dξ/dt = -∂p/∂x.
            - In 2D: keys are 'dx/dt', 'dy/dt', 'dxi/dt', and 'deta/dt', with similar definitions:
              dx/dt = ∂p/∂ξ, dy/dt = ∂p/∂η, dξ/dt = -∂p/∂x, dη/dt = -∂p/∂y.

        Notes
        -----
        - The Hamiltonian here is the principal symbol p(x, ξ) itself.
        - This flow preserves the symplectic structure of phase space.
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
        Check if the pseudo-differential symbol p(x, ξ) is elliptic over a given grid.
    
        A symbol is considered elliptic if its magnitude |p(x, ξ)| remains bounded away from zero 
        across all points in the spatial-frequency domain. This method evaluates the symbol on a 
        grid of spatial and frequency coordinates and checks whether its minimum absolute value 
        exceeds a specified threshold.
    
        Resampling is applied to large grids to prevent excessive memory usage, particularly in 2D.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid: either a 1D array (x) or a tuple of two 1D arrays (x, y).
        xi_grid : ndarray
            Frequency grid: either a 1D array (ξ) or a tuple of two 1D arrays (ξ, η).
        threshold : float, optional
            Minimum acceptable value for |p(x, ξ)|. If the smallest evaluated symbol value falls below this,
            the symbol is not considered elliptic.
    
        Returns
        -------
        bool
            True if the symbol is elliptic on the resampled grid, False otherwise.
        """
        RESAMPLE_SIZE = 32  # Reduced size to prevent memory explosion
        
        if self.dim == 1:
            x_vals = x_grid
            xi_vals = xi_grid
            # Resampling if necessary
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
        
            X, XI = np.meshgrid(x_vals, xi_vals, indexing='ij')
            symbol_vals = self.p_func(X, XI)
        
        elif self.dim == 2:
            x_vals, y_vals = x_grid
            xi_vals, eta_vals = xi_grid
        
            # Spatial resampling
            if len(x_vals) > RESAMPLE_SIZE:
                x_vals = np.linspace(x_vals.min(), x_vals.max(), RESAMPLE_SIZE)
            if len(y_vals) > RESAMPLE_SIZE:
                y_vals = np.linspace(y_vals.min(), y_vals.max(), RESAMPLE_SIZE)
        
            # Frequency resampling
            if len(xi_vals) > RESAMPLE_SIZE:
                xi_vals = np.linspace(xi_vals.min(), xi_vals.max(), RESAMPLE_SIZE)
            if len(eta_vals) > RESAMPLE_SIZE:
                eta_vals = np.linspace(eta_vals.min(), eta_vals.max(), RESAMPLE_SIZE)
        
            X, Y, XI, ETA = np.meshgrid(x_vals, y_vals, xi_vals, eta_vals, indexing='ij')
            symbol_vals = self.p_func(X, Y, XI, ETA)
        
        min_abs_val = np.min(np.abs(symbol_vals))
        return min_abs_val > threshold


    def is_self_adjoint(self, tol=1e-10):
        """
        Check whether the pseudo-differential operator is formally self-adjoint (Hermitian).

        A self-adjoint operator satisfies P = P*, where P* is the formal adjoint of P.
        This property is essential for ensuring real-valued eigenvalues and stable evolution 
        in quantum mechanics and symmetric wave propagation.

        Parameters
        ----------
        tol : float
            Tolerance for symbolic comparison between P and P*. Small numerical differences 
            below this threshold are considered equal.

        Returns
        -------
        bool
            True if the symbol p(x, ξ) equals its formal adjoint p*(x, ξ) within the given tolerance,
            indicating that the operator is self-adjoint.

        Notes:
        - The formal adjoint is computed via conjugation and asymptotic expansion at infinity in ξ.
        - Symbolic simplification is used to verify equality, ensuring robustness against superficial 
          expression differences.
        """
        p = self.expr
        p_star = self.formal_adjoint()
        return simplify(p - p_star).equals(0)

    def visualize_wavefront(self, x_grid, xi_grid, y_grid=None, eta_grid=None, xi0=0.0, eta0=0.0):
        """
        Visualize the wavefront set of the pseudo-differential operator's symbol by plotting |p(x, ξ)| or |p(x, y, ξ₀, η₀)|.
    
        The wavefront set captures the location and direction of singularities in the symbol p, which is crucial in 
        microlocal analysis for understanding propagation of singularities and wave behavior.
    
        In 1D, this method displays |p(x, ξ)| over the phase space (x, ξ). In 2D, it fixes the frequency values (ξ₀, η₀)
        and visualizes |p(x, y, ξ₀, η₀)| over the spatial domain to highlight where the symbol interacts with the fixed frequency.
    
        Parameters
        ----------
        x_grid : ndarray
            1D array of spatial coordinates (x) used for evaluation.
        xi_grid : ndarray
            1D array of frequency coordinates (ξ) used for visualization in 1D or slicing in 2D.
        y_grid : ndarray, optional
            1D array of second spatial coordinate (y), used only in 2D.
        eta_grid : ndarray, optional
            1D array of second frequency coordinate (η), not directly used but kept for API consistency.
        xi0 : float, default=0.0
            Fixed value of ξ for slicing in 2D visualization.
        eta0 : float, default=0.0
            Fixed value of η for slicing in 2D visualization.
    
        Notes
        -----
        - In 1D:
            Displays a 2D color map of |p(x, ξ)| over the phase space with axes (x, ξ).
        - In 2D:
            Evaluates and plots |p(x, y, ξ₀, η₀)| at fixed frequencies (ξ₀, η₀) over the spatial grid (x, y).
        - Uses `imshow` for fast and scalable visualization with automatic aspect ratio adjustment.
        - A colormap ('viridis') is used to enhance contrast and readability of the magnitude variations.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid)
            symbol_vals = self.p_func(X, XI)
            plt.imshow(np.abs(symbol_vals), extent=[xi_grid.min(), xi_grid.max(), x_grid.min(), x_grid.max()],
                       aspect='auto', origin='lower', cmap='viridis')
        elif self.dim == 2:
            X, Y = np.meshgrid(x_grid, y_grid)
            XI = np.full_like(X, xi0)
            ETA = np.full_like(Y, eta0)
            symbol_vals = self.p_func(X, Y, XI, ETA)
            plt.imshow(np.abs(symbol_vals), extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                       aspect='auto', origin='lower', cmap='viridis')
        plt.colorbar(label='|Symbol|')
        plt.xlabel('ξ/Position')
        plt.ylabel('η/Position')
        plt.title('Wavefront Set')
        plt.show()

    def visualize_fiber(self, x_grid, xi_grid, y0=0.0, x0=0.0):
        """
        Plot the cotangent fiber structure at a fixed spatial point (x₀[, y₀]).
    
        This visualization shows how the symbol p(x, ξ) behaves on the cotangent fiber 
        above a fixed spatial point. In microlocal analysis, this provides insight into 
        the frequency content of the operator at that location.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid values (1D) for evaluation in 1D case.
        xi_grid : ndarray
            Frequency grid values (1D) for evaluation in both 1D and 2D cases.
        x0 : float, optional
            Fixed x-coordinate of the base point in space (1D or 2D).
        y0 : float, optional
            Fixed y-coordinate of the base point in space (2D only).
    
        Notes
        -----
        - In 1D: Displays |p(x, ξ)| over the (x, ξ) phase plane near the fixed point.
        - In 2D: Fixes (x₀, y₀) and evaluates p(x₀, y₀, ξ, η), showing the fiber over that point.
        - The color map represents the magnitude of the symbol, highlighting regions where it vanishes or becomes singular.
    
        Raises
        ------
        NotImplementedError
            If called in 2D with missing or improperly formatted grids.
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
        Display the modulus |p(x, ξ)| or |p(x, y, ξ₀, η₀)| as a color map.
    
        This method visualizes the amplitude of the pseudodifferential operator's symbol 
        in either 1D or 2D spatial configuration. In 2D, the frequency variables are fixed 
        to specified values (ξ₀, η₀) for visualization purposes.
    
        Parameters
        ----------
        x_grid, y_grid : ndarray
            Spatial grids over which to evaluate the symbol. y_grid is optional and used only in 2D.
        xi_grid, eta_grid : ndarray
            Frequency grids. In 2D, these define the domain over which the symbol is evaluated,
            but the visualization fixes ξ = ξ₀ and η = η₀.
        xi0, eta0 : float, optional
            Fixed frequency values for slicing in 2D visualization. Defaults to zero.
    
        Notes
        -----
        - In 1D: Visualizes |p(x, ξ)| over the (x, ξ) grid.
        - In 2D: Visualizes |p(x, y, ξ₀, η₀)| at fixed frequencies ξ₀ and η₀.
        - The color intensity represents the magnitude of the symbol, highlighting regions where the symbol is large or small.
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
        Plot the phase (argument) of the pseudodifferential operator's symbol p(x, ξ) or p(x, y, ξ, η).

        This visualization helps in understanding the oscillatory behavior and regularity 
        properties of the operator in phase space. The phase is displayed modulo 2π using 
        a cyclic colormap ('twilight') to emphasize its periodic nature.

        Parameters
        ----------
        x_grid : ndarray
            1D array of spatial coordinates (x).
        xi_grid : ndarray
            1D array of frequency coordinates (ξ).
        y_grid : ndarray, optional
            2D spatial grid for y-coordinate (in 2D problems). Default is None.
        eta_grid : ndarray, optional
            2D frequency grid for η (in 2D problems). Not used directly but kept for API consistency.
        xi0 : float, optional
            Fixed value of ξ for slicing in 2D visualization. Default is 0.0.
        eta0 : float, optional
            Fixed value of η for slicing in 2D visualization. Default is 0.0.

        Notes:
        - In 1D: Displays arg(p(x, ξ)) over the (x, ξ) phase plane.
        - In 2D: Displays arg(p(x, y, ξ₀, η₀)) for fixed frequency values (ξ₀, η₀).
        - Uses plt.pcolormesh with 'twilight' colormap to represent angles from -π to π.

        Raises:
        - NotImplementedError: If the spatial dimension is not 1D or 2D.
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
    def visualize_characteristic_set(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0, levels=[1e-1]):
        """
        Visualize the characteristic set of the pseudo-differential symbol, defined as the approximate zero set p(x, ξ) ≈ 0.
    
        In microlocal analysis, the characteristic set is the locus of points in phase space (x, ξ) where the symbol p(x, ξ) vanishes,
        playing a key role in understanding propagation of singularities and wavefronts.
    
        Parameters
        ----------
        x_grid : ndarray
            Spatial grid values (1D array) for plotting in 1D or evaluation point in 2D.
        xi_grid : ndarray
            Frequency variable grid values (1D array) used to construct the frequency domain.
        x0 : float, optional
            Fixed spatial coordinate in 2D case for evaluating the symbol at a specific x position.
        y0 : float, optional
            Fixed spatial coordinate in 2D case for evaluating the symbol at a specific y position.
    
        Notes
        -----
        - For 1D, this method plots the contour of |p(x, ξ)| = ε with ε = 1e-5 over the (x, ξ) plane.
        - For 2D, it evaluates the symbol at fixed (x₀, y₀) and plots the characteristic set in the (ξ, η) frequency plane.
        - This visualization helps identify directions of degeneracy or hypoellipticity of the operator.
    
        Raises
        ------
        NotImplementedError
            If called on a solver with dimensionality other than 1D or 2D.
    
        Displays
        ------
        A matplotlib contour plot showing either:
            - The characteristic curve in the (x, ξ) phase plane (1D),
            - The characteristic surface slice in the (ξ, η) frequency plane at (x₀, y₀) (2D).
        """
        if self.dim == 1:
            x_grid = np.asarray(x_grid)
            xi_grid = np.asarray(xi_grid)
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI) 
            plt.contour(X, XI, np.abs(symbol_vals), levels=levels, colors='red')
            plt.xlabel('x')
            plt.ylabel('ξ')
            plt.title('Characteristic Set (p(x, ξ) ≈ 0)')
            plt.grid(True)
            plt.show()
        elif self.dim == 2:
            if eta_grid is None:
                raise ValueError("eta_grid must be provided for 2D visualization.")
            xi_grid = np.asarray(xi_grid)
            eta_grid = np.asarray(eta_grid)
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, eta_grid, indexing='ij')
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            plt.contour(xi_grid, eta_grid, np.abs(symbol_vals), levels=levels, colors='red')
            plt.xlabel('ξ')
            plt.ylabel('η')
            plt.title(f'Characteristic Set at x={x0}, y={y0}')
            plt.grid(True)
            plt.show()
        else:
            raise NotImplementedError("Only 1D/2D characteristic sets supported.")

    def visualize_characteristic_gradient(self, x_grid, xi_grid, y_grid=None, eta_grid=None, y0=0.0, x0=0.0):
        """
        Visualize the norm of the gradient of the symbol p(x, ξ) or p(x₀, y₀, ξ, η) over phase space.
    
        This method computes and displays |∇p|, the magnitude of the gradient of the pseudo-differential operator's symbol.
        High values indicate strong variation in the symbol, while low values suggest near-zero regions (characteristic sets).
    
        In 1D:
            The full gradient ∇p = (∂ₓp, ∂ξp) is evaluated over the (x, ξ) grid.
            Displays |∇p(x, ξ)| to highlight areas where the symbol changes rapidly.
    
        In 2D:
            Evaluates ∇p = (∂ξp, ∂ηp) at fixed spatial point (x₀, y₀).
            Displays |∇p(x₀, y₀, ξ, η)| to locate frequency directions where the symbol varies strongly.
    
        Parameters:
            x_grid (ndarray): 1D array of spatial coordinates (x) for 1D case or evaluation point in 2D.
            xi_grid (ndarray): 1D array of frequency coordinates (ξ).
            y_grid (ndarray, optional): 2D spatial grid for y-coordinate (not used directly here). Default: None.
            eta_grid (ndarray, optional): 1D array of frequency coordinates (η) for 2D visualization. Default: None.
            y0 (float, optional): Fixed y-coordinate in 2D mode. Default: 0.0.
            x0 (float, optional): Fixed x-coordinate in 2D mode. Default: 0.0.
    
        Displays:
            A color map showing |∇p| over:
              - (x, ξ) grid in 1D
              - (ξ, η) grid at (x₀, y₀) in 2D
    
        Notes:
            - Gradient is computed numerically using np.gradient().
            - Color intensity reflects the strength of the gradient — useful for identifying singularities.
            - Characteristic sets often lie where |∇p| is small and p ≈ 0.
        """
        if self.dim == 1:
            X, XI = np.meshgrid(x_grid, xi_grid, indexing='ij')
            symbol_vals = self.p_func(X, XI)
            grad_x = np.gradient(symbol_vals, axis=0)
            grad_xi = np.gradient(symbol_vals, axis=1)
            grad_norm = np.sqrt(grad_x**2 + grad_xi**2)
            plt.pcolormesh(X, XI, grad_norm, cmap='inferno', shading='auto')
            plt.colorbar(label='|∇p|')
            plt.title('Gradient Norm (High Near Zeros)')
            plt.grid(True)
            plt.show()
        elif self.dim == 2:
            xi_grid2, eta_grid2 = np.meshgrid(xi_grid, eta_grid, indexing='ij')
            symbol_vals = self.p_func(x0, y0, xi_grid2, eta_grid2)
            grad_xi = np.gradient(symbol_vals, axis=0)
            grad_eta = np.gradient(symbol_vals, axis=1)
            grad_norm = np.sqrt(grad_xi**2 + grad_eta**2)
            plt.pcolormesh(xi_grid, eta_grid, grad_norm, cmap='inferno', shading='auto')
            plt.colorbar(label='|∇p|')
            plt.title(f'Gradient Norm at x={x0}, y={y0}')
            plt.grid(True)
            plt.show()

    def visualize_dynamic_wavefront(self, x_grid, t_grid, y_grid=None, xi0=5.0, eta0=0.0):
        """
        Visualize the propagation of a singularity along bicharacteristic curves as a dynamic wavefront.
    
        This method generates a 1D or 2D spatial-time plot of a wavefield initialized with a given frequency 
        (xi₀, η₀). In 1D, it shows u(x, t) = cos(ξ₀x - ξ₀t), representing a right-moving wave. In 2D, it plots  
        u(x, y, t) = cos(ξ₀x + η₀y - |k|t), where |k| = √(ξ₀² + η₀²), simulating a plane wave propagating in 
        direction (ξ₀, η₀).
    
        Parameters
        ----------
        x_grid : ndarray
            1D or 2D array representing the spatial grid in the x-direction.
        t_grid : ndarray
            Array of time points used to construct the wave evolution.
        y_grid : ndarray, optional
            1D or 2D array for the second spatial dimension (only used in 2D cases).
        xi0 : float, default=5.0
            Initial frequency component in the x-direction.
        eta0 : float, default=0.0
            Initial frequency component in the y-direction (used in 2D only).
    
        Notes
        -----
        - In 1D, this visualizes a simple harmonic wave moving at unit speed.
        - In 2D, the wave propagates with group velocity magnitude |k| = √(ξ₀² + η₀²).
        - The wavefronts are stationary in time for 2D due to plotting at fixed t = t_grid[0].
    
        Displays
        --------
        A matplotlib image plot showing:
            - In 1D: u(x, t) over space-time (x, t)
            - In 2D: u(x, y) at initial time t = t_grid[0]
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
        Integrate and plot the Hamiltonian trajectories of the symbol in phase space.

        This method numerically integrates the Hamiltonian vector field derived from 
        the operator's symbol to visualize how singularities propagate under the flow. 
        It supports both 1D and 2D problems.

        Parameters
        ----------
        x0, xi0 : float
            Initial position and frequency (momentum) in 1D.
        y0, eta0 : float, optional
            Initial position and frequency in 2D; defaults to zero.
        tmax : float
            Final integration time for the ODE solver.
        n_steps : int
            Number of time steps used in the integration.

        Notes
        -----
        - The Hamiltonian vector field is obtained from the symplectic flow of the symbol.
        - If the field is complex-valued, only its real part is used for integration.
        - In 1D, the trajectory is plotted in (x, ξ) phase space.
        - In 2D, the spatial trajectory (x(t), y(t)) is shown along with instantaneous 
          momentum vectors (ξ(t), η(t)) using a quiver plot.

        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.

        Displays
        --------
        matplotlib plot
            Phase space trajectory(ies) showing the evolution of position and momentum 
            under the Hamiltonian dynamics.
        """
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
        Visualize the symplectic vector field (Hamiltonian vector field) associated with the operator's symbol.

        The plotted vector field corresponds to (∂_ξ p, -∂_x p), where p(x, ξ) is the principal symbol 
        of the pseudo-differential operator. This field governs the bicharacteristic flow in phase space.

        Parameters
        ----------
        xlim : tuple of float
            Range for spatial variable x, as (x_min, x_max).
        klim : tuple of float
            Range for frequency variable ξ, as (ξ_min, ξ_max).
        density : int
            Number of grid points per axis for the visualization grid.

        Raises
        ------
        NotImplementedError
            If called on a 2D operator (currently only 1D implementation available).

        Notes
        -----
        - Only supports one-dimensional operators.
        - Uses symbolic differentiation to compute ∂_ξ p and ∂_x p.
        - Numerical evaluation is done via lambdify with NumPy backend.
        - Visualization uses matplotlib quiver plot to show vector directions.
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
        Visualize the micro-support of the operator by plotting the inverse of the symbol magnitude 1 / |p(x, ξ)|.
    
        The micro-support provides insight into the singularities of a pseudo-differential operator 
        in phase space (x, ξ). Regions where |p(x, ξ)| is small correspond to large values in 1/|p(x, ξ)|,
        highlighting areas of significant operator influence or singularity.
    
        Parameters
        ----------
        xlim : tuple
            Spatial domain limits (x_min, x_max).
        klim : tuple
            Frequency domain limits (ξ_min, ξ_max).
        threshold : float
            Threshold below which |p(x, ξ)| is considered effectively zero; used for numerical stability.
        density : int
            Number of grid points along each axis for visualization resolution.
    
        Raises
        ------
        NotImplementedError
            If called on a solver with dimension greater than 1 (only 1D visualization is supported).
    
        Notes
        -----
        - This method evaluates the symbol p(x, ξ) over a grid and plots its reciprocal to emphasize 
          regions where the symbol is near zero.
        - A small constant (1e-10) is added to the denominator to avoid division by zero.
        - The resulting plot helps identify characteristic sets and wavefront set approximations.
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
        Plot the group velocity field ∇_ξ p(x, ξ) for 1D pseudo-differential operators.

        The group velocity represents the speed at which waves of different frequencies propagate 
        in a dispersive medium. It is defined as the gradient of the symbol p(x, ξ) with respect 
        to the frequency variable ξ.

        Parameters
        ----------
        xlim : tuple of float
            Spatial domain limits (x-axis).
        klim : tuple of float
            Frequency domain limits (ξ-axis).
        density : int
            Number of grid points per axis used for visualization.

        Raises
        ------
        NotImplementedError
            If called on a 2D operator, since this visualization is only implemented for 1D.

        Notes
        -----
        - This method visualizes the vector field (∂p/∂ξ) in phase space.
        - Used for analyzing wave propagation properties and dispersion relations.
        - Requires symbolic expression self.expr depending on x and ξ.
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

    def animate_singularity(self, xi0=5.0, eta0=0.0, x0=0.0, y0=0.0,
                            tmax=4.0, n_frames=100, projection=None):
        """
        Animate the propagation of a singularity under the Hamiltonian flow.

        This method visualizes how a singularity (x₀, y₀, ξ₀, η₀) evolves in phase space 
        according to the Hamiltonian dynamics induced by the principal symbol of the operator.
        The animation integrates the Hamiltonian equations of motion and supports various projections:
        position (x-y), frequency (ξ-η), or mixed phase space coordinates.

        Parameters
        ----------
        xi0, eta0 : float
            Initial frequency components (ξ₀, η₀).
        x0, y0 : float
            Initial spatial coordinates (x₀, y₀).
        tmax : float
            Total time of integration (final animation time).
        n_frames : int
            Number of frames in the resulting animation.
        projection : str or None
            Type of projection to display:
                - 'position' : x vs y (or x alone in 1D)
                - 'frequency': ξ vs η (or ξ alone in 1D)
                - 'phase'    : mixed coordinates like x vs ξ or x vs η
                If None, defaults to 'phase' in 1D and 'position' in 2D.

        Returns
        -------
        matplotlib.animation.FuncAnimation
            Animation object that can be displayed interactively in Jupyter notebooks or saved as a video.

        Notes
        -----
        - In 1D, only one spatial and one frequency variable are used.
        - Complex-valued Hamiltonian fields are truncated to their real parts for integration.
        - Trajectories are shown with both instantaneous position (dot) and full path (dashed line).
        """
        rc('animation', html='jshtml')
    
        def make_real(expr):
            return simplify(expr.as_real_imag()[0])
    
        H = self.symplectic_flow()
    
        if any(im(H[k]) != 0 for k in H):
            print("⚠️ The Hamiltonian field is complex. Only the real part is used for integration.")
    
        if self.dim == 1:
            x, = self.vars_x
            xi = symbols('xi', real=True)
    
            dxdt = lambdify((x, xi), make_real(H['dx/dt']), 'numpy')
            dxidt = lambdify((x, xi), make_real(H['dxi/dt']), 'numpy')
    
            def hamilton(t, Y):
                x, xi = Y
                return [dxdt(x, xi), dxidt(x, xi)]
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, xi0],
                            t_eval=np.linspace(0, tmax, n_frames))
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
    
            sol = solve_ivp(hamilton, [0, tmax], [x0, y0, xi0, eta0],
                            t_eval=np.linspace(0, tmax, n_frames))
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
        """
        Launch an interactive dashboard for symbol exploration using ipywidgets.
    
        This function provides a user-friendly interface to visualize various aspects of the pseudo-differential operator's symbol.
        It supports multiple visualization modes in both 1D and 2D, including group velocity fields, micro-support estimates,
        symplectic vector fields, symbol amplitude/phase, cotangent fiber structure, characteristic sets, wavefront sets,
        and Hamiltonian flows.
    
        Parameters
        ----------
        pseudo_op : PseudoDifferentialOperator
            The pseudo-differential operator whose symbol is to be analyzed interactively.
        xlim, ylim : tuple of float
            Spatial domain limits along x and y axes respectively.
        xi_range, eta_range : tuple
            Frequency domain limits along ξ and η axes respectively.
        density : int
            Number of points per axis used to construct the evaluation grid. Controls resolution.
    
        Notes
        -----
        - In 1D mode, sliders control the fixed frequency (ξ₀) and spatial position (x₀).
        - In 2D mode, additional sliders control the second frequency component (η₀) and second spatial coordinate (y₀).
        - Visualization updates dynamically as parameters are adjusted via sliders or dropdown menus.
        - Supported visualization modes:
            'Group Velocity Field'       : ∇_ξ p(x,ξ) or ∇_{ξ,η} p(x,y,ξ,η)
            'Micro-Support (1/|p|)'      : Reciprocal of symbol magnitude
            'Symplectic Vector Field'    : (∇_ξ p, -∇_x p) or similar in 2D
            'Symbol Amplitude'           : |p(x,ξ)| or |p(x,y,ξ,η)|
            'Symbol Phase'               : arg(p(x,ξ)) or similar in 2D
            'Cotangent Fiber'            : Structure of symbol over frequency space at fixed x
            'Characteristic Set'         : Zero set approximation {p ≈ 0}
            'Characteristic Gradient'    : |∇p(x, ξ)| or |∇p(x₀, y₀, ξ, η)|
            'Wavefront Set'              : High-frequency singularities detected via symbol interaction
            'Hamiltonian Flow'           : Trajectories generated by the Hamiltonian vector field
    
        Raises
        ------
        NotImplementedError
            If the spatial dimension is not 1D or 2D.
    
        Prints
        ------
        Interactive matplotlib figures with dynamic updates based on widget inputs.
        """
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
                'Characteristic Gradient',
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
    
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_vals, np.linspace(*xi_range, density), x0=x0)
    
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
                    pseudo_op.visualize_characteristic_set(x_grid=x_vals, xi_grid=np.linspace(*xi_range, density),
                                                  y_grid=y_vals, eta_grid=np.linspace(*eta_range, density), x0=x0, y0=y0)
    
                elif mode == 'Characteristic Gradient':
                    pseudo_op.visualize_characteristic_gradient(x_grid=x_vals, xi_grid=np.linspace(*xi_range, density),
                                                  y_grid=y_vals, eta_grid=np.linspace(*eta_range, density), x0=x0, y0=y0)
    
                elif mode == 'Wavefront Set':
                    pseudo_op.visualize_wavefront(x_grid=x_vals, xi_grid=np.linspace(*xi_range, density),
                                                  y_grid=y_vals, eta_grid=np.linspace(*eta_range, density), xi0=xi0, eta0=eta0)
    
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
    A partial differential equation (PDE) solver based on **spectral methods** using Fourier transforms.

    This solver supports symbolic specification of PDEs via SymPy and numerical solution using high-order spectral techniques. 
    It is designed for both **linear and nonlinear time-dependent PDEs**, as well as **stationary pseudo-differential problems**.
    
    Key Features:
    -------------
    - Symbolic PDE parsing using SymPy expressions
    - 1D and 2D spatial domains with periodic boundary conditions
    - Fourier-based spectral discretization with dealiasing
    - Temporal integration schemes:
        - Default exponential time stepping
        - ETD-RK4 (Exponential Time Differencing Runge-Kutta of 4th order)
    - Nonlinear terms handled through pseudo-spectral evaluation
    - Built-in tools for:
        - Visualization of solutions and error surfaces
        - Symbol analysis of linear and pseudo-differential operators
        - Microlocal analysis (e.g., wavefront set estimation, Hamiltonian flows)
        - CFL condition checking and numerical stability diagnostics

    Supported Operators:
    --------------------
    - Linear differential and pseudo-differential operators
    - Nonlinear terms up to second order in derivatives
    - Symbolic operator composition and adjoints
    - Asymptotic inversion of elliptic operators for stationary problems

    Example Usage:
    --------------
    >>> from sympy import Function, diff, Eq
    >>> from matplotlib import pyplot as plt
    >>> u = Function('u')
    >>> t, x = symbols('t x')
    >>> eq = Eq(diff(u(t, x), t), diff(u(t, x), x, 2) + u(t, x)**2)
    >>> def initial(x): return np.sin(x)
    >>> solver = PDESolver(eq)
    >>> solver.setup(Lx=2*np.pi, Nx=128, Lt=1.0, Nt=1000, initial_condition=initial)
    >>> solver.solve()
    >>> ani = solver.animate()
    >>> HTML(ani.to_jshtml())  # Display animation in Jupyter notebook
    """
    def __init__(self, equation, time_scheme='default', dealiasing_ratio=2/3, verbose=False):
        """
        Initialize the PDE solver with a given equation.

        This method analyzes the input partial differential equation (PDE), 
        identifies the unknown function and its dependencies, determines whether 
        the problem is stationary or time-dependent, and prepares symbolic and 
        numerical structures for solving in spectral space.

        Supported features:
        - 1D and 2D problems
        - Time-dependent and stationary equations
        - Linear and nonlinear terms
        - Pseudo-differential operators via `psiOp`
        - Source terms and boundary conditions

        The equation is parsed to extract linear, nonlinear, source, and 
        pseudo-differential components. Symbolic manipulation is used to derive 
        the Fourier representation of linear operators when applicable.

        Args:
            equation (sympy.Eq): The PDE expressed as a SymPy equation.
            time_scheme (str): Temporal integration scheme; 'default' for exponential 
                               time-stepping or 'ETD-RK4' for fourth-order exponential 
                               time differencing Runge–Kutta.
            dealiasing_ratio (float): Fraction of high-frequency modes to zero out 
                                     during dealiasing (e.g., 2/3 for standard truncation).

        Attributes initialized:
        - self.u: the unknown function (e.g., u(t, x))
        - self.dim: spatial dimension (1 or 2)
        - self.spatial_vars: list of spatial variables (e.g., [x] or [x, y])
        - self.is_stationary: boolean indicating if the problem is stationary
        - self.linear_terms: dictionary mapping derivative orders to coefficients
        - self.nonlinear_terms: list of nonlinear expressions
        - self.source_terms: list of source functions
        - self.pseudo_terms: list of pseudo-differential operator expressions
        - self.has_psi: boolean indicating presence of pseudo-differential operators
        - self.fft / self.ifft: appropriate FFT routines based on spatial dimension
        - self.kx, self.ky: symbolic wavenumber variables for Fourier space

        Raises:
            ValueError: If the equation does not contain exactly one unknown function,
                        if unsupported dimensions are detected, or invalid dependencies.
        """
        self.verbose = verbose
        self.time_scheme = time_scheme # 'default'  or 'ETD-RK4'
        self.dealiasing_ratio = dealiasing_ratio
        
        print("\n*********************************")
        print("* Partial differential equation *")
        print("*********************************\n")
        pprint(equation)
        
        # Extract symbols and function from the equation
        functions = equation.atoms(Function)
        
        # Ignore the wrappers psiOp and Op
        excluded_wrappers = {'psiOp', 'Op'}
        
        # Extract the candidate fonctions (excluding wrappers)
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

        
        if self.dim == 1:
            self.fft = partial(fft, workers=FFT_WORKERS)
            self.ifft = partial(ifft, workers=FFT_WORKERS)
        else:
            self.fft = partial(fft2, workers=FFT_WORKERS)
            self.ifft = partial(ifft2, workers=FFT_WORKERS)
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
            print('⚠️  Pseudo‑differential operator detected: all other linear terms have been rejected.')
            self.is_spatial = False
            for coeff, expr in self.pseudo_terms:
                if expr.has(self.x) or (self.dim == 2 and expr.has(self.y)):
                    self.is_spatial = True
                    break
    
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
        Parse the PDE to separate linear and nonlinear terms, symbolic operators (Op), 
        source terms, and pseudo-differential operators (psiOp).
    
        This method rewrites the input equation in standard form (lhs - rhs = 0),
        expands it, and classifies each term into one of the following categories:
        
        - Linear terms involving derivatives or the unknown function u
        - Nonlinear terms (products with u, powers of u, etc.)
        - Symbolic pseudo-differential operators (Op)
        - Source terms (independent of u)
        - Pseudo-differential operators (psiOp)
    
        Args:
            equation (sympy.Eq): The partial differential equation to be analyzed. 
                                 Can be provided as an Eq object or a sympy expression.
    
        Returns:
            tuple: A 5-tuple containing:
                - linear_terms (dict): Mapping from derivative/function to coefficient.
                - nonlinear_terms (list): List of terms classified as nonlinear.
                - symbol_terms (list): List of (coefficient, symbolic operator) pairs.
                - source_terms (list): List of terms independent of the unknown function.
                - pseudo_terms (list): List of (coefficient, pseudo-differential symbol) pairs.
    
        Notes:
            - If `psiOp` is present in the equation, expansion is skipped for safety.
            - When `psiOp` is used, only nonlinear terms, source terms, and possibly 
              a time derivative are allowed; other linear terms and symbolic operators 
              (Op) are forbidden.
            - Classification logic includes:
                - Detection of nonlinear structures like products or powers of u
                - Mixed terms involving both u and its derivatives
                - External symbolic operators (Op) and pseudo-differential operators (psiOp)
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
    
            # Otherwise, look for psiOp inside (general case)
            if term.has(psiOp):
                psiops = term.atoms(psiOp)
                for psi in psiops:
                    try:
                        coeff = simplify(term / psi)
                        expr = psi.args[0]
                        pseudo_terms.append((coeff, expr))
                        print("  --> Classified as pseudo linear term (psiOp)")
                    except Exception as e:
                        print(f"  ⚠️ Failed to extract psiOp coefficient in term: {term}")
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
            # Check if a time derivative is present among the linear terms
            has_time_derivative = any(
                isinstance(term, Derivative) and self.t in [v for v, _  in term.variable_count]
                for term in linear_terms
            )
            # Extract non-temporal linear terms
            invalid_linear_terms = {
                term: coeff for term, coeff in linear_terms.items()
                if not (
                    isinstance(term, Derivative)
                    and self.t in [v for v, _  in term.variable_count]
                )
                and term != self.u  # exclusion of the simple u term (without derivative)
            }
    
            if invalid_linear_terms or symbol_terms:
                raise ValueError(
                    "When psiOp is used, only nonlinear terms, source terms, "
                    "and possibly a time derivative are allowed. "
                    "Other linear terms and Ops are forbidden."
                )
    
        return linear_terms, nonlinear_terms, symbol_terms, source_terms, pseudo_terms


    def compute_linear_operator(self):
        """
        Compute the symbolic Fourier representation L(k) of the linear operator 
        derived from the linear part of the PDE.
    
        This method constructs a dispersion relation by applying each symbolic derivative
        to a plane wave exp(i(k·x - ωt)) and extracting the resulting expression.
        It handles arbitrary derivative combinations and includes symbolic and
        pseudo-differential terms.
    
        Steps:
        -------
        1. Construct a plane wave φ(x, t) = exp(i(k·x - ωt)).
        2. Apply each term from self.linear_terms to φ.
        3. Normalize by φ and simplify to obtain L(k).
        4. Include symbolic terms (e.g., psiOp) if present.
        5. Detect the temporal order from the dispersion relation.
        6. Build the numerical function L(k) via lambdify.
    
        Sets:
        -----
        self.L_symbolic : sympy.Expr
            Symbolic form of L(k).
        self.L : callable
            Numerical function of L(kx[, ky]).
        self.omega : callable or None
            Frequency root ω(k), if available.
        self.temporal_order : int
            Order of time derivatives detected.
        self.psi_ops : list of (coeff, PseudoDifferentialOperator)
            Pseudo-differential terms present in the equation.
    
        Raises:
        -------
        ValueError if the dimension is unsupported or the dispersion relation fails.
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
        except Exception as e:
            warnings.warn(f"Could not determine temporal order: {e}", RuntimeWarning)
            self.temporal_order = 0
        print(f"Temporal order from dispersion relation: {self.temporal_order}")
        print('self.pseudo_terms = ', self.pseudo_terms)
        if self.pseudo_terms:
            coeff_time = 1
            for term, coeff in self.linear_terms.items():
                if isinstance(term, Derivative) and any(var == self.t for var, _  in term.variable_count):
                    coeff_time = coeff
                    print(f"✅ Time derivative coefficient detected: {coeff_time}")
            self.psi_ops = []
            for coeff, sym_expr in self.pseudo_terms:
                # expr est le Sympy expr. différentiel, var_x la liste [x] ou [x,y]
                psi = PseudoDifferentialOperator(sym_expr / coeff_time, self.spatial_vars, self.u, mode='symbol')
                
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

    def setup(self, Lx, Ly=None, Nx=None, Ny=None, Lt=1.0, Nt=100, boundary_condition='periodic',
              initial_condition=None, initial_velocity=None, n_frames=100):
        """
        Configure the spatial/temporal grid and initialize the solution field.
    
        This method sets up the computational domain, initializes spatial and temporal grids,
        applies boundary conditions, and prepares symbolic and numerical operators.
        It also performs essential analyses such as:
        
            - CFL condition verification (for stability)
            - Symbol analysis (e.g., dispersion relation, regularity)
            - Wave propagation analysis for second-order equations
    
        If pseudo-differential operators (ψOp) are present, symbolic analysis is skipped
        in favor of interactive exploration via `interactive_symbol_analysis`.
    
        Parameters
        ----------
        Lx : float
            Size of the spatial domain along x-axis.
        Ly : float, optional
            Size of the spatial domain along y-axis (for 2D problems).
        Nx : int
            Number of spatial points along x-axis.
        Ny : int, optional
            Number of spatial points along y-axis (for 2D problems).
        Lt : float, default=1.0
            Total simulation time.
        Nt : int, default=100
            Number of time steps.
        initial_condition : callable
            Function returning the initial state u(x, 0) or u(x, y, 0).
        initial_velocity : callable, optional
            Function returning the initial time derivative ∂ₜu(x, 0) or ∂ₜu(x, y, 0),
            required for second-order equations.
        n_frames : int, default=100
            Number of time frames to store during simulation for visualization or output.
    
        Raises
        ------
        ValueError
            If mandatory parameters are missing (e.g., Nx not given in 1D, Ly/Ny not given in 2D).
    
        Notes
        -----
        - The spatial discretization assumes periodic boundary conditions by default.
        - Fourier transforms are computed using real-to-complex FFTs (`scipy.fft.fft`, `fft2`).
        - Frequency arrays (`KX`, `KY`) are defined following standard spectral conventions.
        - Dealiasing is applied using a sharp cutoff filter at a fraction of the maximum frequency.
        - For second-order equations, initial acceleration is derived from the governing operator.
        - Symbolic analysis includes plotting of the symbol's real/imaginary/absolute values,
          wavefront propagation, and dispersion relation.
    
        See Also
        --------
        setup_1D : Sets up internal variables for one-dimensional problems.
        setup_2D : Sets up internal variables for two-dimensional problems.
        initialize_conditions : Applies initial data and enforces compatibility.
        check_cfl_condition : Verifies time step against stability constraints.
        plot_symbol : Visualizes the linear operator’s symbol in frequency space.
        analyze_wave_propagation : Analyzes group velocity and wavefront dynamics.
        interactive_symbol_analysis : Interactive tools for ψOp-based equations.
        """
        
        # Temporal parameters
        self.Lt, self.Nt = Lt, Nt
        self.dt = Lt / Nt
        self.n_frames = n_frames
        self.frames = []
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition

        if self.boundary_condition == 'dirichlet' and not self.has_psi:
            raise ValueError(
                "Dirichlet boundary conditions require the equation to be defined via a pseudo-differential operator (psiOp). "
                "Please provide an equation involving psiOp for non-periodic boundary treatment."
            )
    
        # Dimension checks
        if self.dim == 1:
            if Nx is None:
                raise ValueError("Nx must be specified in 1D.")
            self.setup_1D(Lx, Nx)
        else:
            if None in (Ly, Ny):
                raise ValueError("In 2D, Ly and Ny must be provided.")
            self.setup_2D(Lx, Ly, Nx, Ny)
    
        # Initialization of solution and velocities
        if not self.is_stationary:
            self.initialize_conditions(initial_condition, initial_velocity)
            
        # Symbol analysis if present
        if self.has_psi:
            print("⚠️ For psiOp, use interactive_symbol_analysis.")
        else:
            if self.L_symbolic == 0:
                print("⚠️ Linear operator is null.")
            else:
                self.check_cfl_condition()
                self.check_symbol_conditions()
                self.plot_symbol()
                if self.temporal_order == 2:
                    self.analyze_wave_propagation()

    def setup_1D(self, Lx, Nx):
        """
        Configure internal variables for one-dimensional (1D) problems.
    
        This private method initializes spatial and frequency grids, applies dealiasing,
        and prepares either pseudo-differential symbols or linear operators for use in time evolution.
        
        It assumes periodic boundary conditions and uses real-to-complex FFT conventions.
        The spatial domain is centered at zero: [-Lx/2, Lx/2].
    
        Parameters
        ----------
        Lx : float
            Physical size of the spatial domain along the x-axis.
        Nx : int
            Number of grid points in the x-direction.
    
        Attributes Set
        --------------
        self.Lx : float
            Size of the spatial domain.
        self.Nx : int
            Number of spatial points.
        self.x_grid : np.ndarray
            1D array of spatial coordinates.
        self.X : np.ndarray
            Alias to `self.x_grid`, used in physical space computations.
        self.kx : np.ndarray
            Array of wavenumbers corresponding to the Fourier transform.
        self.KX : np.ndarray
            Alias to `self.kx`, used in frequency space computations.
        self.dealiasing_mask : np.ndarray
            Boolean mask used to suppress aliased frequencies during nonlinear calculations.
        self.exp_L : np.ndarray
            Exponential of the linear operator scaled by time step: exp(L(k) · dt).
        self.omega_val : np.ndarray
            Frequency values ω(k) = Re[√(L(k))] used in second-order time stepping.
        self.cos_omega_dt, self.sin_omega_dt : np.ndarray
            Cosine and sine of ω(k)·dt for dispersive propagation.
        self.inv_omega : np.ndarray
            Inverse of ω(k), used to avoid division-by-zero in time stepping.
    
        Notes
        -----
        - Frequencies are computed using `scipy.fft.fftfreq` and then shifted to center zero frequency.
        - Dealiasing is applied using a sharp cutoff filter based on `self.dealiasing_ratio`.
        - If pseudo-differential operators (ψOp) are present, symbolic tables are precomputed via `prepare_symbol_tables`.
        - For second-order equations, the dispersion relation ω(k) is extracted from the linear operator L(k).
    
        See Also
        --------
        setup_2D : Equivalent setup for two-dimensional problems.
        prepare_symbol_tables : Precomputes symbolic arrays for ψOp evaluation.
        setup_omega_terms : Sets up terms involving ω(k) for second-order evolution.
        """
        self.Lx, self.Nx = Lx, Nx
        self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        self.X = self.x_grid
        self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
        self.KX = self.kx
    
        # Dealiasing mask
        k_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
        self.dealiasing_mask = (np.abs(self.KX) <= k_max)
    
        # Preparation of symbol or linear operator
        if self.has_psi:
            self.prepare_symbol_tables()
        else:
            L_vals = np.array(self.L(self.KX), dtype=np.complex128)
            self.exp_L = np.exp(L_vals * self.dt)
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX)
                self.setup_omega_terms(omega_val)
    
    def setup_2D(self, Lx, Ly, Nx, Ny):
        """
        Configure internal variables for two-dimensional (2D) problems.
    
        This private method initializes spatial and frequency grids, applies dealiasing,
        and prepares either pseudo-differential symbols or linear operators for use in time evolution.
        
        It assumes periodic boundary conditions and uses real-to-complex FFT conventions.
        The spatial domain is centered at zero: [-Lx/2, Lx/2] × [-Ly/2, Ly/2].
    
        Parameters
        ----------
        Lx : float
            Physical size of the spatial domain along the x-axis.
        Ly : float
            Physical size of the spatial domain along the y-axis.
        Nx : int
            Number of grid points along the x-direction.
        Ny : int
            Number of grid points along the y-direction.
    
        Attributes Set
        --------------
        self.Lx, self.Ly : float
            Size of the spatial domain in each direction.
        self.Nx, self.Ny : int
            Number of spatial points in each direction.
        self.x_grid, self.y_grid : np.ndarray
            1D arrays of spatial coordinates in x and y directions.
        self.X, self.Y : np.ndarray
            2D meshgrids of spatial coordinates for physical space computations.
        self.kx, self.ky : np.ndarray
            Arrays of wavenumbers corresponding to Fourier transforms in x and y directions.
        self.KX, self.KY : np.ndarray
            Meshgrids of wavenumbers used in frequency space computations.
        self.dealiasing_mask : np.ndarray
            Boolean mask used to suppress aliased frequencies during nonlinear calculations.
        self.exp_L : np.ndarray
            Exponential of the linear operator scaled by time step: exp(L(kx, ky) · dt).
        self.omega_val : np.ndarray
            Frequency values ω(kx, ky) = Re[√(L(kx, ky))] used in second-order time stepping.
        self.cos_omega_dt, self.sin_omega_dt : np.ndarray
            Cosine and sine of ω(kx, ky)·dt for dispersive propagation.
        self.inv_omega : np.ndarray
            Inverse of ω(kx, ky), used to avoid division-by-zero in time stepping.
    
        Notes
        -----
        - Frequencies are computed using `scipy.fft.fftfreq` and then shifted to center zero frequency.
        - Dealiasing is applied using a sharp cutoff filter based on `self.dealiasing_ratio`.
        - If pseudo-differential operators (ψOp) are present, symbolic tables are precomputed via `prepare_symbol_tables`.
        - For second-order equations, the dispersion relation ω(kx, ky) is extracted from the linear operator L(kx, ky).
    
        See Also
        --------
        setup_1D : Equivalent setup for one-dimensional problems.
        prepare_symbol_tables : Precomputes symbolic arrays for ψOp evaluation.
        setup_omega_terms : Sets up terms involving ω(kx, ky) for second-order evolution.
        """
        self.Lx, self.Ly = Lx, Ly
        self.Nx, self.Ny = Nx, Ny
        self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
        self.y_grid = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
        self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
        self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
        self.ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)
        self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
    
        # Dealiasing mask
        kx_max = self.dealiasing_ratio * np.max(np.abs(self.kx))
        ky_max = self.dealiasing_ratio * np.max(np.abs(self.ky))
        self.dealiasing_mask = (np.abs(self.KX) <= kx_max) & (np.abs(self.KY) <= ky_max)
    
        # Preparation of symbol or linear operator
        if self.has_psi:
            self.prepare_symbol_tables()
        else:
            L_vals = self.L(self.KX, self.KY)
            self.exp_L = np.exp(L_vals * self.dt)
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX, self.KY)
                self.setup_omega_terms(omega_val)
    
    def setup_omega_terms(self, omega_val):
        """
        Initialize terms derived from the angular frequency ω for time evolution.
    
        This private method precomputes and stores key trigonometric and inverse quantities
        based on the dispersion relation ω(k), used in second-order time integration schemes.
        
        These values are essential for solving wave-like equations with dispersive behavior:
            cos(ω·dt), sin(ω·dt), 1/ω
        
        The inverse frequency is computed safely to avoid division by zero.
    
        Parameters
        ----------
        omega_val : np.ndarray
            Array of angular frequency values ω(k) evaluated at discrete wavenumbers.
            Can be one-dimensional (1D) or two-dimensional (2D) depending on spatial dimension.
    
        Attributes Set
        --------------
        self.omega_val : np.ndarray
            Copy of the input angular frequency array.
        self.cos_omega_dt : np.ndarray
            Cosine of ω(k) multiplied by time step: cos(ω(k) · dt).
        self.sin_omega_dt : np.ndarray
            Sine of ω(k) multiplied by time step: sin(ω(k) · dt).
        self.inv_omega : np.ndarray
            Inverse of ω(k), with zeros where ω(k) == 0 to avoid division by zero.
    
        Notes
        -----
        - This method is typically called during setup when solving second-order PDEs
          involving dispersive waves (e.g., Klein-Gordon, Schrödinger, or water wave equations).
        - The safe computation of 1/ω ensures numerical stability even when low frequencies are present.
        - These precomputed arrays are used in spectral propagators for accurate time stepping.
    
        See Also
        --------
        setup_1D : Sets up internal variables for one-dimensional problems.
        setup_2D : Sets up internal variables for two-dimensional problems.
        solve : Time integration using the computed frequency terms.
        """
        self.omega_val = omega_val
        self.cos_omega_dt = np.cos(omega_val * self.dt)
        self.sin_omega_dt = np.sin(omega_val * self.dt)
        self.inv_omega = np.zeros_like(omega_val)
        nonzero = omega_val != 0
        self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]

    def evaluate_source_at_t0(self):
        """
        Evaluate source terms at initial time t = 0 over the spatial grid.
    
        This private method computes the total contribution of all source terms at the initial time,
        evaluated across the entire spatial domain. It supports both one-dimensional (1D) and
        two-dimensional (2D) configurations.
    
        Returns
        -------
        np.ndarray
            A numpy array representing the evaluated source term at t=0:
            - In 1D: Shape (Nx,), evaluated at each x in `self.x_grid`.
            - In 2D: Shape (Nx, Ny), evaluated at each (x, y) pair in the grid.
    
        Notes
        -----
        - The symbolic expressions in `self.source_terms` are substituted with numerical values at t=0.
        - In 1D, each term is evaluated at (t=0, x=x_val).
        - In 2D, each term is evaluated at (t=0, x=x_val, y=y_val).
        - Evaluated using SymPy's `evalf()` to ensure numeric conversion.
        - This method assumes that the source terms have already been lambdified or are compatible with symbolic substitution.
    
        See Also
        --------
        setup : Initializes the spatial grid and source terms.
        solve : Uses this evaluation during the first time step.
        """
        if self.dim == 1:
            # Evaluation on the 1D spatial grid
            return np.array([
                sum(term.subs(self.t, 0).subs(self.x, x_val).evalf()
                    for term in self.source_terms)
                for x_val in self.x_grid
            ], dtype=np.float64)
        else:
            # Evaluation on the 2D spatial grid
            return np.array([
                [sum(term.subs({self.t: 0, self.x: x_val, self.y: y_val}).evalf()
                      for term in self.source_terms)
                 for y_val in self.y_grid]
                for x_val in self.x_grid
            ], dtype=np.float64)
    
    def initialize_conditions(self, initial_condition, initial_velocity):
        """
        Initialize the solution and velocity fields at t = 0.
    
        This private method sets up the initial state of the solution `u_prev` and, if applicable,
        the time derivative (velocity) `v_prev` for second-order evolution equations.
        
        For second-order equations, it also computes the backward-in-time value `u_prev2`
        needed by the Leap-Frog method. The acceleration at t = 0 is computed from:
            ∂ₜ²u = L(u) + N(u) + f(x, t=0)
        where L is the linear operator, N is the nonlinear term, and f is the source term.
    
        Parameters
        ----------
        initial_condition : callable
            Function returning the initial condition u(x, 0) or u(x, y, 0).
        initial_velocity : callable or None
            Function returning the initial velocity ∂ₜu(x, 0) or ∂ₜu(x, y, 0). Required for
            second-order equations; ignored otherwise.
    
        Raises
        ------
        ValueError
            If `initial_velocity` is not provided for second-order equations.
    
        Notes
        -----
        - Applies periodic boundary conditions after setting initial data.
        - Stores a copy of the initial state in `self.frames` for visualization/output.
        - In second-order systems, initializes `self.u_prev2` using a Taylor expansion:
          u_prev2 = u_prev - dt * v_prev + 0.5 * dt² * (∂ₜ²u)
    
        See Also
        --------
        apply_boundary : Enforces periodic boundary conditions on the solution field.
        psiOp_apply : Computes pseudo-differential operator action for acceleration.
        linear_rhs : Evaluates linear part of the equation in Fourier space.
        apply_nonlinear : Handles nonlinear terms with spectral differentiation.
        evaluate_source_at_t0 : Evaluates source terms at the initial time.
        """
        # Initial condition
        if self.dim == 1:
            self.u_prev = initial_condition(self.X)
        else:
            self.u_prev = initial_condition(self.X, self.Y)
        self.apply_boundary(self.u_prev)
    
        # Initial velocity (second order)
        if self.temporal_order == 2:
            if initial_velocity is None:
                raise ValueError("Initial velocity is required for second-order equations.")
            if self.dim == 1:
                self.v_prev = initial_velocity(self.X)
            else:
                self.v_prev = initial_velocity(self.X, self.Y)
            self.u0 = np.copy(self.u_prev)
            self.v0 = np.copy(self.v_prev)
    
            # Calculation of u_prev2 (initial acceleration)
            if not hasattr(self, 'u_prev2'):
                if self.has_psi:
                    acc0 = self.apply_psiOp(self.u_prev)
                else:
                    acc0 = self.linear_rhs(self.u_prev, is_v=False)
                rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                acc0 += rhs_nl
                if hasattr(self, 'source_terms') and self.source_terms:
                    acc0 += self.evaluate_source_at_t0()
                self.u_prev2 = self.u_prev - self.dt * self.v_prev + 0.5 * self.dt**2 * acc0
    
        self.frames = [self.u_prev.copy()]
           
    def apply_boundary(self, u):
        """
        Apply boundary conditions to the solution array based on the specified type.
    
        This method supports two types of boundary conditions:
        
        - 'periodic': Enforces periodicity by copying opposite boundary values.
        - 'dirichlet': Sets all boundary values to zero (homogeneous Dirichlet condition).
    
        Parameters
        ----------
        u : np.ndarray
            The solution array representing the field values on a spatial grid.
            In 1D, shape must be (Nx,). In 2D, shape must be (Nx, Ny).
    
        Raises
        ------
        ValueError
            If `self.boundary_condition` is not one of {'periodic', 'dirichlet'}.
    
        Notes
        -----
        - For 'periodic':
            * In 1D: u[0] = u[-2], u[-1] = u[1]
            * In 2D: First and last rows/columns are set equal to their neighbors.
        - For 'dirichlet':
            * All boundary points are explicitly set to zero.
        """
    
        if self.boundary_condition == 'periodic':
            if self.dim == 1:
                u[0] = u[-2]
                u[-1] = u[1]
            elif self.dim == 2:
                u[0, :] = u[-2, :]
                u[-1, :] = u[1, :]
                u[:, 0] = u[:, -2]
                u[:, -1] = u[:, 1]
    
        elif self.boundary_condition == 'dirichlet':
            if self.dim == 1:
                u[0] = 0
                u[-1] = 0
            elif self.dim == 2:
                u[0, :] = 0
                u[-1, :] = 0
                u[:, 0] = 0
                u[:, -1] = 0
    
        else:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary_condition}'. "
                "Supported types are 'periodic' and 'dirichlet'."
            )

    def apply_nonlinear(self, u, is_v=False):
        """
        Apply nonlinear terms to the solution using spectral differentiation with dealiasing.

        This method evaluates all nonlinear terms present in the PDE by substituting spatial 
        derivatives with their spectral approximations computed via FFT. The dealiasing mask 
        ensures numerical stability by removing high-frequency components that could lead 
        to aliasing errors.

        Parameters:
            u (numpy.ndarray): Current solution array on the spatial grid.
            is_v (bool): If True, evaluates nonlinear terms for the velocity field v instead of u.

        Returns:
            numpy.ndarray: Array representing the contribution of nonlinear terms multiplied by dt.

        Notes:
        - In 1D, computes ∂ₓu via FFT and substitutes any derivative term in the nonlinear expressions.
        - In 2D, computes ∂ₓu and ∂ᵧu via FFT and performs similar substitutions.
        - Uses lambdify to evaluate symbolic nonlinear expressions numerically.
        - Derivatives are replaced symbolically with 'u_x' and 'u_y' before evaluation.
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
        """
        Precompute and store evaluated pseudo-differential operator symbols for spectral methods.

        This method evaluates all pseudo-differential operators (ψOp) present in the PDE
        over the spatial and frequency grids, scales them by their respective coefficients,
        and combines them into a single composite symbol used in time-stepping and inversion.

        The evaluation is performed via the `evaluate` method of each PseudoDifferentialOperator,
        which computes p(x, ξ) or p(x, y, ξ, η) numerically over the current grid configuration.

        Side Effects:
            self.precomputed_symbols : list of (coeff, symbol_array)
                Each tuple contains a coefficient and its evaluated symbol on the grid.
            self.combined_symbol : np.ndarray
                Sum of all scaled symbol arrays: ∑(coeffₖ * ψₖ(x, ξ))

        Raises:
            ValueError: If the spatial dimension is not 1D or 2D.
        """
        self.precomputed_symbols = []
        self.combined_symbol = 0
        for coeff, psi in self.psi_ops:
            if self.dim == 1:
                raw = psi.evaluate(self.X, None, self.KX, None)
            elif self.dim == 2:
                raw = psi.evaluate(self.X, self.Y, self.KX, self.KY)
            else:
                raise ValueError('Unsupported spatial dimension.')
            raw_flat = raw.flatten()
            converted = np.array([complex(N(val)) for val in raw_flat], dtype=np.complex128)
            raw_eval = converted.reshape(raw.shape)
            self.precomputed_symbols.append((coeff, raw_eval))
        self.combined_symbol = sum((coeff * sym for coeff, sym in self.precomputed_symbols))
        self.combined_symbol = np.array(self.combined_symbol, dtype=np.complex128)

    def total_symbol_expr(self):
        """
        Compute the total pseudo-differential symbol expression from all pseudo_terms.

        This method constructs the full symbol of the pseudo-differential operator
        by summing up all coefficient-weighted symbolic expressions.

        The result is cached in self.symbol_expr to avoid recomputation.

        Returns:
            sympy.Expr: The combined symbol expression, representing the full
                        pseudo-differential operator in symbolic form.

        Example:
            Given pseudo_terms = [(2, ξ²), (1, x·ξ)], this returns 2·ξ² + x·ξ.
        """
        if not hasattr(self, '_symbol_expr'):
            self.symbol_expr = sum(coeff * expr for coeff, expr in self.pseudo_terms)
        return self.symbol_expr

    def build_symbol_func(self, expr):
        """
        Build a numerical evaluation function from a symbolic pseudo-differential operator expression.
    
        This method converts a symbolic expression representing a pseudo-differential operator into
        a callable NumPy-compatible function. The function accepts spatial and frequency variables
        depending on the dimensionality of the problem.
    
        Parameters:
            expr (sympy.Expr): A SymPy expression representing the symbol of the pseudo-differential operator.
                                It may depend on spatial variables (x, y) and frequency variables (xi, eta).
    
        Returns:
            function: A lambdified function that takes:
                - In 1D: `(x, xi)` — spatial coordinate and frequency.
                - In 2D: `(x, y, xi, eta)` — spatial coordinates and frequencies.
              Returns a NumPy array of evaluated symbol values over input grids.
    
        Notes:
            - Uses `lambdify` from SymPy with the `'numpy'` backend for efficient vectorized evaluation.
            - Real variable assumptions are enforced to ensure proper behavior in numerical contexts.
            - Used internally by methods like `apply_psiOp`, `evaluate`, and visualization tools.
        """
        if self.dim == 1:
            x, xi = symbols('x xi', real=True)
            return lambdify((x, xi), expr, 'numpy')
        else:
            x, y, xi, eta = symbols('x y xi eta', real=True)
            return lambdify((x, y, xi, eta), expr, 'numpy')

    def apply_psiOp(self, u):
        """
        Apply the pseudo-differential operator to the input field u.
    
        This method dispatches the application of the pseudo-differential operator based on:
        - Whether the symbol is spatially dependent (x/y)
        - The boundary condition in use (periodic or dirichlet)
    
        Supported operations:
        - Constant-coefficient symbols: applied via Fourier multiplication.
        - Spatially varying symbols: applied via Kohn–Nirenberg quantization.
        - Dirichlet boundary conditions: handled with non-periodic convolution-like quantization.
    
        Dispatch Logic:
        if not self.is_spatial:
            u ↦ Op(p)(D) ⋅ u = 𝓕⁻¹[ p(ξ) ⋅ 𝓕(u) ]
        elif periodic:
            u ↦ Op(p)(x,D) ⋅ u ≈ ∫ eᶦˣᶿ p(x, ξ) 𝓕(u)(ξ) dξ
        elif dirichlet:
            u ↦ Op(p)(x,y,D) ⋅ u ≈ non-periodic extension + convolution
    
        Parameters:
            u (np.ndarray): Input field to which the operator is applied.
                            Should be 1D or 2D depending on the problem dimension.
    
        Returns:
            np.ndarray: Result of applying the pseudo-differential operator to u.
    
        Raises:
            ValueError: If an unsupported boundary condition is specified.
        """
        if self.verbose:
            print("apply_psiOp")
        if not self.is_spatial:
            return self.apply_psiOp_constant(u)
        elif self.boundary_condition == 'periodic':
            return self.apply_psiOp_kohn_nirenberg_fft(u)
        elif self.boundary_condition == 'dirichlet':
            return self.apply_psiOp_kohn_nirenberg_nonperiodic(u)
        else:
            raise ValueError(f"Invalid boundary condition '{self.boundary_condition}'")

    def apply_psiOp_constant(self, u):
        """
        Apply a constant-coefficient pseudo-differential operator in Fourier space.

        This method assumes the symbol is diagonal in the Fourier basis and acts as a 
        multiplication operator. It performs the operation:
        
            (ψu)(x) = 𝓕⁻¹[ -σ(k) · 𝓕[u](k) ]

        where:
        - σ(k) is the combined pseudo-differential operator symbol
        - 𝓕 denotes the forward Fourier transform
        - 𝓕⁻¹ denotes the inverse Fourier transform

        The dealiasing mask is applied before returning to physical space.
        
        Parameters:
        -----------
        u : numpy.ndarray
            Input function in physical space (real-valued or complex-valued)

        Returns:
        --------
        numpy.ndarray
            Result of applying the pseudo-differential operator to u, same shape as input
        """
        if self.verbose:
            print("apply_psiOp_constant")
        u_hat = self.fft(u)
        u_hat *= -self.combined_symbol
        u_hat *= self.dealiasing_mask
        return self.ifft(u_hat)

    def apply_psiOp_kohn_nirenberg_fft(self, u):
        """
        Apply a pseudo-differential operator using the Kohn–Nirenberg quantization in Fourier space.
    
        This method evaluates the action of a pseudo-differential operator defined by the total symbol,
        computed from all psiOp terms in the equation. It uses the fast Fourier transform (FFT) for
        efficiency in periodic domains.
    
        Parameters:
            u (np.ndarray): Input function in real space to which the operator is applied.
    
        Returns:
            np.ndarray: Resulting function after applying the pseudo-differential operator.
    
        Process:
            1. Compute the total symbolic expression of the pseudo-differential operator.
            2. Build a callable numerical function from the symbol.
            3. Evaluate Op(p)(u) via the Kohn–Nirenberg quantization using FFT.
    
        Note:
            - Assumes periodic boundary conditions.
            - The returned result is the negative of the standard definition due to PDE sign conventions.
        """
        if self.verbose:
            print("apply_psiOp_kohn_nirenberg_fft")
        total_symbol = self.total_symbol_expr()
        symbol_func = self.build_symbol_func(total_symbol)
        return -self.kohn_nirenberg_fft(u_vals=u, symbol_func=symbol_func)

    def apply_psiOp_kohn_nirenberg_nonperiodic(self, u):
        """
        Apply a pseudo-differential operator using the Kohn–Nirenberg quantization on non-periodic domains.
    
        This method evaluates the action of a pseudo-differential operator Op(p) on a function u 
        via the Kohn–Nirenberg representation. It supports both 1D and 2D cases and uses spatial 
        and frequency grids to evaluate the operator symbol p(x, ξ).
    
        The operator symbol p(x, ξ) is extracted from the PDE and evaluated numerically using 
        `_total_symbol_expr` and `_build_symbol_func`.
    
        Parameters:
            u (np.ndarray): Input function (real space) to which the operator is applied.
    
        Returns:
            np.ndarray: Result of applying Op(p) to u in real space.
    
        Notes:
            - For 1D: p(x, ξ) is evaluated over x_grid and xi_grid.
            - For 2D: p(x, y, ξ, η) is evaluated over (x_grid, y_grid) and (xi_grid, eta_grid).
            - The result is computed using `kohn_nirenberg_nonperiodic`, which handles non-periodic boundary conditions.
        """
        if self.verbose:
            print("apply_psiOp_kohn_nirenberg_nonperiodic")
        total_symbol = self.total_symbol_expr()
        symbol_func = self.build_symbol_func(total_symbol)
        if self.dim == 1:
            return -self.kohn_nirenberg_nonperiodic(u_vals=u, x_grid=self.x_grid, xi_grid=self.kx, symbol_func=symbol_func)
        else:
            return -self.kohn_nirenberg_nonperiodic(u_vals=u, x_grid=(self.x_grid, self.y_grid), xi_grid=(self.kx, self.ky), symbol_func=symbol_func)
    
    
    def step_order1_with_psi(self, source_contribution):
        """
        Advance the solution by one time step for first-order PDEs with pseudo-differential operators (ψOp).
    
        This method integrates the solution forward in time using an exponential integrator scheme.
        It supports both periodic boundary conditions with direct Fourier space evolution,
        and general spatially varying symbols using ETD1 (Exponential Time Differencing of order 1).
    
        Algorithm:
        ──────────
        For periodic boundary conditions and constant-in-space symbols:
    
            u_hat = FFT(uₙ)
            u_hat ← exp(-Δt ⋅ L) ⋅ u_hat
            u_symb = IFFT(u_hat)
            u_new = u_symb + N(uₙ) + source
    
        For non-periodic or spatially varying symbols (ETD1):
    
            L = combined symbol from ψOp
            exp_L = exp(-Δt ⋅ L)
            phi1_L = (exp_L - 1) / (-Δt ⋅ L), with limit handling
            u_hat_new = exp_L ⋅ FFT(uₙ) + Δt ⋅ phi1_L ⋅ (FFT(N(uₙ)) + FFT(source))
            u_new = IFFT(u_hat_new)
    
        where:
            uₙ      = current solution
            N(uₙ)   = nonlinear terms evaluated at uₙ
            L       = linear operator (from ψOp)
            Δt      = time step size
    
        Parameters:
        ────────────
        source_contribution : float or np.ndarray
            External source term to be added during the update. If scalar, a zero-filled array matching
            the shape of self.u_prev is created.
    
        Returns:
        ─────────
        u_new : np.ndarray
            The updated solution after one time step.
    
        Notes:
        ──────
        - Recomputes the symbol if it depends on spatial variables (self.is_spatial).
        - Applies dealiasing mask in Fourier space to avoid aliasing errors.
        - Supports both 1D and 2D configurations seamlessly.
        - Handles division-by-zero in phi1_L safely via np.isnan replacement.
        """
        if self.verbose:
            print("step_order1_with_psi")
        # Handling null source
        if np.isscalar(source_contribution):
            source = np.zeros_like(self.u_prev)
        else:
            source = source_contribution
    
        # Recalculate symbol if necessary
        if self.is_spatial:
            self.prepare_symbol_tables()  # Recalculates self.combined_symbol
    
        # Case with FFT (symbol diagonalizable in Fourier space)
        if self.boundary_condition == 'periodic' and not self.is_spatial:
            if self.verbose:
                print("periodic free of spatial variables")
            u_hat = self.fft(self.u_prev)
            u_hat *= np.exp(-self.dt * self.combined_symbol)
            u_hat *= self.dealiasing_mask
            u_symb = self.ifft(u_hat)
            u_nl = self.apply_nonlinear(self.u_prev)
            u_new = u_symb + u_nl + source
        else:
            if self.is_spatial:
                print("⚠️ For psiOp depending on spatial variables and temporal order 1") 
                print("⚠️ the method is not implemented, it simply uses symbol evaluted") 
                print("⚠️ at this point, which is FALSE !!!") 
            if self.verbose:
                print("ETD1")
            # General case with ETD1
            u_nl = self.apply_nonlinear(self.u_prev)
    
            # Calculation of exp(dt * L) and phi1(dt * L)
            L_vals = self.combined_symbol  # Uses the updated symbol
            exp_L = np.exp(-self.dt * L_vals)
            phi1_L = (exp_L - 1.0) / (self.dt * L_vals)
            phi1_L[np.isnan(phi1_L)] = 1.0  # Handling division by zero
    
            # Fourier transform
            u_hat = self.fft(self.u_prev)
            u_nl_hat = self.fft(u_nl)
            source_hat = self.fft(source)
    
            # Assembling the solution in Fourier space
            u_hat_new = exp_L * u_hat + self.dt * phi1_L * (u_nl_hat + source_hat)
            u_new = self.ifft(u_hat_new)
#            else:
#                if self.verbose:
#                    print("With of spatial variables")
#                u_nl = self.apply_nonlinear(self.u_prev)
#                Lu_prev_1 = self.apply_psiOp(self.u_prev)
#                u_prev_2 = self.u_prev + self.dt * Lu_prev_1
#                Lu_prev_2 = self.apply_psiOp(u_prev_2)
#                u_new =  self.u_prev + 0.5 * self.dt * (Lu_prev_1 + Lu_prev_2)
                
    
        # Applying boundary conditions
        self.apply_boundary(u_new)
        return u_new

    def step_order2_with_psi(self, source_contribution):
        """
        Perform one time step of a second-order time evolution using a pseudo-differential operator.
    
        This method updates the solution field using a second-order accurate scheme suitable for wave-like equations.
        The update includes contributions from:
        - Linear dynamics via a pseudo-differential operator (e.g., dispersion or stiffness)
        - Nonlinear terms computed via spectral differentiation
        - External source contributions
    
        Discretization follows a leapfrog-style finite difference in time:
        
            uₙ₊₁ = 2uₙ − uₙ₋₁ + Δt² ⋅ (L(uₙ) + N(uₙ) + F)
    
        where:
            L(uₙ) = linear part evaluated via pseudo-differential operator
            N(uₙ) = nonlinear contribution at current time step
            F     = external source term at current time step
            Δt    = time step size
    
        Boundary conditions are applied after each update to ensure consistency.
    
        Args:
            source_contribution (np.ndarray): Array representing the external source term at current time step.
                                              Must match the spatial dimensions of self.u_prev.
    
        Returns:
            np.ndarray: Updated solution array after one time step.
        """
        if self.verbose:
            print("step_order2_with_psi")
        Lu_prev = self.apply_psiOp(self.u_prev)
        rhs_nl = self.apply_nonlinear(self.u_prev, is_v=False)
        u_new = 2 * self.u_prev - self.u_prev2 + self.dt ** 2 * (Lu_prev + rhs_nl + source_contribution)
        self.apply_boundary(u_new)
        self.u_prev2 = self.u_prev
        self.u_prev = u_new
        self.u = u_new
        return u_new

    def solve(self):
        """
        Solve the partial differential equation numerically using spectral methods.
        
        This method evolves the solution in time using a combination of:
        - Fourier-based linear evolution (with dealiasing)
        - Nonlinear term handling via pseudo-spectral evaluation
        - Support for pseudo-differential operators (psiOp)
        - Source terms and boundary conditions
        
        The solver supports:
        - 1D and 2D spatial domains
        - First and second-order time evolution
        - Periodic and Dirichlet boundary conditions
        - Time-stepping schemes: default, ETD-RK4
        
        Returns:
            list[np.ndarray]: A list of solution arrays at each saved time frame.
        
        Side Effects:
            - Updates self.frames: stores solution snapshots
            - Updates self.energy_history: records total energy if enabled
            
        Algorithm Overview:
            For each time step:
                1. Evaluate source contributions (if any)
                2. Apply time evolution:
                    - Order 1:
                        - With psiOp: uses step_order1_with_psi
                        - With ETD-RK4: exponential time differencing
                        - Default: linear + nonlinear update
                    - Order 2:
                        - With psiOp: uses step_order2_with_psi
                        - With ETD-RK4: second-order exponential scheme
                        - Default: second-order leapfrog-style update
                3. Enforce boundary conditions
                4. Save solution snapshot periodically
                5. Record energy (for second-order systems without psiOp)
        """
        print('\n*******************')
        print('* Solving the PDE *')
        print('*******************\n')
        save_interval = max(1, self.Nt // self.n_frames)
        self.energy_history = []
        for step in range(self.Nt):
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
                        print(f'Error evaluating source term {term}: {e}')
            else:
                source_contribution = 0

            if self.temporal_order == 1:
                if self.has_psi:
                    u_new = self.step_order1_with_psi(source_contribution)
                elif hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                    u_new = self.step_ETD_RK4(self.u_prev)
                else:
                    u_hat = self.fft(self.u_prev)
                    u_hat *= self.exp_L
                    u_hat *= self.dealiasing_mask
                    u_lin = self.ifft(u_hat)
                    u_nl = self.apply_nonlinear(u_lin)
                    u_new = u_lin + u_nl + source_contribution
                self.apply_boundary(u_new)
                self.u_prev = u_new

            elif self.temporal_order == 2:
                if self.has_psi:
                    u_new = self.step_order2_with_psi(source_contribution)
                else:
                    if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                        u_new, v_new = self.step_ETD_RK4_order2(self.u_prev, self.v_prev)
                    else:
                        u_hat = self.fft(self.u_prev)
                        v_hat = self.fft(self.v_prev)
                        u_new_hat = self.cos_omega_dt * u_hat + self.sin_omega_dt * self.inv_omega * v_hat
                        v_new_hat = -self.omega_val * self.sin_omega_dt * u_hat + self.cos_omega_dt * v_hat
                        u_new = self.ifft(u_new_hat)
                        v_new = self.ifft(v_new_hat)
                        u_nl = self.apply_nonlinear(self.u_prev, is_v=False)
                        v_nl = self.apply_nonlinear(self.v_prev, is_v=True)
                        u_new += (u_nl + source_contribution) * self.dt ** 2 / 2
                        v_new += (u_nl + source_contribution) * self.dt
                    self.apply_boundary(u_new)
                    self.apply_boundary(v_new)
                    self.u_prev = u_new
                    self.v_prev = v_new

            if step % save_interval == 0:
                self.frames.append(self.u_prev.copy())

            if self.temporal_order == 2 and (not self.has_psi):
                E = self.compute_energy()
                self.energy_history.append(E)

        return self.frames  
                
    def solve_stationary_psiOp(self, order=3):
        """
        Solve stationary pseudo-differential equations of the form P[u] = f(x) or P[u] = f(x,y) using asymptotic inversion.
    
        This method computes the solution to a stationary (time-independent) pseudo-differential equation
        where the operator P is defined via symbolic expressions (psiOp). It constructs an asymptotic right inverse R 
        such that P∘R ≈ Id, then applies it to the source term f using either direct Fourier multiplication 
        (when the symbol is spatially independent) or Kohn–Nirenberg quantization (when spatial dependence is present).
    
        The inversion is based on the principal symbol of the operator and its asymptotic expansion up to the given order.
        Ellipticity of the symbol is checked numerically before inversion to ensure well-posedness.
    
        Parameters
        ----------
        order : int, default=3
            Order of the asymptotic expansion used to construct the right inverse of the pseudo-differential operator.
        method : str, optional
            Inversion strategy:
            - 'diagonal' (default): Fast approximate inversion using diagonal operators in frequency space.
            - 'full'                : Pointwise exact inversion (slower but more accurate).
    
        Returns
        -------
        ndarray
            The computed solution u(x) in 1D or u(x, y) in 2D as a NumPy array over the spatial grid.
    
        Raises
        ------
        ValueError
            If no pseudo-differential operator (psiOp) is defined.
            If linear or nonlinear terms other than psiOp are present.
            If the symbol is not elliptic on the grid.
            If no source term is provided for the right-hand side.
    
        Notes
        -----
        - The method assumes the problem is fully stationary: time derivatives must be absent.
        - Requires the equation to be purely pseudo-differential (no Op, Derivative, or nonlinear terms).
        - Symbol evaluation and inversion are dimension-aware (supports both 1D and 2D problems).
        - Supports optimization paths when the symbol does not depend on spatial variables.
    
        See Also
        --------
        right_inverse_asymptotic : Constructs the asymptotic inverse of the pseudo-differential operator.
        kohn_nirenberg           : Numerical implementation of general pseudo-differential operators.
        is_elliptic_numerically  : Verifies numerical ellipticity of the symbol.
        """

        print("\n*******************************")
        print("* Solving the stationnary PDE *")
        print("*******************************\n")
        print("boundary condition: ",self.boundary_condition)
        

        if not self.has_psi:
            raise ValueError("Only supports problems with psiOp.")
    
        if self.linear_terms or self.nonlinear_terms:
            raise ValueError("Stationary psiOp problems must be linear and purely pseudo-differential.")

        if self.boundary_condition not in ('periodic', 'dirichlet'):
            raise ValueError(
                "For stationary PDEs, boundary conditions must be explicitly defined. "
                "Supported types are 'periodic' and 'dirichlet'."
            )    
            
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
        pprint(R_symbol, num_columns=150)

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
    
        if self.boundary_condition == 'periodic':
            if self.dim == 1:
                if not R_symbol.has(x):
                    print("⚡ Optimization: symbol independent of x — direct product in Fourier.")
                    R_vals = R_func(self.KX)
                    u_hat = R_vals * f_hat
                    u = self.ifft(u_hat)
                else:
                    print("⚙️ 1D Kohn-Nirenberg Quantification")
                    x, xi = symbols('x xi', real=True)
                    R_func = lambdify((x, xi), R_symbol, 'numpy')  # Still 2 args for uniformity
                    u = self.kohn_nirenberg_fft(u_vals=rhs, symbol_func=R_func)
                    
            elif self.dim == 2:
                if not R_symbol.has(x) and not R_symbol.has(y):
                    print("⚡ Optimization: Symbol independent of x and y — direct product in 2D Fourier.")
                    R_vals = np.vectorize(R_func)(self.KX, self.KY)
                    u_hat = R_vals * f_hat
                    u = self.ifft(u_hat)
                else:
                    print("⚙️ 2D Kohn-Nirenberg Quantification")
                    x, xi, y, eta = symbols('x xi y eta', real=True)
                    R_func = lambdify((x, y, xi, eta), R_symbol, 'numpy')  # Still 2 args for uniformity
                    u = self.kohn_nirenberg_fft(u_vals=rhs, symbol_func=R_func)
            self.u = u
            return u
        elif self.boundary_condition == 'dirichlet':
            if self.dim == 1:
                x, xi = symbols('x xi', real=True)
                R_func = lambdify((x, xi), R_symbol, 'numpy')
                u = self.kohn_nirenberg_nonperiodic(u_vals=rhs, x_grid=X, xi_grid=KX, symbol_func=R_func)
            elif self.dim == 2:
                x, xi, y, eta = symbols('x xi y eta', real=True)
                R_func = lambdify((x, y, xi, eta), R_symbol, 'numpy')
                u = self.kohn_nirenberg_nonperiodic(u_vals=rhs, x_grid=(self.x_grid, self.y_grid), xi_grid=(self.kx, self.ky), symbol_func=R_func)
            self.u = u
            return u   
        else:
            raise ValueError(
                f"Invalid boundary condition '{self.boundary_condition}'. "
                "Supported types are 'periodic' and 'dirichlet'."
            )

    def kohn_nirenberg_fft(self, u_vals, symbol_func,
                           freq_window='gaussian', clamp=1e6,
                           space_window=False):
        """
        Numerically stable Kohn–Nirenberg quantization of a pseudo-differential operator.
        
        Applies the pseudo-differential operator Op(p) to the function f via the Kohn–Nirenberg quantization:
        
            [Op(p)f](x) = (1/(2π)^d) ∫ p(x, ξ) e^{ix·ξ} ℱ[f](ξ) dξ
        
        where p(x, ξ) is a symbol that may depend on both spatial variables x and frequency variables ξ.
        
        This method supports both 1D and 2D cases and includes optional smoothing techniques to improve numerical stability.
    
        Parameters
        ----------
        u_vals : np.ndarray
            Spatial samples of the input function f(x) or f(x, y), defined on a uniform grid.
        symbol_func : callable
            A function representing the full symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
            Must accept NumPy-compatible array inputs and return a complex-valued array.
        freq_window : {'gaussian', 'hann', None}, optional
            Type of frequency-domain window to apply:
            - 'gaussian': smooth decay near high frequencies
            - 'hann': cosine-based tapering with hard cutoff
            - None: no frequency window applied
        clamp : float, optional
            Upper bound on the absolute value of the symbol. Prevents numerical blow-up from large values.
        space_window : bool, optional
            Whether to apply a spatial Gaussian window to suppress edge effects in physical space.
    
        Returns
        -------
        np.ndarray
            The result of applying the pseudo-differential operator to f, returned as a real or complex array
            of the same shape as u_vals.
    
        Notes
        -----
        - The implementation uses FFT-based quadrature of the inverse Fourier transform.
        - Symbol evaluation is vectorized over spatial and frequency grids.
        - Frequency and spatial windows help mitigate oscillatory behavior and aliasing.
        - In 2D, the integration is performed over a 4D tensor product grid (x, y, ξ, η).
        """
        # === Common setup ===
        xg = self.x_grid
        dx = xg[1] - xg[0]
    
        if self.dim == 1:
            # === 1D case ===
    
            # Frequency grid (shifted to center zero)
            Nx = self.Nx
            k = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
            dk = k[1] - k[0]
    
            # Centered FFT of input
            f_shift = fftshift(u_vals)
            f_hat = self.fft(f_shift) * dx
            f_hat = fftshift(f_hat)
    
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
            kx = 2 * np.pi * fftshift(fftfreq(Nx, d=dx))
            ky = 2 * np.pi * fftshift(fftfreq(Ny, d=dy))
            dkx = kx[1] - kx[0]
            dky = ky[1] - ky[0]
    
            # 2D FFT of f(x, y)
            f_shift = fftshift(u_vals)
            f_hat = self.fft(f_shift) * dx * dy
            f_hat = fftshift(f_hat)
    
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
        
    def kohn_nirenberg_nonperiodic(self, u_vals, x_grid, xi_grid, symbol_func,
                                   freq_window='gaussian', clamp=1e6, space_window=False):
        """
        Numerically applies the Kohn–Nirenberg quantization of a pseudo-differential operator 
        in a non-periodic setting.
    
        This method computes:
        
        [Op(p)u](x) = (1/(2π)^d) ∫ p(x, ξ) e^{i x·ξ} ℱ[u](ξ) dξ
        
        where p(x, ξ) is a general symbol that may depend on both spatial and frequency variables.
        It supports both 1D and 2D inputs and includes optional numerical smoothing techniques 
        to enhance stability for non-smooth or oscillatory symbols.
    
        Parameters
        ----------
        u_vals : np.ndarray
            Input function values defined on a uniform spatial grid. Can be 1D (Nx,) or 2D (Nx, Ny).
        x_grid : np.ndarray
            Spatial grid points along each axis. In 1D: shape (Nx,). In 2D: tuple of two arrays (X, Y)
            or list of coordinate arrays.
        xi_grid : np.ndarray
            Frequency grid points. In 1D: shape (Nxi,). In 2D: tuple of two arrays (Xi, Eta)
            or list of frequency arrays.
        symbol_func : callable
            A function representing the full symbol p(x, ξ) in 1D or p(x, y, ξ, η) in 2D.
            Must accept NumPy-compatible array inputs and return a complex-valued array.
        freq_window : {'gaussian', 'hann', None}, optional
            Type of frequency-domain window to apply for regularization:
            
            - 'gaussian': Smooth exponential decay near high frequencies.
            - 'hann': Cosine-based tapering with hard cutoff.
            - None: No frequency window applied.
        clamp : float, optional
            Maximum absolute value allowed for the symbol to prevent numerical overflow.
            Default is 1e6.
        space_window : bool, optional
            If True, applies a smooth spatial Gaussian window centered in the domain to reduce
            boundary artifacts. Default is False.
    
        Returns
        -------
        np.ndarray
            The result of applying the pseudo-differential operator Op(p) to u. Shape matches u_vals.
        
        Notes
        -----
        - This version does not assume periodicity and is suitable for Dirichlet or Neumann boundary conditions.
        - In 1D, the integral is evaluated as a sum over (x, ξ), using matrix exponentials.
        - In 2D, the integration is performed over a 4D tensor product grid (x, y, ξ, η), which can be computationally intensive.
        - Symbol evaluation should be vectorized for performance.
        - For large grids, consider reducing resolution via resampling before calling this function.
    
        See Also
        --------
        kohn_nirenberg_fft : Faster implementation for periodic domains using FFT.
        PseudoDifferentialOperator : Class for symbolic manipulation of pseudo-differential operators.
        """
        if u_vals.ndim == 1:
            # === 1D case ===
            x = x_grid
            xi = xi_grid
            dx = x[1] - x[0]
            dxi = xi[1] - xi[0]
    
            phase_ft = np.exp(-1j * np.outer(xi, x))  # (Nxi, Nx)
            u_hat = dx * np.dot(phase_ft, u_vals)     # (Nxi,)
    
            X, XI = np.meshgrid(x, xi, indexing='ij')  # (Nx, Nxi)
            sigma_vals = symbol_func(X, XI)
    
            # Clamp values
            sigma_vals = np.clip(sigma_vals, -clamp, clamp)
    
            # Frequency window
            if freq_window == 'gaussian':
                sigma = 0.8 * np.max(np.abs(XI))
                window = np.exp(-(XI / sigma)**4)
                sigma_vals *= window
            elif freq_window == 'hann':
                window = 0.5 * (1 + np.cos(np.pi * XI / np.max(np.abs(XI))))
                sigma_vals *= window * (np.abs(XI) < np.max(np.abs(XI)))
    
            # Spatial window
            if space_window:
                x_center = (x[0] + x[-1]) / 2
                L = (x[-1] - x[0]) / 2
                window = np.exp(-((X - x_center)/L)**2)
                sigma_vals *= window
    
            exp_matrix = np.exp(1j * np.outer(x, xi))  # (Nx, Nxi)
            integrand = sigma_vals * u_hat[np.newaxis, :] * exp_matrix
            result = dxi * np.sum(integrand, axis=1) / (2 * np.pi)
            return result
    
        elif u_vals.ndim == 2:
            # === 2D case ===
            x1, x2 = x_grid
            xi1, xi2 = xi_grid
            dx1 = x1[1] - x1[0]
            dx2 = x2[1] - x2[0]
            dxi1 = xi1[1] - xi1[0]
            dxi2 = xi2[1] - xi2[0]
    
            X1, X2 = np.meshgrid(x1, x2, indexing='ij')
            XI1, XI2 = np.meshgrid(xi1, xi2, indexing='ij')
    
            # Fourier transform of u(x1, x2)
            phase_ft = np.exp(-1j * (np.tensordot(x1, xi1, axes=0)[:, None, :, None] +
                                     np.tensordot(x2, xi2, axes=0)[None, :, None, :]))
            u_hat = np.tensordot(u_vals, phase_ft, axes=([0,1], [0,1])) * dx1 * dx2
    
            # Symbol evaluation
            sigma_vals = symbol_func(X1[:, :, None, None], X2[:, :, None, None],
                                     XI1[None, None, :, :], XI2[None, None, :, :])
    
            # Clamp values
            sigma_vals = np.clip(sigma_vals, -clamp, clamp)
    
            # Frequency window
            if freq_window == 'gaussian':
                sigma_xi1 = 0.8 * np.max(np.abs(XI1))
                sigma_xi2 = 0.8 * np.max(np.abs(XI2))
                window = np.exp(-(XI1[None, None, :, :] / sigma_xi1)**4 -
                                (XI2[None, None, :, :] / sigma_xi2)**4)
                sigma_vals *= window
            elif freq_window == 'hann':
                # Frequency window - Hanning
                wx = 0.5 * (1 + np.cos(np.pi * XI1 / np.max(np.abs(XI1))))
                wy = 0.5 * (1 + np.cos(np.pi * XI2 / np.max(np.abs(XI2))))
                
                # Mask to zero outside max frequency
                mask_x = (np.abs(XI1) < np.max(np.abs(XI1)))
                mask_y = (np.abs(XI2) < np.max(np.abs(XI2)))
                
                # Expand wx and wy to match sigma_vals shape: (64, 64, 64, 64)
                sigma_vals *= wx[:, :, None, None] * wy[:, :, None, None]
                sigma_vals *= mask_x[:, :, None, None] * mask_y[:, :, None, None]
    
            # Spatial window
            if space_window:
                x_center = (x1[0] + x1[-1])/2
                y_center = (x2[0] + x2[-1])/2
                Lx = (x1[-1] - x1[0])/2
                Ly = (x2[-1] - x2[0])/2
                window = np.exp(-((X1 - x_center)/Lx)**2 - ((X2 - y_center)/Ly)**2)
                sigma_vals *= window[:, :, None, None]
    
            # Oscillatory phase
            phase = np.exp(1j * (X1[:, :, None, None] * XI1[None, None, :, :] +
                                 X2[:, :, None, None] * XI2[None, None, :, :]))

            integrand = sigma_vals * u_hat[None, None, :, :] * phase
            result = dxi1 * dxi2 * np.sum(integrand, axis=(2, 3)) / (2 * np.pi)**2
            return result
    
        else:
            raise NotImplementedError("Only 1D and 2D supported")

    def step_ETD_RK4(self, u):
        """
        Perform one Exponential Time Differencing Runge-Kutta of 4th order (ETD-RK4) time step 
        for first-order in time PDEs of the form:
        
            ∂ₜu = L u + N(u)
        
        where L is a linear operator (possibly nonlocal or pseudo-differential), and N is a 
        nonlinear term treated via pseudo-spectral methods. This method evaluates the 
        exponential integrator up to fourth-order accuracy in time.
    
        The ETD-RK4 scheme uses four stages to approximate the integral of the variation-of-constants formula:
        
            uⁿ⁺¹ = e^(L Δt) uⁿ + Δt ∫₀¹ e^(L Δt (1 - τ)) φ(N(u(τ))) dτ
        
        where φ denotes the nonlinear contributions evaluated at intermediate stages.
    
        Args:
            u (np.ndarray): Current solution in real space (physical grid values).
    
        Returns:
            np.ndarray: Updated solution in real space after one ETD-RK4 time step.
    
        Notes:
        - The linear part L is diagonal in Fourier space and precomputed as self.L(k).
        - Nonlinear terms are evaluated in physical space and transformed via FFT.
        - The functions φ₁(z) and φ₂(z) are entire functions arising from the ETD scheme:
          
              φ₁(z) = (eᶻ - 1)/z   if z ≠ 0
                     = 1            if z = 0
    
              φ₂(z) = (eᶻ - 1 - z)/z²   if z ≠ 0
                     = ½              if z = 0
    
        - This implementation assumes periodic boundary conditions and uses spectral differentiation via FFT.
        - See Hochbruck & Ostermann (2010) for theoretical background on exponential integrators.
    
        See Also:
            step_ETD_RK4_order2 : For second-order in time equations.
            psiOp_apply           : For applying pseudo-differential operators.
            apply_nonlinear      : For handling nonlinear terms in the PDE.
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
        Perform one time step of the Exponential Time Differencing Runge-Kutta 4th-order (ETD-RK4) scheme for second-order PDEs.
    
        This method evolves the solution u and its time derivative v forward in time by one step using the ETD-RK4 integrator. 
        It is designed for systems of the form:
        
            ∂ₜ²u = L u + N(u)
            
        where L is a linear operator and N is a nonlinear term computed via self.apply_nonlinear.
        
        The exponential integrator handles the linear part exactly in Fourier space, while the nonlinear terms are integrated 
        using a fourth-order Runge-Kutta-like approach. This ensures high accuracy and stability for stiff systems.
    
        Parameters:
            u (np.ndarray): Current solution array in real space.
            v (np.ndarray): Current time derivative of the solution (∂ₜu) in real space.
    
        Returns:
            tuple: (u_new, v_new), updated solution and its time derivative after one time step.
    
        Notes:
            - Assumes periodic boundary conditions and uses FFT-based spectral methods.
            - Handles both 1D and 2D problems seamlessly.
            - Uses phi functions to compute exponential integrators efficiently.
            - Suitable for wave equations and other second-order evolution equations with stiffness.
        """
        dt = self.dt
    
        L_fft = self.L(self.KX) if self.dim == 1 else self.L(self.KX, self.KY)
        fft = self.fft
        ifft = self.ifft
    
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
    
        # Stage D
        D = rhs(uc)
    
        # Final update
        u_new = u + dt * v + (dt**2 / 6.0) * (A + 2*B + 2*C + D)
        v_new = v + (dt / 6.0) * (A + 2*B + 2*C + D)
    
        return u_new, v_new

    def check_cfl_condition(self):
        """
        Check the CFL (Courant–Friedrichs–Lewymann) condition based on group velocity 
        for second-order time-dependent PDEs.
    
        This method verifies whether the chosen time step dt satisfies the numerical stability 
        condition derived from the maximum wave propagation speed in the system. It supports both 
        1D and 2D problems, with or without a symbolic dispersion relation ω(k).
    
        The CFL condition ensures that information does not propagate further than one grid cell 
        per time step. A safety factor of 0.5 is applied by default to ensure robustness.
    
        Notes:
        - In 1D, the group velocity v₉(k) = dω/dk is used to compute the maximum wave speed.
        - In 2D, the x- and y-directional group velocities are evaluated independently.
        - If no dispersion relation is available, the imaginary part of the linear operator L(k) 
          is used as an approximation for wave speed.
    
        Raises:
        - NotImplementedError: If the spatial dimension is not 1D or 2D.
    
        Prints:
        - Warning message if the current time step dt exceeds the CFL-stable limit.
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
        Check strict analytic conditions on the linear symbol self.L_symbolic:
        
            This method evaluates three key properties of the Fourier multiplier 
            symbol a(k) = self.L(k), which are crucial for well-posedness, stability,
            and numerical efficiency. The checks apply to both 1D and 2D cases.
        
        Conditions checked:
        ------------------
        1. **Stability condition**: Re(a(k)) ≤ 0 for all k ≠ 0
           Ensures that the system does not exhibit exponential growth in time.
    
        2. **Dissipation condition**: Re(a(k)) ≤ -δ |k|² for large |k|
           Ensures sufficient damping at high frequencies to avoid oscillatory instability.
    
        3. **Growth condition**: |a(k)| ≤ C (1 + |k|)^m with m ≤ 4
           Ensures that the symbol does not grow too rapidly with frequency, 
           which would otherwise cause numerical instability or unphysical amplification.
    
        Parameters:
        -----------
        k_range : tuple or None, optional
            Specifies the range of frequencies to test in the form (k_min, k_max, N).
            If None, defaults are used: [-10, 10] with 500 points in 1D, or [-10, 10] 
            with 100 points per axis in 2D.
    
        verbose : bool, default=True
            If True, prints detailed results of each condition check.
    
        Returns:
        --------
        None
            Output is printed directly to the console for interpretability.
    
        Notes:
        ------
        - In 2D, the radial frequency |k| = √(kx² + ky²) is used for comparisons.
        - The dissipation threshold assumes δ = 0.01 and p = 2 by default.
        - The growth ratio is compared against |k|⁴; values above 100 indicate rapid growth.
        - This function is typically called during solver setup or analysis phase.
    
        See Also:
        ---------
        analyze_wave_propagation : For further symbolic and numerical analysis of dispersion.
        plot_symbol : Visualizes the symbol's behavior over the frequency domain.
        """
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
                print("⚠️ Symbol grows rapidly: |a(k)| ≳ |k|^4")
        else:
            if verbose:
                print("✅ Reasonable spectral growth")
    
        if verbose:
            print("✔ Symbol analysis completed.")

    def analyze_wave_propagation(self):
        """
        Perform a detailed analysis of wave propagation characteristics based on the dispersion relation ω(k).
    
        This method visualizes key wave properties in both 1D and 2D settings:
        - Dispersion relation: ω(k)
        - Phase velocity: v_p(k) = ω(k)/|k|
        - Group velocity: v_g(k) = ∇ₖ ω(k)
        - Anisotropy in 2D (via magnitude of group velocity)
    
        The symbolic dispersion relation 'omega_symbolic' must be defined beforehand.
        This is typically available only for second-order-in-time equations.
    
        In 1D:
            Plots ω(k), v_p(k), and v_g(k) over a range of k values.
    
        In 2D:
            Displays heatmaps of ω(kx, ky), v_p(kx, ky), and |v_g(kx, ky)| over a 2D wavenumber grid.
    
        Raises:
            AttributeError: If 'omega_symbolic' is not defined, the method exits gracefully with a message.
    
        Side Effects:
            Generates and displays matplotlib plots.
        """
        print("\n*****************************")
        print("* Wave propagation analysis *")
        print("*****************************\n")
        if not hasattr(self, 'omega_symbolic'):
            print("❌ omega_symbolic not defined. Only available for 2nd order in time.")
            return
    
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
        Visualize the spectral symbol L(k) or L(kx, ky) in 1D or 2D.
    
        This method plots the linear operator's symbolic Fourier representation 
        either as a function of a single wavenumber k (1D), or two wavenumbers 
        kx and ky (2D). The user can choose to display the real part, imaginary part, 
        or absolute value of the symbol.
    
        Parameters:
            component : str {'abs', 're', 'im'}
                Component of the symbol to visualize:
                    - 'abs' : absolute value |a(k)|
                    - 're'  : real part Re[a(k)]
                    - 'im'  : imaginary part Im[a(k)]
            k_range : tuple (kmin, kmax, N), optional
                Wavenumber range for evaluation:
                    - kmin: minimum wavenumber
                    - kmax: maximum wavenumber
                    - N: number of sampling points
                If None, defaults to [-10, 10] with high resolution.
            cmap : str, optional
                Colormap used for 2D surface plots. Default is 'viridis'.
    
        Raises:
            ValueError: If the spatial dimension is not 1D or 2D.
    
        Notes:
            - In 1D, the symbol is plotted using a standard 2D line plot.
            - In 2D, a 3D surface plot is generated with color-mapped height.
            - Symbol evaluation uses self.L(k), which must be defined and callable.
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
        Compute the total energy of the wave equation solution for second-order temporal PDEs. 
        The energy is defined as:
            E(t) = 1/2 ∫ [ (∂ₜu)² + |L¹ᐟ²u|² ] dx
        where L is the linear operator associated with the spatial part of the PDE,
        and L¹ᐟ² denotes its square root in Fourier space.
    
        This method supports both 1D and 2D problems and is only meaningful when 
        self.temporal_order == 2 (second-order time derivative).
    
        Returns:
        - float or None: Total energy at current time step. Returns None if the 
          temporal order is not 2 or if no valid velocity data (v_prev) is available.
    
        Notes:
        - Uses FFT-based spectral differentiation to compute the spatial contributions.
        - Assumes periodic boundary conditions.
        - Handles both real and complex-valued solutions.
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
        Plot the time evolution of the total energy for wave equations. 
        Visualizes the energy computed during simulation for both 1D and 2D cases. 
        Requires temporal_order=2 and prior execution of compute_energy() during solve().
        
        Parameters:
            log : bool
                If True, displays energy on a logarithmic scale to highlight exponential decay/growth.
        
        Notes:
            - Energy is defined as E(t) = 1/2 ∫ [ (∂ₜu)² + |L¹⸍²u|² ] dx
            - Only available if energy monitoring was activated in solve()
            - Automatically skips plotting if no energy data is available
        
        Displays:
            - Time vs. Total Energy plot with grid and legend
            - Appropriate axis labels and dimensional context (1D/2D)
            - Logarithmic or linear scaling based on input parameter
        """
        if not hasattr(self, 'energy_history') or not self.energy_history:
            print("No energy data recorded. Call compute_energy() within solve().")
            return
    
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

    def show_stationary_solution(self, u=None, component='abs', cmap='viridis'):
        """
        Display the stationary solution computed by solve_stationary_psiOp.

        This method visualizes the solution of a pseudo-differential equation 
        solved in stationary mode. It supports both 1D and 2D spatial domains, 
        with options to display different components of the solution (real, 
        imaginary, absolute value, or phase).

        Parameters
        ----------
        u : ndarray, optional
            Precomputed solution array. If None, calls solve_stationary_psiOp() 
            to compute the solution.
        component : str, optional {'real', 'imag', 'abs', 'angle'}
            Component of the complex-valued solution to display:
            - 'real': Real part
            - 'imag': Imaginary part
            - 'abs' : Absolute value (modulus)
            - 'angle' : Phase (argument)
        cmap : str, optional
            Colormap used for 2D visualization (default: 'viridis').

        Raises
        ------
        ValueError
            If an invalid component is specified or if the spatial dimension 
            is not supported (only 1D and 2D are implemented).

        Notes
        -----
        - In 1D, the solution is displayed using a standard line plot.
        - In 2D, the solution is visualized as a 3D surface plot.
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
            plt.title('Stationary solution (2D)')    
            data0 = get_component(u)
            ax.plot_surface(self.X, self.Y, data0, cmap='viridis')
            plt.tight_layout()
            plt.show()
    
        else:
            raise ValueError("Only 1D and 2D display are supported.")

    
    def animate(self, component='abs', overlay='contour'):
        """
        Create an animated plot of the solution evolution over time.

        This method generates a dynamic visualization of the solution array `self.frames`, 
        animating either the real part, imaginary part, absolute value, or complex angle 
        of the field. It supports both 1D line plots and 2D surface plots with optional 
        contour overlays.

        Parameters
        ----------
        component : str in {'real', 'imag', 'abs', 'angle'}
            The component of the solution to visualize:
            - 'real' : Real part Re(u)
            - 'imag' : Imaginary part Im(u)
            - 'abs' : Absolute value |u|
            - 'angle' : Complex argument arg(u)

        overlay : str in {'contour', 'front'}, optional
            Type of overlay for 2D animations:
            - 'contour' : Adds contour lines beneath the surface at each frame.
            - 'front' : Used for tracking wavefronts.

        Returns
        -------
        FuncAnimation
            A Matplotlib `FuncAnimation` object that can be displayed or saved as a video.

        Notes
        -----
        - Uses linear interpolation to map simulation frames to target animation frames.
        - In 2D, the z-axis dynamically rescales based on current data range.
        - For 'angle' component, color scaling is fixed between -π and π for consistency.
        - The animation interval is fixed at 50 ms per frame for smooth playback.
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
        target_times = np.linspace(0, self.Lt, self.n_frames // 2)
        
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
                elif overlay == 'front':
                    dx = self.x_grid[1] - self.x_grid[0]
                    dy = self.y_grid[1] - self.y_grid[0]
                    du_dx, du_dy = np.gradient(current_data, dx, dy)
                    grad_norm = np.sqrt(du_dx**2 + du_dy**2)
                    local_max = (grad_norm == maximum_filter(grad_norm, size=5))
                    normalized = grad_norm[local_max] / np.max(grad_norm)
                    colors = cm.plasma(normalized)
    
                    ax.scatter(self.X[local_max], self.Y[local_max],
                               z_offset * np.ones_like(self.X[local_max]),
                               color=colors, s=10, alpha=0.8)
    
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

        This method quantitatively compares the numerical solution with a provided exact solution 
        at a specified time using either relative or absolute error norms. It supports both 
        stationary and time-dependent problems in 1D and 2D. If enabled, it also generates plots 
        of the solution, exact solution, and pointwise error.

        Parameters
        ----------
        u_exact : callable
            Exact solution function taking spatial coordinates and optionally time as arguments.
        t_eval : float, optional
            Time at which to compare solutions. For non-stationary problems, defaults to final time Lt.
            Ignored for stationary problems.
        norm : str {'relative', 'absolute'}
            Type of error norm used in comparison.
        threshold : float
            Acceptable error threshold; raises an assertion if exceeded.
        plot : bool
            Whether to display visual comparison plots (default: True).
        component : str {'real', 'imag', 'abs'}
            Component of the solution to compare and visualize.

        Raises
        ------
        ValueError
            If unsupported dimension is encountered or requested evaluation time exceeds simulation duration.
        AssertionError
            If computed error exceeds the given threshold.

        Prints
        ------
        - Information about the closest available frame to the requested evaluation time.
        - Computed error value and comparison to threshold.

        Notes
        -----
        - For time-dependent problems, the solution is extracted from precomputed frames.
        - Plots are adapted to spatial dimension: line plots for 1D, image plots for 2D.
        - The method ensures consistent handling of real, imaginary, and magnitude components.
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
