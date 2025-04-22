import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fft, ifft, fftfreq
from sympy import (
    symbols, Function, diff, exp, I, solve, pprint, Mul,
    lambdify, expand, Eq, Derivative, sin, cos, simplify, sqrt,
)
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from functools import partial
plt.rcParams['text.usetex'] = False

class Op(Function):
    """Custom symbolic wrapper for pseudo-differential operators in Fourier space.
    Usage: Op(symbol_expr, u)
    """
    nargs = 2

class PDESolver:
    def __init__(self, equation, boundary_condition='dirichlet', boundary_func=None, interpolation=True, time_scheme='default', dealiasing_ratio=2/3):
        """
        Initialize the PDE solver with a given equation and boundary condition.
        Args:
            equation (sympy.Eq): Partial Differential Equation to solve.
            boundary_condition (str, optional): Type of boundary condition. Defaults to 'dirichlet'.
            interpolation (bool, optional): Whether to use interpolation for boundary conditions. Defaults to True.
        Raises:
            ValueError: If the equation does not contain exactly one unknown function or 
                        if the function does not depend on t, x, y.
        """
        self.interpolation = interpolation
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
        self.temporal_order = 0  # Order of the temporal derivative
        self.linear_terms, self.nonlinear_terms, self.symbol_terms = self.parse_equation(equation)

        # Boundary condition type ('dirichlet', 'periodic', 'neumann' ou 'robin')
        self.boundary_condition = boundary_condition
        self.boundary_func = boundary_func 

        # Initialisation des masques et informations pour domaine curviligne
        self.domain_mask = None
        self.boundary_mask = None
        self.boundary_normals = None
        self.boundary_curvature = None

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
            tuple: Dictionary of linear terms, list of nonlinear terms, and list of extra symbolic terms with coefficients.
        """
        def contains_u_and_derivative(expr, u_func):
            if expr.func == Mul:
                factors = expr.args
                contains_u = any(arg.func == u_func.func for arg in factors)
                contains_derivative = any(isinstance(arg, Derivative) and arg.expr.func == u_func.func for arg in factors)
                return contains_u and contains_derivative
            for arg in expr.args:
                if contains_u_and_derivative(arg, u_func):
                    return True
            return False
    
        print("\n********************")
        print("* Equation parsing *")
        print("********************\n")
    
        # Rewrite the equation in standard form: LHS - RHS = 0
        if isinstance(equation, Eq):
            lhs = equation.lhs - equation.rhs
        else:
            lhs = equation
    
        print(f"\nEquation rewritten in standard form: {lhs}")
    
        # Expand the equation
        lhs_expanded = expand(lhs)
        print(f"\nExpanded equation: {lhs_expanded}")
    
        linear_terms = {}
        nonlinear_terms = []
        symbol_terms = []
    
        # Extract custom Op() symbols from RHS (before any classification)
        for expr in lhs.atoms(Op):
            full_term = [term for term in lhs.as_ordered_terms() if expr in term.args or term == expr]
            if full_term:
                coeff = full_term[0].as_coeff_mul()[0]
                symbol_expr = expr.args[0]
                symbol_terms.append((coeff, symbol_expr))
    
        # Temporal derivative order detection
        self.temporal_order = 0
        for term in lhs_expanded.as_ordered_terms():
            derivs = term.atoms(Derivative)
            for deriv in derivs:
                if deriv.expr == self.u and self.t in deriv.variables:
                    order = deriv.variables.count(self.t)
                    self.temporal_order = max(self.temporal_order, order)
    
        print(f"Temporal derivative order detected: {self.temporal_order}")
    
        # Parse terms, excluding Op(...) from classification
        for term in lhs_expanded.as_ordered_terms():
            print(f"Analyzing term: {term}")
    
            if term.has(Op):
                print("  --> Detected symbolic operator term (Op), excluded from classification.")
                continue
    
            if contains_u_and_derivative(term, self.u) or term.has(self.u**2) or term.has(self.u**3):
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
            else:
                if self.u in term.atoms(Function):
                    coeff = term.as_coefficients_dict().get(self.u, 1)
                    linear_terms[self.u] = linear_terms.get(self.u, 0) + coeff
                    print("  --> Classified as linear")
                else:
                    raise ValueError(f"Unrecognized term: {term}")
    
        print(f"Final linear terms: {linear_terms}")
        print(f"Final nonlinear terms: {nonlinear_terms}")
        print(f"Symbol terms: {symbol_terms}")
    
        return linear_terms, nonlinear_terms, symbol_terms

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
    
        if lhs == 0 and self.symbol_terms:
            print("⚠ Aucune dérivée détectée. Opérateur linéaire purement symbolique (Op(...)).")
            self.L_symbolic = sum(
                coeff * symbol.subs(dict(zip(symbol.free_symbols, self.k_symbols)))
                for coeff, symbol in self.symbol_terms
            )
            self.L = lambdify(self.k_symbols, self.L_symbolic, "numpy")
            print("\n--- Final linear operator ---")
            pprint(self.L_symbolic)
            return
    
        # --- Step 3: dispersion relation ---
        equation = simplify(lhs / plane_wave)
        print("\nRaw characteristic equation:")
        pprint(equation)
    
        dispersion = solve(Eq(equation, 0), omega)
        if not dispersion:
            raise ValueError("No solution found for omega")
        print("Solutions found:")
        pprint(dispersion)
    
        if self.temporal_order == 2:
            omega_expr = simplify(sqrt(dispersion[0]**2))
            self.omega_symbolic = omega_expr
            self.omega = lambdify(self.k_symbols, omega_expr, "numpy")
            self.L_symbolic = -omega_expr**2
        else:
            self.L_symbolic = -I * dispersion[0]
    
        # --- Step 4: add Op(...) terms ---
        for coeff, symbol in self.symbol_terms:
            self.L_symbolic += coeff * symbol.subs(dict(zip(symbol.free_symbols, self.k_symbols)))
    
        self.L = lambdify(self.k_symbols, self.L_symbolic, "numpy")
    
        print("\n--- Final linear operator ---")
        pprint(self.L_symbolic)
        

    def check_cfl_condition(self):
        """
        Check the CFL condition to ensure numerical stability for any PDE.
        Raises:
            ValueError: If the CFL condition is violated.
        """
        # Generic CFL coefficient (safety factor)
        cfl_factor = 0.5  # Typically between 0.1 and 1.0
    
        if self.dim == 1:
            # Maximum propagation speed (related to the linear operator)
            max_speed = np.max(np.abs(self.L(self.kx)))
            dx = self.Lx / self.Nx  # Spatial step
            cfl_limit = cfl_factor * dx / max_speed
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")
        elif self.dim == 2:
            # Maximum propagation speed (related to the 2D linear operator)
            max_speed_x = np.max(np.abs(self.L(self.kx, 0)))  # Contribution in x
            max_speed_y = np.max(np.abs(self.L(0, self.ky)))  # Contribution in y
            dx = self.Lx / self.Nx  # Spatial step in x
            dy = self.Ly / self.Ny  # Spatial step in y
            cfl_limit = cfl_factor / (max_speed_x / dx + max_speed_y / dy)
            if self.dt > cfl_limit:
                print(f"CFL condition violated: dt = {self.dt}, max allowed dt = {cfl_limit}")

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


    def setup(self, Lx, Ly=None, Nx=None, Ny=None, Lt=1.0, Nt=100, initial_condition=None, initial_velocity=None, domain_func=None, boundary_func=None, n_frames=100):
        """
        Set up the computational grid, initial conditions, and (optionally) the curvilinear domain.
        Args:
            Lx (float): Domain length in the x-direction (for the enclosing rectangle).
            Ly (float): Domain length in the y-direction (for the enclosing rectangle).
            Nx (int): Number of grid points in the x-direction.
            Ny (int): Number of grid points in the y-direction.
            Lt (float): Total simulation time.
            Nt (int): Number of time steps.
            initial_condition (callable): Function generating the initial condition for u.
            initial_velocity (callable, optional): Function generating the initial condition for v (if temporal_order == 2).
            domain_func (callable, optional): Characteristic function f(x,y) defining the curvilinear domain by f(x,y)<=0.
                                              If None, the rectangular domain is used.
            n_frames (int): Number of pictures in the animation
        """
        self.Lt, self.Nt = Lt, Nt
        self.dt = Lt / Nt
        self.n_frames = n_frames

        if self.dim == 1:
            if Nx is None:
                raise ValueError("Nx must be specified in 1D.")
        else:
            if None in (Ly, Ny):
                raise ValueError("Both Ly and Ny must be specified in 2D.")
    
        if self.dim == 1:
            self.Lx = Lx
            self.Nx = Nx
    
            # Spatial grid and frequencies
            self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
            self.X = self.x_grid
            self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
            self.KX = self.kx
    
            # Dealiasing mask
            k_max_x = self.dealiasing_ratio * np.max(np.abs(self.kx))
            self.dealiasing_mask = (np.abs(self.KX) <= k_max_x)
    
            # Exponential operator
            L_values = np.array(self.L(self.KX), dtype=np.complex128) 
            self.exp_L = np.exp(L_values * self.dt)
    
            # Initial condition
            self.u_prev = initial_condition(self.X)
    
            # No curvilinear domain in 1D
            self.domain_mask = None
            self.boundary_mask = None
            self.boundary_normals = None
            self.boundary_curvature = None
    
            # Boundary function
            if boundary_func is not None:
                self.boundary_func = boundary_func
    
            # Apply boundary condition
            self.apply_boundary(self.u_prev)
    
            if self.temporal_order == 2:
                if initial_velocity is None:
                    raise ValueError("Initial velocity must be provided for second-order temporal derivatives")
                self.v_prev = initial_velocity(self.X)
                self.apply_boundary(self.v_prev)
            else:
                self.v_prev = None
    
            self.frames = [self.u_prev.copy()]
    
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX)
                self.omega_val = omega_val
                self.cos_omega_dt = np.cos(omega_val * self.dt)
                self.sin_omega_dt = np.sin(omega_val * self.dt)
                self.inv_omega = np.zeros_like(omega_val)
                nonzero = omega_val != 0
                self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]

        elif self.dim == 2:
            self.Lx, self.Ly = Lx, Ly
            self.Nx, self.Ny = Nx, Ny

            self.x_grid = np.linspace(-Lx/2, Lx/2, Nx, endpoint=False)
            self.y_grid = np.linspace(-Ly/2, Ly/2, Ny, endpoint=False)
            self.X, self.Y = np.meshgrid(self.x_grid, self.y_grid, indexing='ij')
    
            self.kx = 2 * np.pi * fftfreq(Nx, d=Lx / Nx)
            self.ky = 2 * np.pi * fftfreq(Ny, d=Ly / Ny)
            self.KX, self.KY = np.meshgrid(self.kx, self.ky, indexing='ij')
    
            k_max_x = self.dealiasing_ratio * np.max(np.abs(self.kx))
            k_max_y = self.dealiasing_ratio * np.max(np.abs(self.ky))
            self.dealiasing_mask = (np.abs(self.KX) <= k_max_x) & (np.abs(self.KY) <= k_max_y)
    
            self.exp_L = np.exp(self.L(self.KX, self.KY) * self.dt)
    
            self.u_prev = initial_condition(self.X, self.Y)
    
            if domain_func is not None:
                dom = domain_func(self.X, self.Y) <= 0
                self.domain_mask = dom.copy()
                self._compute_boundary_info()
            else:
                self.domain_mask = np.ones_like(self.X, dtype=bool)
                self.boundary_mask = None
                self.boundary_normals = None
                self.boundary_curvature = None
    
            if boundary_func is not None:
                self.boundary_func = boundary_func
    
            self.apply_boundary(self.u_prev)
    
            if self.temporal_order == 2:
                if initial_velocity is None:
                    raise ValueError("Initial velocity must be provided for second-order temporal derivatives")
                self.v_prev = initial_velocity(self.X, self.Y)
                self.apply_boundary(self.v_prev)
            else:
                self.v_prev = None
    
            self.frames = [self.u_prev.copy()]
    
            if self.temporal_order == 2:
                omega_val = self.omega(self.KX, self.KY)
                self.omega_val = omega_val
                self.cos_omega_dt = np.cos(omega_val * self.dt)
                self.sin_omega_dt = np.sin(omega_val * self.dt)
                self.inv_omega = np.zeros_like(omega_val)
                nonzero = omega_val != 0
                self.inv_omega[nonzero] = 1.0 / omega_val[nonzero]
    
        else:
            raise NotImplementedError("Only 1D and 2D problems are supported.")

        print("\n*****************")
        print("* CFL condition *")
        print("*****************\n")
        self.check_cfl_condition()

        print("\n********************")
        print("* Symbol condition *")
        print("********************\n")
        self.check_symbol_conditions()

        print("\n*******************")
        print("* Symbol plotting *")
        print("*******************\n")
        self.plot_symbol()

        if self.temporal_order == 2:
            print("\n*****************************")
            print("* Wave propagation analysis *")
            print("*****************************\n")
            self.analyze_wave_propagation()
            
    def apply_boundary(self, u):
        """
        Apply boundary conditions to the solution grid.
        If a curvilinear domain is defined, the boundary condition is applied
        only on the detected boundary. Otherwise, the conditions on the
        rectangle are used.
        Args:
            u (numpy.ndarray): Solution grid to apply boundary conditions.
        Raises:
            ValueError: If an unknown boundary condition is specified.
        """
        # Processing points outside the domain (for all domain types)
        if self.domain_mask is not None:
            # Set all points outside the domain to zero
            u[~self.domain_mask] = 0
        
        # Apply specific boundary conditions
        if self.domain_mask is not None and self.boundary_mask is not None:
            # Curvilinear domain
            self._apply_curvilinear_boundary(u)
        else:
            # Rectangular domain
            self._apply_rectangular_boundary(u)

    def _apply_rectangular_boundary(self, u):
        """Apply boundary conditions for a rectangular domain."""
        if self.dim == 1:
            if self.boundary_condition == 'dirichlet':
                if self.boundary_func is not None:
                    u[0] = self.boundary_func(self.X[0])
                    u[-1] = self.boundary_func(self.X[-1])
                else:
                    u[0] = 0
                    u[-1] = 0
            elif self.boundary_condition == 'periodic':
                u[0] = u[-2]
                u[-1] = u[1]
            elif self.boundary_condition == 'neumann':
                u[0] = u[1]
                u[-1] = u[-2]
            elif self.boundary_condition == 'robin':
                if self.boundary_func is None:
                    raise ValueError("Robin boundary condition requires a boundary_func returning (alpha, beta, g)")
                
                alpha_L, beta_L, g_L = self.boundary_func(np.array([self.X[0]]))
                alpha_R, beta_R, g_R = self.boundary_func(np.array([self.X[-1]]))
                dx = self.x_grid[1] - self.x_grid[0]
            
                # Approximate du/dn ≈ (u[1] - u[0]) / dx
                dudn_L = (u[1] - u[0]) / dx
                u[0] = (g_L - beta_L * dudn_L) / alpha_L
            
                # Approximate du/dn ≈ (u[-1] - u[-2]) / dx
                dudn_R = (u[-1] - u[-2]) / dx
                u[-1] = (g_R - beta_R * dudn_R) / alpha_R
            else:
                raise ValueError(f"Unknown boundary condition: {self.boundary_condition}")
        elif self.dim == 2:
            if self.boundary_condition == 'dirichlet':
                if self.boundary_func is not None:
                    u[0, :] = self.boundary_func(self.X[0, :], self.Y[0, :])
                    u[-1, :] = self.boundary_func(self.X[-1, :], self.Y[-1, :])
                    u[:, 0] = self.boundary_func(self.X[:, 0], self.Y[:, 0])
                    u[:, -1] = self.boundary_func(self.X[:, -1], self.Y[:, -1])
                else:
                    u[0, :] = 0
                    u[-1, :] = 0
                    u[:, 0] = 0
                    u[:, -1] = 0
            elif self.boundary_condition == 'periodic':
                u[0, :] = u[-2, :]
                u[-1, :] = u[1, :]
                u[:, 0] = u[:, -2]
                u[:, -1] = u[:, 1]
            elif self.boundary_condition == 'neumann':
                u[0, :] = u[1, :]
                u[-1, :] = u[-2, :]
                u[:, 0] = u[:, 1]
                u[:, -1] = u[:, -2]
            elif self.boundary_condition == 'robin':
                if self.boundary_func is None:
                    raise ValueError("Robin boundary condition requires a boundary_func(x, y) → (alpha, beta, g)")
            
                dx = self.x_grid[1] - self.x_grid[0]
                dy = self.y_grid[1] - self.y_grid[0]
            
                # Bord gauche (x = min)
                x = self.X[0, :]
                y = self.Y[0, :]
                alpha, beta, g = self.boundary_func(x, y)
                dudx = (u[1, :] - u[0, :]) / dx  # ∂u/∂x ≈ (u1 - u0)/dx
                u[0, :] = (g - beta * dudx) / alpha
            
                # Bord droit (x = max)
                x = self.X[-1, :]
                y = self.Y[-1, :]
                alpha, beta, g = self.boundary_func(x, y)
                dudx = (u[-1, :] - u[-2, :]) / dx
                u[-1, :] = (g - beta * dudx) / alpha
            
                # Bord bas (y = min)
                x = self.X[:, 0]
                y = self.Y[:, 0]
                alpha, beta, g = self.boundary_func(x, y)
                dudy = (u[:, 1] - u[:, 0]) / dy
                u[:, 0] = (g - beta * dudy) / alpha
            
                # Bord haut (y = max)
                x = self.X[:, -1]
                y = self.Y[:, -1]
                alpha, beta, g = self.boundary_func(x, y)
                dudy = (u[:, -1] - u[:, -2]) / dy
                u[:, -1] = (g - beta * dudy) / alpha
            else:
                raise ValueError(f"Unknown boundary condition: {self.boundary_condition}")


    def _apply_curvilinear_boundary(self, u):
        """Apply boundary conditions for a curvilinear domain."""
        if self.boundary_func is not None and self.boundary_condition == 'robin':
            # Robin condition: α u + β ∂u/∂n = g
            alpha, beta, g = self.boundary_func(
                self.X[self.boundary_mask], self.Y[self.boundary_mask]
            )
    
            # Approximate ∂u/∂n using finite differences along normal direction
            nx = self.boundary_normals[..., 0][self.boundary_mask]
            ny = self.boundary_normals[..., 1][self.boundary_mask]
    
            dx = self.x_grid[1] - self.x_grid[0]
            dy = self.y_grid[1] - self.y_grid[0]
    
            # Gradient of u
            du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
            du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dy)
    
            du_dx_vals = du_dx[self.boundary_mask]
            du_dy_vals = du_dy[self.boundary_mask]
    
            dudn = nx * du_dx_vals + ny * du_dy_vals
    
            # Solve α u = g - β ∂u/∂n
            u[self.boundary_mask] = (g - beta * dudn) / alpha
    
            if self.interpolation:
                u[:] = self._smooth_boundary_interpolation(u)
            else:
                u[:] = self._apply_smooth_boundary_condition(u)
            return
    
        # Cas général : sans fonction de bord explicite
        if self.boundary_func is not None:
            u[self.boundary_mask] = self.boundary_func(
                self.X[self.boundary_mask], self.Y[self.boundary_mask]
            )
            if self.interpolation:
                u[:] = self._smooth_boundary_interpolation(u)
            else:
                u[:] = self._apply_smooth_boundary_condition(u)
            return
    
        # Sinon, appliquer les cas standard
        if self.boundary_condition == 'dirichlet':
            u[self.boundary_mask] = 0
        elif self.boundary_condition == 'neumann':
            self._apply_curvilinear_neumann_improved(u)
        elif self.boundary_condition == 'periodic':
            self._apply_curvilinear_periodic(u)
        else:
            raise ValueError(f"Unknown boundary condition: {self.boundary_condition}")

        
    def _smooth_boundary_interpolation(self, u):
        """Apply smooth interpolation at the boundary using a Laplacian smoother."""
        # Create a copy of the current solution
        u_smoothed = u.copy()
        
        # Identify boundary and near-boundary points
        from scipy import ndimage
        
        # Create a dilated boundary (including points just inside and outside)
        boundary_region = ndimage.binary_dilation(self.boundary_mask, iterations=5)
        
        # Points to exclude from smoothing (deep inside or far outside)
        exclude_mask = ~boundary_region
        
        # Apply Laplacian smoothing to the boundary region
        for _ in range(10):  # Number of smoothing iterations
            # Create a convolution kernel for Laplacian smoothing
            kernel = np.array([[0.05, 0.2, 0.05], 
                               [0.2,  0.0, 0.2], 
                               [0.05, 0.2, 0.05]])
            
            # Apply convolution while preserving original values in excluded regions
            u_conv = ndimage.convolve(u_smoothed, kernel, mode='reflect')
            
            # Update only the boundary region
            u_smoothed[boundary_region] = u_conv[boundary_region]
            
            # Ensure boundary conditions are preserved exactly on the actual boundary
            u_smoothed[self.boundary_mask] = u[self.boundary_mask]
            
            # Keep the excluded regions unchanged
            u_smoothed[exclude_mask] = u[exclude_mask]
        
        # Update the main grid
        return u_smoothed
    
    def _apply_curvilinear_neumann(self, u):
        """
        Apply Neumann conditions (zero flux) for a curvilinear domain.
        This method uses a gradient approximation to estimate the boundary normal.
        """
        # Identifying boundary points
        boundary_points = np.argwhere(self.boundary_mask)
        
        # For each boundary point
        for i, j in boundary_points:
            # Create a small neighborhood around the point
            i_min, i_max = max(0, i-1), min(self.Nx-1, i+1)
            j_min, j_max = max(0, j-1), min(self.Ny-1, j+1)
            
            # Identify the interior points in this neighborhood
            local_mask = self.domain_mask[i_min:i_max+1, j_min:j_max+1]
            local_mask[i-i_min, j-j_min] = False  # Exclude the boundary point itself
            
            if np.any(local_mask):
                # Average the values of neighboring interior points
                local_u = u[i_min:i_max+1, j_min:j_max+1]
                interior_values = local_u[local_mask]
                u[i, j] = np.mean(interior_values)
            else:
                # No inner neighbors, use a larger approximation
                u[i, j] = 0  # Default value    

    def _apply_curvilinear_periodic(self, u):
        """
        Apply the periodic conditions for a curvilinear domain.
        Note: This implementation assumes that the domain is approximately
        similar to a rectangle with deformations, making it possible to identify
        correspondences between boundary points.
        """
        # Find the limits of the domain
        x_min, x_max = self.X[self.domain_mask].min(), self.X[self.domain_mask].max()
        y_min, y_max = self.Y[self.domain_mask].min(), self.Y[self.domain_mask].max()
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        
        # Divide the boundary into four regions (per quadrant)
        boundary_points = np.argwhere(self.boundary_mask)
        
        for i, j in boundary_points:
            x, y = self.X[i, j], self.Y[i, j]
            
            # Determine the approximate boundary region
            x_rel = (x - x_center) / (x_max - x_min) * 2  # Normaliser à [-1, 1]
            y_rel = (y - y_center) / (y_max - y_min) * 2  # Normaliser à [-1, 1]
            
            # Find the approximate “opposite” point
            if abs(x_rel) > abs(y_rel):  # Mainly horizontal boundary
                # Search in the opposite x direction
                opposite_x = x_min if x > x_center else x_max
                # Find the nearest boundary point opposite
                candidates = boundary_points[(boundary_points[:, 0] != i) & 
                                            (np.abs(self.Y[boundary_points[:, 0], boundary_points[:, 1]] - y) < 
                                            (y_max - y_min) * 0.1)]
                if len(candidates) > 0:
                    distances = np.abs(self.X[candidates[:, 0], candidates[:, 1]] - opposite_x)
                    closest_idx = np.argmin(distances)
                    i_opposite, j_opposite = candidates[closest_idx]
                    u[i, j] = u[i_opposite, j_opposite]
            else:  # Mainly vertical boundary
                # Search in the opposite y direction
                opposite_y = y_min if y > y_center else y_max
                # Find the nearest boundary point opposite
                candidates = boundary_points[(boundary_points[:, 1] != j) & 
                                            (np.abs(self.X[boundary_points[:, 0], boundary_points[:, 1]] - x) < 
                                            (x_max - x_min) * 0.1)]
                if len(candidates) > 0:
                    distances = np.abs(self.Y[candidates[:, 0], candidates[:, 1]] - opposite_y)
                    closest_idx = np.argmin(distances)
                    i_opposite, j_opposite = candidates[closest_idx]
                    u[i, j] = u[i_opposite, j_opposite]

    def _compute_boundary_info(self):
        """
        Calculate detailed information about the curvilinear domain boundary.
        This method improves boundary detection and calculates local normals.
        """
        if self.domain_mask is None:
            return
            
        # Identify related domain and complementary components
        from scipy import ndimage
        labeled_domain, num_domain = ndimage.label(self.domain_mask)
        labeled_exterior, num_exterior = ndimage.label(~self.domain_mask)
        
        # The boundary is the interface between these regions
        boundary = np.zeros_like(self.domain_mask, dtype=bool)
        
        # For each domain component
        for i in range(1, num_domain + 1):
            domain_comp = (labeled_domain == i)
            # Expand the domain and find the intersection with the exterior
            dilated = ndimage.binary_dilation(domain_comp)
            boundary |= (dilated & ~domain_comp)
        
        self.boundary_mask = boundary
    
        # Calculate signed distance field (SDF)
        # Positive inside the domain, negative outside
        distance_inside = ndimage.distance_transform_edt(self.domain_mask)
        distance_outside = ndimage.distance_transform_edt(~self.domain_mask)
        self.signed_distance = distance_inside - distance_outside
        
        # Calculate boundary normals (approximate gradient)
        self.boundary_normals = np.zeros((self.Nx, self.Ny, 2))
        
        # Use a Sobel filter to calculate the mask gradient
        dx = ndimage.sobel(self.domain_mask.astype(float), axis=0)
        dy = ndimage.sobel(self.domain_mask.astype(float), axis=1)
        
        # Normalize the gradient to obtain the unit normal
        norm = np.sqrt(dx**2 + dy**2)
        mask = (norm > 0) & self.boundary_mask
        self.boundary_normals[mask, 0] = dx[mask] / norm[mask]
        self.boundary_normals[mask, 1] = dy[mask] / norm[mask]
        
        # Calculate local curvature (may be useful for certain applications)
        self.boundary_curvature = np.zeros((self.Nx, self.Ny))
        ddx = ndimage.sobel(dx / np.maximum(norm, 1e-10), axis=0)
        ddy = ndimage.sobel(dy / np.maximum(norm, 1e-10), axis=1)
        self.boundary_curvature[mask] = (ddx[mask] + ddy[mask])

    def _apply_smooth_boundary_condition(self, u):
        """
        Apply boundary conditions with smooth transition based on signed distance.
        """
        # Define the width of the transition region
        transition_width = 3.0  # in grid units
        
        # Calculate weights based on the signed distance field
        # 1.0 at the boundary, smoothly decreasing to 0.0 within the transition width
        weights = np.clip(1.0 - np.abs(self.signed_distance) / transition_width, 0.0, 1.0)
        
        # Apply boundary function where weights > 0
        weight_mask = weights > 0
        if np.any(weight_mask):
            boundary_values = self.boundary_func(self.X[weight_mask], self.Y[weight_mask])
            
            # Apply weighted combination of boundary values and current values
            u[weight_mask] = weights[weight_mask] * boundary_values + \
                            (1.0 - weights[weight_mask]) * u[weight_mask]
        
        # Ensure exact boundary conditions at the actual boundary
        u[self.boundary_mask] = self.boundary_func(
            self.X[self.boundary_mask], self.Y[self.boundary_mask]
        )
        
        return u 

    def _apply_curvilinear_neumann_improved(self, u):
        """
        Apply Neumann's conditions for a curvilinear domain 
        using calculated normals with reflection for zero-flux.
        """
        # Identifying boundary points
        boundary_points = np.argwhere(self.boundary_mask)
    
        for i, j in boundary_points:
            nx, ny = self.boundary_normals[i, j]
            # Opposite direction (into the domain)
            di = int(np.round(-nx))
            dj = int(np.round(-ny))
    
            ni = i + di
            nj = j + dj
    
            # Ensure neighbor is inside the domain
            if (0 <= ni < self.Nx) and (0 <= nj < self.Ny) and self.domain_mask[ni, nj]:
                # Reflect value to the boundary point
                u[i, j] = u[ni, nj]
            else:
                # Fallback: use average of all neighbors inside domain
                i_min = max(0, i - 1)
                i_max = min(self.Nx, i + 2)
                j_min = max(0, j - 1)
                j_max = min(self.Ny, j + 2)
                neighborhood = u[i_min:i_max, j_min:j_max]
                mask = self.domain_mask[i_min:i_max, j_min:j_max]
                if np.any(mask):
                    u[i, j] = np.mean(neighborhood[mask])

        
    def apply_nonlinear(self, u, is_v=False):
        """
        Apply nonlinear terms to the solution with dealiasing.
        Args:
            u (numpy.ndarray): Current solution grid.
            is_v (bool): Whether to compute nonlinear terms for v.
        Returns:
            numpy.ndarray: Contribution from nonlinear terms.
        """
        if not self.nonlinear_terms:
            return np.zeros_like(u, dtype=np.complex128)  # Initialize as complex
    
        nonlinear_term = np.zeros_like(u, dtype=np.complex128)  # Initialize as complex
        if self.dim == 1:
            dx = self.x_grid[1] - self.x_grid[0]
            u_x = (np.roll(u, -1) - np.roll(u, 1)) / (2 * dx)
            # Transform to spectral space
            u_hat = self.fft(u)
            u_x_hat = self.fft(u_x)
            # Dealiasing
            u_hat *= self.dealiasing_mask
            u_x_hat *= self.dealiasing_mask
            # Back to physical space
            u = self.ifft(u_hat)
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
            dx = self.x_grid[1] - self.x_grid[0]
            dy = self.y_grid[1] - self.y_grid[0]
            u_x = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)
            u_y = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dy)
            # Transform to spectral space
            u_hat = self.fft(u)
            u_x_hat = self.fft(u_x)
            u_y_hat = self.fft(u_y)
            # Dealiasing
            u_hat *= self.dealiasing_mask
            u_x_hat *= self.dealiasing_mask
            u_y_hat *= self.dealiasing_mask
            # Back to physical space
            u = self.ifft(u_hat)
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
    

    def solve(self):
        """
        Solve the PDE with the chosen time integration scheme.
        """
        print("\n*******************")
        print("* Solving the PDE *")
        print("*******************\n")
        
        save_interval = max(1, self.Nt // self.n_frames)
    
        for step in range(self.Nt):
            # 🚫 Mask out-of-domain points before update
            if self.domain_mask is not None:
                self.u_prev[~self.domain_mask] = 0
                if self.temporal_order == 2 and self.v_prev is not None:
                    self.v_prev[~self.domain_mask] = 0
    
            if self.temporal_order == 1:
                if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                    u_new = self.step_ETD_RK4(self.u_prev)
                else:
                    if self.dim == 1:
                        u_hat = self.fft(self.u_prev)
                        u_hat *= self.exp_L
                        u_hat *= self.dealiasing_mask
                        u_lin = self.ifft(u_hat)
                    else:
                        u_hat = self.fft(self.u_prev)
                        u_hat *= self.exp_L
                        u_hat *= self.dealiasing_mask
                        u_lin = self.ifft(u_hat)
    
                    u_nl = self.apply_nonlinear(u_lin)
                    u_new = u_lin + u_nl
    
                # 🚫 Remasquer après calcul
                if self.domain_mask is not None:
                    u_new[~self.domain_mask] = 0
    
                # ✅ Appliquer conditions aux bords
                self.apply_boundary(u_new)
                self.u_prev = u_new
    
            elif self.temporal_order == 2:
                if hasattr(self, 'time_scheme') and self.time_scheme == 'ETD-RK4':
                    u_new, v_new = self.step_ETD_RK4_order2(self.u_prev, self.v_prev)
                else:
                    if self.dim == 1:
                        u_hat = self.fft(self.u_prev)
                        v_hat = self.fft(self.v_prev)
    
                        u_new_hat = (self.cos_omega_dt * u_hat +
                                     self.sin_omega_dt * self.inv_omega * v_hat)
    
                        v_new_hat = (-self.omega_val * self.sin_omega_dt * u_hat +
                                      self.cos_omega_dt * v_hat)
    
                        u_new = self.ifft(u_new_hat)
                        v_new = self.ifft(v_new_hat)
    
                    elif self.dim == 2:
                        u_hat = self.fft(self.u_prev)
                        v_hat = self.fft(self.v_prev)
    
                        u_new_hat = (self.cos_omega_dt * u_hat +
                                     self.sin_omega_dt * self.inv_omega * v_hat)
    
                        v_new_hat = (-self.omega_val * self.sin_omega_dt * u_hat +
                                      self.cos_omega_dt * v_hat)
    
                        u_new = self.ifft(u_new_hat)
                        v_new = self.ifft(v_new_hat)
    
                    else:
                        raise NotImplementedError("Unsupported spatial dimension")
    
                # 🚫 Masquer les points hors domaine
                if self.domain_mask is not None:
                    u_new[~self.domain_mask] = 0
                    v_new[~self.domain_mask] = 0
    
                self.apply_boundary(u_new)
                self.apply_boundary(v_new)
                self.u_prev = u_new
                self.v_prev = v_new
            
            # ⏱️ Enregistrement des résultats
            if step % save_interval == 0:
                self.frames.append(self.u_prev.copy())

    
    def step_ETD_RK4(self, u):
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
        Perform one ETD-RK4 step for second-order time PDEs.
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

    
    def animate(self, component='abs', overlay='contour'):
        """
        Create an animated plot of the solution evolution.
        In 1D: line plot.
        In 2D: surface plot with optional overlay.
    
        Args:
            component (str): 'real', 'imag', 'abs', or 'angle'.
            overlay (str): Only used in 2D: 'contour' or 'front'.
        """
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation
        import matplotlib.cm as cm
        from scipy.ndimage import maximum_filter
        import numpy as np
    
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
    
        if self.dim == 1:
            fig, ax = plt.subplots()
            line, = ax.plot(self.X, get_component(self.frames[0]))
            ax.set_ylim(np.min(self.frames[0]), np.max(self.frames[0]))
            ax.set_xlabel('x')
            ax.set_ylabel(f'{component} of u')
            ax.set_title('Initial condition')
            plt.tight_layout()
            plt.show()
            
            def update(i):
                ydata = get_component(self.frames[i])
                line.set_ydata(ydata)
                ax.set_ylim(np.min(ydata), np.max(ydata))
                ax.set_title(f't = {i * self.dt:.2f}')
                return line,
    
            ani = FuncAnimation(fig, update, frames=len(self.frames), interval=50)
            return ani
        
        else:
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
    
            frame_indices = np.linspace(0, len(self.frames) - 1, min(self.n_frames, len(self.frames)), dtype=int)
    
            def update(frame_index):
                frame = frame_indices[frame_index]
                ax.clear()
                current_data = get_component(self.frames[frame])
                z_offset = np.max(current_data) + 0.05 * (np.max(current_data) - np.min(current_data))
    
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
                ax.set_title(f'Solution at t = {frame * self.dt:.2f}')
                return surf
    
            ani = FuncAnimation(fig, update, frames=len(frame_indices), interval=50)
            return ani
    

    def _apply_rectangular_boundary(self, u):
        """Dispatch to appropriate rectangular boundary condition method."""
        handler = self._get_boundary_handler()
        handler(u)

    def _get_boundary_handler(self):
        """Return the boundary condition handler based on dimension and type."""
        if self.dim == 1:
            return getattr(self, f"_apply_{self.boundary_condition}_1D", self._unknown_boundary)
        elif self.dim == 2:
            return getattr(self, f"_apply_{self.boundary_condition}_2D", self._unknown_boundary)
        else:
            raise ValueError("Only 1D and 2D supported.")

    def _unknown_boundary(self, u):
        raise ValueError(f"Unknown boundary condition: {self.boundary_condition}")

    def _apply_dirichlet_1D(self, u):
        u[0] = self.boundary_func(self.X[0]) if self.boundary_func else 0
        u[-1] = self.boundary_func(self.X[-1]) if self.boundary_func else 0

    def _apply_dirichlet_2D(self, u):
        if self.boundary_func:
            u[0, :]  = self.boundary_func(self.X[0, :], self.Y[0, :])
            u[-1, :] = self.boundary_func(self.X[-1, :], self.Y[-1, :])
            u[:, 0]  = self.boundary_func(self.X[:, 0], self.Y[:, 0])
            u[:, -1] = self.boundary_func(self.X[:, -1], self.Y[:, -1])
        else:
            u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0

    def _apply_periodic_1D(self, u):
        u[0] = u[-2]
        u[-1] = u[1]

    def _apply_periodic_2D(self, u):
        u[0, :] = u[-2, :]
        u[-1, :] = u[1, :]
        u[:, 0] = u[:, -2]
        u[:, -1] = u[:, 1]

    def _apply_neumann_1D(self, u):
        u[0] = u[1]
        u[-1] = u[-2]

    def _apply_neumann_2D(self, u):
        u[0, :] = u[1, :]
        u[-1, :] = u[-2, :]
        u[:, 0] = u[:, 1]
        u[:, -1] = u[:, -2]

    def _apply_robin_1D(self, u):
        if self.boundary_func is None:
            raise ValueError("Robin boundary condition requires a boundary_func returning (alpha, beta, g)")
        dx = self.x_grid[1] - self.x_grid[0]

        alpha_L, beta_L, g_L = self.boundary_func(np.array([self.X[0]]))
        alpha_R, beta_R, g_R = self.boundary_func(np.array([self.X[-1]]))

        dudn_L = (u[1] - u[0]) / dx
        u[0] = (g_L - beta_L * dudn_L) / alpha_L

        dudn_R = (u[-1] - u[-2]) / dx
        u[-1] = (g_R - beta_R * dudn_R) / alpha_R

    def _apply_robin_2D(self, u):
        if self.boundary_func is None:
            raise ValueError("Robin boundary condition requires a boundary_func(x, y) → (alpha, beta, g)")

        dx = self.x_grid[1] - self.x_grid[0]
        dy = self.y_grid[1] - self.y_grid[0]

        x, y = self.X[0, :], self.Y[0, :]
        alpha, beta, g = self.boundary_func(x, y)
        dudx = (u[1, :] - u[0, :]) / dx
        u[0, :] = (g - beta * dudx) / alpha

        x, y = self.X[-1, :], self.Y[-1, :]
        alpha, beta, g = self.boundary_func(x, y)
        dudx = (u[-1, :] - u[-2, :]) / dx
        u[-1, :] = (g - beta * dudx) / alpha

        x, y = self.X[:, 0], self.Y[:, 0]
        alpha, beta, g = self.boundary_func(x, y)
        dudy = (u[:, 1] - u[:, 0]) / dy
        u[:, 0] = (g - beta * dudy) / alpha

        x, y = self.X[:, -1], self.Y[:, -1]
        alpha, beta, g = self.boundary_func(x, y)
        dudy = (u[:, -1] - u[:, -2]) / dy
        u[:, -1] = (g - beta * dudy) / alpha

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
        u_num = self.frames[-1]
    
        if self.dim == 1:
            u_ex = u_exact(self.X, t_eval)
        elif self.dim == 2:
            u_ex = u_exact(self.X, self.Y, t_eval)
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
    
        print(f"Test error = {error:.3e}")
        assert error < threshold, f"Error too large: {error:.3e}"
    
        # Optional plots
        if plot:
            if self.dim == 1:
                plt.figure(figsize=(12, 6))
                plt.subplot(2, 1, 1)
                plt.plot(self.X, np.real(u_num), label='Numerical')
                plt.plot(self.X, np.real(u_ex), label='Exact', linestyle='--')
                plt.legend()
                plt.title(f'Solution at t = {t_eval}, error = {error:.2e}')
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
