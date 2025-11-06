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
Geometric Visualization of Pseudodifferential Operator Symbols

This module provides a comprehensive toolkit for visualizing and analyzing
the geometric structure of symbols H(x, ξ) and their connection to spectral
properties through semiclassical analysis.

Key concepts:
- Hamiltonian surface and flow
- Geodesics (classical trajectories)
- Caustics (focal points)
- Periodic orbits and quantization
- Spectral analysis via trace formula
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from numpy.fft import fft, fftfreq
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Geodesic:
    """Represents a geodesic (classical trajectory) in phase space"""
    t: np.ndarray           # Time array
    x: np.ndarray           # Position trajectory
    xi: np.ndarray          # Momentum trajectory
    H: np.ndarray           # Energy along trajectory
    J: np.ndarray           # Jacobian ∂x/∂ξ₀ (focusing measure)
    K: np.ndarray           # ∂ξ/∂ξ₀
    
    @property
    def energy(self) -> float:
        """Initial energy"""
        return self.H[0]
    
    @property
    def caustics(self) -> np.ndarray:
        """Indices where caustics occur (J ≈ 0)"""
        return np.where(np.abs(self.J) < 0.01)[0]


@dataclass
class PeriodicOrbit:
    """Represents a periodic orbit in phase space"""
    x0: float               # Initial position
    xi0: float              # Initial momentum
    period: float           # Period T
    action: float           # Classical action S = ∮ p dq
    energy: float           # Energy E
    stability: float        # Lyapunov exponent (stability measure)
    x_cycle: np.ndarray     # Position along one period
    xi_cycle: np.ndarray    # Momentum along one period
    t_cycle: np.ndarray     # Time array


@dataclass
class Spectrum:
    """Semiclassical spectrum information"""
    energies: np.ndarray    # Energy values
    intensity: np.ndarray   # Spectral intensity
    trace_t: np.ndarray     # Time array for trace
    trace: np.ndarray       # Tr[exp(-iHt/ℏ)]


# ============================================================================
# CORE: SYMBOL GEOMETRY ENGINE
# ============================================================================
def _sanitize(expr):
    """Remove DiracDelta, Heaviside, and undefined sign terms for numeric use."""
    expr = expr.replace(sp.DiracDelta, lambda *args: 0)
    expr = expr.replace(sp.Heaviside, lambda *args: 1)
    expr = sp.simplify(expr)
    return expr
    
class SymbolGeometry:
    """
    Analyzes the geometric structure of a symbol H(x, ξ)
    
    This class computes:
    - Hamiltonian flow (geodesics)
    - Jacobian (focusing)
    - Caustics (singularities)
    - Periodic orbits
    - Semiclassical spectrum
    """
    
    def __init__(self, symbol: sp.Expr, x_sym: sp.Symbol, xi_sym: sp.Symbol):
        """
        Initialize with a symbolic Hamiltonian
        
        Parameters
        ----------
        symbol : sympy expression
            The Hamiltonian H(x, ξ)
        x_sym, xi_sym : sympy symbols
            Position and momentum variables
        """
        self.H = symbol
        self.x_sym = x_sym
        self.xi_sym = xi_sym
        
        # Compute derivatives symbolically (DRY principle)
        self._compute_derivatives()
        
        # Convert to numerical functions (cached)
        self._lambdify_functions()
    
    def _compute_derivatives(self):
        """Compute all necessary derivatives (DRY)"""
        dH_x = sp.diff(self.H, self.x_sym)
        self.dH_dx = _sanitize(dH_x)
        dH_xi = sp.diff(self.H, self.xi_sym)
        self.dH_dxi = _sanitize(dH_xi)
        d2H_x2 = sp.diff(self.dH_dx, self.x_sym)
        self.d2H_dx2 = _sanitize(d2H_x2)        
        d2H_xi2 = sp.diff(self.dH_dxi, self.xi_sym)
        self.d2H_dxi2 = _sanitize(d2H_xi2)        
        d2H_xxi = sp.diff(self.dH_dx, self.xi_sym)
        self.d2H_dxdxi = _sanitize(d2H_xxi)
    
    def _lambdify_functions(self):
        """Convert symbolic expressions to numerical functions (DRY)"""
        vars_tuple = (self.x_sym, self.xi_sym)
        
        self.f_H = sp.lambdify(vars_tuple, self.H, 'numpy')
        self.f_dH_dx = sp.lambdify(vars_tuple, self.dH_dx, 'numpy')
        self.f_dH_dxi = sp.lambdify(vars_tuple, self.dH_dxi, 'numpy')
        self.f_d2H_dx2 = sp.lambdify(vars_tuple, self.d2H_dx2, 'numpy')
        self.f_d2H_dxi2 = sp.lambdify(vars_tuple, self.d2H_dxi2, 'numpy')
        self.f_d2H_dxdxi = sp.lambdify(vars_tuple, self.d2H_dxdxi, 'numpy')
    
    def compute_geodesic(self, x0: float, xi0: float, t_max: float, 
                        n_points: int = 500) -> Geodesic:
        """
        Compute geodesic with Jacobian (for caustics detection)
        
        Solves the augmented system:
        dx/dt = ∂H/∂ξ
        dξ/dt = -∂H/∂x
        dJ/dt = ∂²H/∂ξ² J + ∂²H/∂x∂ξ K  (variational equation)
        dK/dt = -∂²H/∂x∂ξ J - ∂²H/∂x² K
        
        Parameters
        ----------
        x0, xi0 : float
            Initial conditions
        t_max : float
            Final time
        n_points : int
            Number of points
            
        Returns
        -------
        Geodesic
            Complete geodesic information
        """
        def system(t, z):
            x, xi, J, K = z
            try:
                # Hamilton equations
                dx = float(self.f_dH_dxi(x, xi))
                dxi = float(-self.f_dH_dx(x, xi))
                
                # Variational equations (Jacobian evolution)
                d2H_dxi2 = float(self.f_d2H_dxi2(x, xi))
                d2H_dxdxi = float(self.f_d2H_dxdxi(x, xi))
                d2H_dx2 = float(self.f_d2H_dx2(x, xi))
                
                dJ = d2H_dxi2 * J + d2H_dxdxi * K
                dK = -d2H_dxdxi * J - d2H_dx2 * K
                
                return [dx, dxi, dJ, dK]
            except:
                return [0, 0, 0, 0]
        
        # Initial conditions: J(0)=0, K(0)=1 (standard initial condition)
        z0 = [x0, xi0, 0.0, 1.0]
        
        sol = solve_ivp(
            system, [0, t_max], z0,
            t_eval=np.linspace(0, t_max, n_points),
            method='DOP853',
            rtol=1e-10, atol=1e-12
        )
        
        # Compute energy along trajectory
        H_traj = np.array([self.f_H(sol.y[0][i], sol.y[1][i]) 
                          for i in range(len(sol.t))])
        
        return Geodesic(
            t=sol.t,
            x=sol.y[0],
            xi=sol.y[1],
            H=H_traj,
            J=sol.y[2],
            K=sol.y[3]
        )
    
    def find_periodic_orbits(self, energy: float, 
                            x_range: Tuple[float, float],
                            xi_range: Tuple[float, float],
                            n_attempts: int = 50,
                            tol_period: float = 1e-3) -> List[PeriodicOrbit]:
        """
        Find periodic orbits at fixed energy
        
        Strategy: Sample energy surface H(x,ξ)=E and look for closed orbits
        
        Parameters
        ----------
        energy : float
            Target energy level
        x_range, xi_range : tuple
            Search domain
        n_attempts : int
            Number of initial conditions to try
        tol_period : float
            Tolerance for periodicity detection
            
        Returns
        -------
        list of PeriodicOrbit
            Found periodic orbits
        """
        orbits = []
        x_samples = np.linspace(x_range[0], x_range[1], int(np.sqrt(n_attempts)))
        
        for x0_test in x_samples:
            # Solve H(x0, ξ0) = E for ξ0
            def energy_eq(xi0):
                try:
                    return self.f_H(x0_test, xi0) - energy
                except:
                    return 1e10
            
            xi_guesses = np.linspace(xi_range[0], xi_range[1], 5)
            
            for xi_guess in xi_guesses:
                try:
                    result = fsolve(energy_eq, xi_guess, full_output=True)
                    
                    if result[2] != 1:  # Check convergence
                        continue
                    
                    xi0 = result[0][0]
                    
                    # Verify we're on energy surface
                    if abs(self.f_H(x0_test, xi0) - energy) > 1e-6:
                        continue
                    
                    # Integrate to detect periodicity
                    T_max = 20
                    geo = self.compute_geodesic(x0_test, xi0, T_max, 2000)
                    
                    # Find returns to initial point
                    distances = np.sqrt((geo.x - x0_test)**2 + (geo.xi - xi0)**2)
                    
                    # Find local minima (except t=0)
                    minima_idx = []
                    for i in range(10, len(distances)-10):
                        if (distances[i] < distances[i-1] and 
                            distances[i] < distances[i+1] and
                            distances[i] < tol_period):
                            minima_idx.append(i)
                    
                    if minima_idx:
                        idx_period = minima_idx[0]
                        period = geo.t[idx_period]
                        
                        if period > 0.1 and distances[idx_period] < tol_period:
                            # Compute action S = ∮ ξ dx
                            x_cycle = geo.x[:idx_period+1]
                            xi_cycle = geo.xi[:idx_period+1]
                            t_cycle = geo.t[:idx_period+1]
                            
                            dx_dt = np.gradient(x_cycle, t_cycle)
                            action = np.trapz(xi_cycle * dx_dt, t_cycle)
                            
                            # Compute stability (Lyapunov exponent)
                            stability = self._compute_stability(x0_test, xi0, period)
                            
                            orbits.append(PeriodicOrbit(
                                x0=x0_test,
                                xi0=xi0,
                                period=period,
                                action=action,
                                energy=energy,
                                stability=stability,
                                x_cycle=x_cycle,
                                xi_cycle=xi_cycle,
                                t_cycle=t_cycle
                            ))
                
                except:
                    continue
        
        # Remove duplicates
        return self._remove_duplicate_orbits(orbits)
    
    def _compute_stability(self, x0: float, xi0: float, T: float) -> float:
        """Compute Lyapunov exponent (orbit stability)"""
        def linearized_system(t, z):
            x, xi, dx, dxi = z
            try:
                vx = float(self.f_dH_dxi(x, xi))
                vxi = float(-self.f_dH_dx(x, xi))
                
                # Linearization
                A12 = float(self.f_d2H_dxi2(x, xi))
                A21 = float(-self.f_d2H_dxdxi(x, xi))
                
                ddx = A12 * dxi
                ddxi = A21 * dx
                
                return [vx, vxi, ddx, ddxi]
            except:
                return [0, 0, 0, 0]
        
        epsilon = 1e-6
        z0 = [x0, xi0, epsilon, 0]
        
        sol = solve_ivp(linearized_system, [0, T], z0, method='DOP853', rtol=1e-10)
        
        if sol.success and len(sol.y[2]) > 0:
            perturbation_final = np.sqrt(sol.y[2][-1]**2 + sol.y[3][-1]**2)
            return np.log(perturbation_final / epsilon) / T
        else:
            return np.nan
    
    def _remove_duplicate_orbits(self, orbits: List[PeriodicOrbit]) -> List[PeriodicOrbit]:
        """Remove duplicate periodic orbits"""
        unique = []
        for orb in orbits:
            is_duplicate = False
            for orb_unique in unique:
                if (abs(orb.period - orb_unique.period) < 0.1 and
                    abs(orb.action - orb_unique.action) < 0.1):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique.append(orb)
        return unique
    
    def gutzwiller_trace_formula(self, periodic_orbits: List[PeriodicOrbit],
                                 t_values: np.ndarray, hbar: float = 1.0) -> np.ndarray:
        """
        Gutzwiller trace formula (semiclassical)
        
        Tr[exp(-iHt/ℏ)] ≈ Σ_γ A_γ exp(iS_γ/ℏ - iπμ_γ/2)
        
        Parameters
        ----------
        periodic_orbits : list
            List of periodic orbits
        t_values : array
            Time values
        hbar : float
            Reduced Planck constant
            
        Returns
        -------
        array
            Trace as function of time
        """
        trace = np.zeros(len(t_values), dtype=complex)
        
        for orb in periodic_orbits:
            T = orb.period
            S = orb.action
            lambda_stab = orb.stability
            
            # Include repetitions of the orbit
            for k in range(1, 5):
                T_k = k * T
                S_k = k * S
                
                # Stability factor
                if not np.isnan(lambda_stab):
                    det_factor = abs(2 * np.sinh(k * lambda_stab * T))
                else:
                    det_factor = 1
                
                if det_factor > 1e-10:
                    amplitude = T_k / np.sqrt(det_factor)
                    
                    # Maslov index (simplified: 0)
                    mu = 0
                    
                    phase = S_k / hbar - np.pi * mu / 2
                    contribution = amplitude * np.exp(1j * phase) * np.sinc((t_values - T_k) / T_k)
                    trace += contribution
        
        return trace
    
    def semiclassical_spectrum(self, periodic_orbits: List[PeriodicOrbit],
                              hbar: float = 1.0, resolution: int = 1000) -> Spectrum:
        """
        Extract semiclassical spectrum via Fourier transform of trace
        
        Parameters
        ----------
        periodic_orbits : list
            Periodic orbits
        hbar : float
            Reduced Planck constant
        resolution : int
            Number of points
            
        Returns
        -------
        Spectrum
            Spectral information
        """
        t_max = 50 / hbar
        t_values = np.linspace(0, t_max, resolution)
        
        trace = self.gutzwiller_trace_formula(periodic_orbits, t_values, hbar)
        
        # Fourier transform: t → E
        energies_fft = fftfreq(len(t_values), d=t_values[1]-t_values[0]) * 2 * np.pi * hbar
        spectrum_fft = fft(trace)
        
        return Spectrum(
            energies=energies_fft,
            intensity=np.abs(spectrum_fft),
            trace_t=t_values,
            trace=trace
        )


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class SymbolVisualizer:
    """
    Comprehensive visualization of symbol geometry
    
    Produces 15 panels showing:
    1. Hamiltonian surface (3D)
    2. Energy level sets (phase space foliation)
    3. Hamiltonian vector field
    4. Group velocity ∂H/∂ξ
    5. Spatial projection (caustics)
    6. Jacobian (focusing measure)
    7. Curvature (focusing tendency)
    8. Energy conservation
    9. Periodic orbits (phase space)
    10. Period-energy diagram
    11. EBK quantization
    12. Trace formula
    13. Semiclassical spectrum
    14. Orbit stability
    15. Level spacing distribution
    """
    
    def __init__(self, geometry: SymbolGeometry):
        """
        Parameters
        ----------
        geometry : SymbolGeometry
            Initialized geometry engine
        """
        self.geo = geometry
    
    def visualize_complete(self, 
                          x_range: Tuple[float, float],
                          xi_range: Tuple[float, float],
                          geodesics_params: List[Tuple],
                          E_range: Optional[Tuple[float, float]] = None,
                          hbar: float = 1.0,
                          resolution: int = 100) -> Tuple:
        """
        Create complete geometric atlas
        
        Parameters
        ----------
        x_range, xi_range : tuple
            Domain limits
        geodesics_params : list of tuples
            Each tuple: (x0, xi0, t_max, color)
        E_range : tuple, optional
            Energy range for spectral analysis
        hbar : float
            Reduced Planck constant
        resolution : int
            Grid resolution
            
        Returns
        -------
        fig, geodesics, periodic_orbits, spectrum
        """
        # Compute grid
        x_grid = np.linspace(x_range[0], x_range[1], resolution)
        xi_grid = np.linspace(xi_range[0], xi_range[1], resolution)
        X, Xi = np.meshgrid(x_grid, xi_grid)
        
        # Evaluate Hamiltonian and derivatives on grid
        grids = self._evaluate_grids(X, Xi)
        
        # Compute geodesics
        geodesics = self._compute_geodesics(geodesics_params)
        
        # Find periodic orbits (if E_range specified)
        periodic_orbits = []
        spectrum = None
        if E_range:
            energies = np.linspace(E_range[0], E_range[1], 8)
            for E in energies:
                orbits = self.geo.find_periodic_orbits(E, x_range, xi_range)
                periodic_orbits.extend(orbits)
            
            if periodic_orbits:
                spectrum = self.geo.semiclassical_spectrum(periodic_orbits, hbar)
        
        # Create figure
        fig = self._create_figure(X, Xi, grids, geodesics, periodic_orbits, spectrum, hbar)
        
        return fig, geodesics, periodic_orbits, spectrum
    
    def _evaluate_grids(self, X: np.ndarray, Xi: np.ndarray) -> Dict:
        """Evaluate all necessary fields on grid (DRY)"""
        grids = {}
        
        for name, func in [
            ('H', self.geo.f_H),
            ('dH_dxi', self.geo.f_dH_dxi),
            ('dH_dx', self.geo.f_dH_dx),
            ('d2H_dxdxi', self.geo.f_d2H_dxdxi)
        ]:
            grid = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    try:
                        grid[i, j] = func(X[i, j], Xi[i, j])
                    except:
                        grid[i, j] = np.nan
            grids[name] = grid
        
        return grids
    
    def _compute_geodesics(self, params: List[Tuple]) -> List[Geodesic]:
        """Compute all geodesics"""
        geodesics = []
        for p in params:
            x0, xi0, t_max = p[:3]
            geo = self.geo.compute_geodesic(x0, xi0, t_max)
            geo.color = p[3] if len(p) > 3 else 'blue'
            geodesics.append(geo)
        return geodesics
    
    def _create_figure(self, X, Xi, grids, geodesics, periodic_orbits, spectrum, hbar):
        """Create the complete visualization figure"""
        fig = plt.figure(figsize=(24, 18))
        
        # Panel 1-8: Geometry
        self._plot_hamiltonian_surface(fig, X, Xi, grids['H'], geodesics, 1)
        self._plot_level_sets(fig, X, Xi, grids['H'], geodesics, 2)
        self._plot_vector_field(fig, X, Xi, grids, geodesics, 3)
        self._plot_group_velocity(fig, X, Xi, grids['dH_dxi'], geodesics, 4)
        self._plot_spatial_projection(fig, geodesics, 5)
        self._plot_jacobian(fig, geodesics, 6)
        self._plot_curvature(fig, X, Xi, grids['d2H_dxdxi'], geodesics, 7)
        self._plot_energy_conservation(fig, geodesics, 8)
        
        # Panel 9-15: Spectral analysis
        if periodic_orbits:
            self._plot_periodic_orbits(fig, X, Xi, grids['H'], periodic_orbits, 9)
            self._plot_period_energy(fig, periodic_orbits, 10)
            self._plot_ebk_quantization(fig, periodic_orbits, hbar, 11)
            
            if spectrum:
                self._plot_trace_formula(fig, spectrum, 12)
                self._plot_spectrum(fig, spectrum, 13)
                self._plot_stability(fig, periodic_orbits, 14)
                self._plot_level_spacing(fig, spectrum, 15)
        
        plt.suptitle(f'Geometric and Semiclassical Atlas: H = {self.geo.H}',
                     fontsize=18, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])

        
        return fig
    
    # Individual plotting methods (KISS principle: each does one thing)
    
    def _plot_hamiltonian_surface(self, fig, X, Xi, H_grid, geodesics, panel):
        """Panel 1: Hamiltonian surface in 3D"""
        ax = fig.add_subplot(3, 5, panel, projection='3d')
        ax.plot_surface(X, Xi, H_grid, cmap='viridis', alpha=0.8, 
                        linewidth=0, antialiased=True)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'red')
            ax.plot(geo.x, geo.xi, geo.H, color=color, linewidth=3)
            ax.scatter([geo.x[0]], [geo.xi[0]], [geo.H[0]], 
                       color=color, s=100, edgecolors='black', linewidths=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_zlabel('H(x,ξ)')
        ax.set_title('Hamiltonian Surface\n+ Geodesics', fontweight='bold')
        ax.view_init(elev=25, azim=45)
        
        # 🔧 Ajustements pour taille cohérente
        ax.set_box_aspect((1, 1, 0.6))   # équilibre visuel (x, ξ, H)
        ax.margins(0)                    # supprime marges internes
        ax.set_proj_type('ortho')        # projection orthographique = moins de distorsion
    
    def _plot_level_sets(self, fig, X, Xi, H_grid, geodesics, panel):
        """Panel 2: Energy level sets (symplectic foliation)"""
        ax = fig.add_subplot(3, 5, panel)
        levels = np.linspace(np.nanmin(H_grid), np.nanmax(H_grid), 20)
        contour = ax.contour(X, Xi, H_grid, levels=levels, cmap='viridis')
        ax.clabel(contour, inline=True, fontsize=8)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'red')
            ax.plot(geo.x, geo.xi, color=color, linewidth=2.5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Level Sets H=const\nSymplectic Foliation', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('auto')     
        ax.margins(0.05)          
    
    
    def _plot_vector_field(self, fig, X, Xi, grids, geodesics, panel):
        """Panel 3: Hamiltonian vector field"""
        ax = fig.add_subplot(3, 5, panel)
        
        step = max(1, X.shape[0] // 20)
        X_sub = X[::step, ::step]
        Xi_sub = Xi[::step, ::step]
        vx = grids['dH_dxi'][::step, ::step]
        vy = -grids['dH_dx'][::step, ::step]
        
        magnitude = np.sqrt(vx**2 + vy**2)
        magnitude[magnitude == 0] = 1
        
        ax.quiver(X_sub, Xi_sub, vx/magnitude, vy/magnitude,
                 magnitude, cmap='plasma', alpha=0.7)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'cyan')
            ax.plot(geo.x, geo.xi, color=color, linewidth=3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Hamiltonian Vector Field\n(Infinitesimal generator)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_group_velocity(self, fig, X, Xi, dH_dxi, geodesics, panel):
        """Panel 4: Group velocity ∂H/∂ξ"""
        ax = fig.add_subplot(3, 5, panel)
        
        im = ax.contourf(X, Xi, dH_dxi, levels=30, cmap='RdBu_r')
        plt.colorbar(im, ax=ax, label='∂H/∂ξ')
        ax.contour(X, Xi, dH_dxi, levels=[0], colors='black', 
                  linewidths=2, linestyles='--')
        
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color='yellow', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Group Velocity v_g = ∂H/∂ξ\n(Wave propagation speed)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_spatial_projection(self, fig, geodesics, panel):
        """Panel 5: Spatial projection (with caustics)"""
        ax = fig.add_subplot(3, 5, panel)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.t, color=color, linewidth=2.5)
            
            # Mark caustics
            caust_idx = geo.caustics
            if len(caust_idx) > 0:
                ax.scatter(geo.x[caust_idx], geo.t[caust_idx],
                          color='red', s=150, marker='*', zorder=15,
                          edgecolors='darkred', linewidths=1.5)
        
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_title('Spatial Projection\n★ = Caustics', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_jacobian(self, fig, geodesics, panel):
        """Panel 6: Jacobian (focusing measure)"""
        ax = fig.add_subplot(3, 5, panel)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.t, geo.J, color=color, linewidth=2.5)
        
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('t')
        ax.set_ylabel('J = ∂x/∂ξ₀')
        ax.set_title('Jacobian (Focusing)\nJ→0: rays converge', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_curvature(self, fig, X, Xi, curvature, geodesics, panel):
        """Panel 7: Sectional curvature"""
        ax = fig.add_subplot(3, 5, panel)
        
        im = ax.contourf(X, Xi, curvature, levels=30, cmap='seismic')
        plt.colorbar(im, ax=ax, label='∂²H/∂x∂ξ')
        
        for geo in geodesics:
            ax.plot(geo.x, geo.xi, color='lime', linewidth=2)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Sectional Curvature\nRed>0: focusing | Blue<0: defocusing', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_energy_conservation(self, fig, geodesics, panel):
        """Panel 8: Energy conservation (integration quality)"""
        ax = fig.add_subplot(3, 5, panel)
        
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            H_variation = (geo.H - geo.H[0]) / (np.abs(geo.H[0]) + 1e-10)
            ax.semilogy(geo.t, np.abs(H_variation) + 1e-16,
                       color=color, linewidth=2.5, label=f'E₀={geo.H[0]:.2f}')
        
        ax.set_xlabel('t')
        ax.set_ylabel('|ΔH/H₀|')
        ax.set_title('Energy Conservation\n(Numerical quality)', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_periodic_orbits(self, fig, X, Xi, H_grid, periodic_orbits, panel):
        """Panel 9: Periodic orbits in phase space"""
        ax = fig.add_subplot(3, 5, panel)
        
        # Energy level sets
        energies = np.unique([orb.energy for orb in periodic_orbits])
        contour = ax.contour(X, Xi, H_grid, levels=energies, 
                            cmap='viridis', linewidths=1.5, alpha=0.6)
        
        # Periodic orbits
        colors_orb = plt.cm.rainbow(np.linspace(0, 1, len(periodic_orbits)))
        for idx, orb in enumerate(periodic_orbits):
            ax.plot(orb.x_cycle, orb.xi_cycle, 
                   color=colors_orb[idx], linewidth=3, alpha=0.8)
            ax.scatter([orb.x0], [orb.xi0], color=colors_orb[idx], 
                      s=100, marker='o', edgecolors='black', linewidths=2, zorder=10)
        
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Periodic Orbits\n(Phase space)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_period_energy(self, fig, periodic_orbits, panel):
        """Panel 10: Period-Energy relation"""
        ax = fig.add_subplot(3, 5, panel)
        
        E_orb = [orb.energy for orb in periodic_orbits]
        T_orb = [orb.period for orb in periodic_orbits]
        S_orb = [orb.action for orb in periodic_orbits]
        
        scatter = ax.scatter(E_orb, T_orb, c=S_orb, s=150,
                           cmap='plasma', edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax, label='Action S')
        
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Period T')
        ax.set_title('Period-Energy Diagram\nT(E)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_ebk_quantization(self, fig, periodic_orbits, hbar, panel):
        """Panel 11: EBK quantization (Einstein-Brillouin-Keller)"""
        ax = fig.add_subplot(3, 5, panel)
        
        E_orb = [orb.energy for orb in periodic_orbits]
        S_orb = [orb.action for orb in periodic_orbits]
        T_orb = [orb.period for orb in periodic_orbits]
        
        scatter = ax.scatter(E_orb, S_orb, s=150, c=T_orb, cmap='cool',
                           edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax, label='Period T')
        
        # EBK quantization rules: S = 2πℏ(n + α)
        E_max = max(E_orb) if E_orb else 10
        for n in range(15):
            S_quant = 2 * np.pi * hbar * (n + 0.25)  # α ≈ 1/4 for 1D
            if S_quant < max(S_orb) if S_orb else 10:
                ax.axhline(S_quant, color='red', linestyle='--', alpha=0.3, linewidth=1)
                ax.text(min(E_orb) if E_orb else 0, S_quant, f'n={n}',
                       fontsize=8, color='red', va='bottom')
        
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Action S')
        ax.set_title('EBK Quantization\nS = 2πℏ(n+α)', fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def _plot_trace_formula(self, fig, spectrum, panel):
        """Panel 12: Gutzwiller trace formula"""
        ax = fig.add_subplot(3, 5, panel)
        
        # Plot only first part for clarity
        n_plot = min(500, len(spectrum.trace_t))
        ax.plot(spectrum.trace_t[:n_plot], np.real(spectrum.trace[:n_plot]),
               'b-', linewidth=1.5, label='Re[Tr]')
        ax.plot(spectrum.trace_t[:n_plot], np.imag(spectrum.trace[:n_plot]),
               'r-', linewidth=1.5, alpha=0.7, label='Im[Tr]')
        
        ax.set_xlabel('Time t')
        ax.set_ylabel('Tr[exp(-iHt/ℏ)]')
        ax.set_title('Gutzwiller Trace Formula\nΣ_γ A_γ exp(iS_γ/ℏ)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_spectrum(self, fig, spectrum, panel):
        """Panel 13: Semiclassical spectrum"""
        ax = fig.add_subplot(3, 5, panel)
        
        # Only positive energies
        mask = spectrum.energies > 0
        E_positive = spectrum.energies[mask]
        I_positive = spectrum.intensity[mask]
        
        # Detect peaks
        peaks, properties = find_peaks(I_positive, 
                                      height=np.max(I_positive)*0.1,
                                      distance=20)
        
        ax.plot(E_positive, I_positive, 'b-', linewidth=1.5)
        ax.plot(E_positive[peaks], I_positive[peaks],
               'ro', markersize=10, label='Energy levels')
        
        # Annotate first levels
        for i, peak in enumerate(peaks[:10]):
            E_level = E_positive[peak]
            ax.text(E_level, I_positive[peak], f'E_{i}',
                   fontsize=9, ha='center', va='bottom')
        
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Spectral density')
        ax.set_title('Semiclassical Spectrum\n(Fourier transform of trace)', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_stability(self, fig, periodic_orbits, panel):
        """Panel 14: Orbit stability (Lyapunov exponents)"""
        ax = fig.add_subplot(3, 5, panel)
        
        stab = [orb.stability for orb in periodic_orbits]
        E_stab = [orb.energy for orb in periodic_orbits]
        T_stab = [orb.period for orb in periodic_orbits]
        
        scatter = ax.scatter(E_stab, stab, s=150, c=T_stab, cmap='autumn',
                           edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax, label='Period T')
        ax.axhline(0, color='green', linestyle='--', linewidth=2,
                  label='Marginal stability')
        
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Lyapunov exponent λ')
        ax.set_title('Orbit Stability\nλ>0: unstable | λ<0: stable', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_level_spacing(self, fig, spectrum, panel):
        """Panel 15: Level spacing distribution (integrability test)"""
        ax = fig.add_subplot(3, 5, panel)
        
        # Extract energy levels
        mask = spectrum.energies > 0
        E_positive = spectrum.energies[mask]
        I_positive = spectrum.intensity[mask]
        
        peaks, _ = find_peaks(I_positive, height=np.max(I_positive)*0.1)
        
        if len(peaks) > 1:
            E_levels = E_positive[peaks]
            spacings = np.diff(E_levels)
            
            # Normalize spacings
            s_mean = np.mean(spacings)
            s_normalized = spacings / s_mean
            
            # Histogram
            ax.hist(s_normalized, bins=20, density=True, alpha=0.7,
                   color='blue', edgecolor='black', label='Data')
            
            # Theoretical distributions
            s = np.linspace(0, np.max(s_normalized), 100)
            
            # Poisson (integrable systems)
            poisson = np.exp(-s)
            ax.plot(s, poisson, 'g--', linewidth=2, label='Poisson (integrable)')
            
            # Wigner (chaotic systems)
            wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
            ax.plot(s, wigner, 'r-', linewidth=2, label='Wigner (chaotic)')
            
            ax.set_xlabel('Normalized spacing s')
            ax.set_ylabel('P(s)')
            ax.set_title('Level Spacing Distribution\nIntegrable vs Chaotic', fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)


# ============================================================================
# UTILITY: ADDITIONAL ANALYSES
# ============================================================================

class SpectralAnalysis:
    """
    Additional spectral analysis tools
    """
    
    @staticmethod
    def weyl_law(energy: float, dimension: int, hbar: float = 1.0) -> float:
        """
        Weyl's law: asymptotic density of states
        
        N(E) ~ (1/2πℏ)^d × Vol{H(x,p) ≤ E}
        
        Parameters
        ----------
        energy : float
            Energy threshold
        dimension : int
            Phase space dimension
        hbar : float
            Reduced Planck constant
            
        Returns
        -------
        float
            Approximate number of states below energy E
        """
        # Simplified: assumes phase space volume ~ E^d
        prefactor = (1 / (2 * np.pi * hbar)) ** dimension
        return prefactor * (energy ** dimension)
    
    @staticmethod
    def analyze_integrability(spacings: np.ndarray) -> Dict:
        """
        Determine if system is integrable or chaotic via level statistics
        
        Parameters
        ----------
        spacings : array
            Energy level spacings
            
        Returns
        -------
        dict
            Statistical measures and classification
        """
        s_mean = np.mean(spacings)
        s_normalized = spacings / s_mean
        
        # Brody parameter (0: Poisson, 1: Wigner)
        # Fit P(s) = a s^β exp(-b s^(β+1))
        # Simplified: use ratio test
        
        # <s²>/<s>² ratio
        ratio = np.mean(s_normalized**2) / (np.mean(s_normalized)**2)
        
        # Poisson: ratio ≈ 2
        # Wigner: ratio ≈ 1.27
        
        if ratio > 1.7:
            classification = "Integrable (Poisson-like)"
        elif ratio < 1.4:
            classification = "Chaotic (Wigner-like)"
        else:
            classification = "Intermediate"
        
        return {
            'ratio': ratio,
            'mean_spacing': s_mean,
            'std_spacing': np.std(spacings),
            'classification': classification
        }
    
    @staticmethod
    def berry_tabor_formula(periodic_orbits: List[PeriodicOrbit], 
                           energy: float) -> float:
        """
        Berry-Tabor formula for integrable systems
        
        Smoothed density of states from periodic orbits
        
        Parameters
        ----------
        periodic_orbits : list
            Periodic orbits
        energy : float
            Energy at which to evaluate density
            
        Returns
        -------
        float
            Density of states ρ(E)
        """
        density = 0.0
        
        for orb in periodic_orbits:
            if abs(orb.energy - energy) < 0.1:
                # Contribution from this orbit
                # ρ(E) ~ T(E) / (2π) for integrable systems
                density += orb.period / (2 * np.pi)
        
        return density


# ============================================================================
# MAIN INTERFACE
# ============================================================================

def visualize_symbol(symbol: sp.Expr,
                    x_range: Tuple[float, float],
                    xi_range: Tuple[float, float],
                    geodesics_params: List[Tuple],
                    E_range: Optional[Tuple[float, float]] = None,
                    hbar: float = 1.0,
                    resolution: int = 100,
                    x_sym: Optional[sp.Symbol] = None,
                    xi_sym: Optional[sp.Symbol] = None) -> Tuple:
    """
    Main interface: complete visualization of a symbol with spectral analysis examples.
    
    Parameters
    ----------
    symbol : sympy expression
        Hamiltonian H(x, ξ)
    x_range, xi_range : tuple
        Domain (min, max)
    geodesics_params : list of tuples
        Each: (x0, xi0, t_max) or (x0, xi0, t_max, color)
    E_range : tuple, optional
        Energy range for spectral analysis
    hbar : float
        Reduced Planck constant
    resolution : int
        Grid resolution
    x_sym, xi_sym : sympy symbols, optional
        If not provided, will use 'x' and 'xi'
        
    Returns
    -------
    fig, geodesics, periodic_orbits, spectrum
        Complete visualization and computed data
    """
    # Set default symbols
    if x_sym is None:
        x_sym = sp.symbols('x', real=True)
    if xi_sym is None:
        xi_sym = sp.symbols('xi', real=True)
    
    # Initialize geometry engine
    geometry = SymbolGeometry(symbol, x_sym, xi_sym)
    
    # Create visualizer
    visualizer = SymbolVisualizer(geometry)
    
    # Generate complete visualization
    fig, geodesics, periodic_orbits, spectrum = visualizer.visualize_complete(
        x_range=x_range,
        xi_range=xi_range,
        geodesics_params=geodesics_params,
        E_range=E_range,
        hbar=hbar,
        resolution=resolution
    )
    
    # ==========================
    # SpectralAnalysis examples
    # ==========================
    if spectrum:
        print("\n=== Spectral Analysis Examples ===")
        
        # 1. Weyl law estimate
        E_max = max(spectrum.energies) if len(spectrum.energies) > 0 else 1.0
        dim = 1  # 1D phase space
        N_est = SpectralAnalysis.weyl_law(E_max, dim, hbar)
        print(f"Weyl law estimate of states below E={E_max:.2f}: N(E) ~ {N_est:.2f}")
        
        # 2. Level spacing analysis
        mask = spectrum.energies > 0
        E_positive = spectrum.energies[mask]
        I_positive = spectrum.intensity[mask]
        peaks, _ = find_peaks(I_positive, height=np.max(I_positive)*0.1)
        if len(peaks) > 1:
            E_levels = E_positive[peaks]
            spacings = np.diff(E_levels)
            integrability_info = SpectralAnalysis.analyze_integrability(spacings)
            print("Level spacing analysis:")
            print(integrability_info)
        
        # 3. Berry-Tabor formula (if periodic orbits found)
        if periodic_orbits:
            energy_eval = (E_range[0] + E_range[1]) / 2 if E_range else E_max/2
            rho = SpectralAnalysis.berry_tabor_formula(periodic_orbits, energy_eval)
            print(f"Berry-Tabor density of states at E={energy_eval:.2f}: ρ(E) ~ {rho:.3f}")
    
    return fig, geodesics, periodic_orbits, spectrum



# ============================================================================
# DOCUMENTATION
# ============================================================================

def print_theory():
    """Print theoretical background"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║           GEOMETRIC VISUALIZATION OF SYMBOLS                          ║
║        Pseudodifferential Operators & Spectral Analysis               ║
╚══════════════════════════════════════════════════════════════════════╝

GEOMETRIC STRUCTURE
───────────────────

1. HAMILTONIAN SURFACE
   H: T*M → ℝ defines a hypersurface in phase space
   Geodesics are integral curves of the Hamiltonian vector field

2. SYMPLECTIC FOLIATION
   Level sets {H = const} foliate the phase space
   Each leaf is a symplectic manifold

3. HAMILTONIAN FLOW
   Generated by X_H = (∂H/∂ξ, -∂H/∂x)
   Preserves symplectic structure

4. CAUSTICS
   Singularities where Jacobian ∂x/∂ξ₀ vanishes
   Focal points where rays concentrate

SPECTRAL THEORY
───────────────

5. WEYL'S LAW (1911)
   N(E) ~ (2πℏ)^(-d) Vol{H ≤ E}
   Relates spectrum to phase space volume

6. EBK QUANTIZATION (1917)
   ∮_γ p dq = 2πℏ(n + α)
   Periodic orbits quantize the spectrum

7. GUTZWILLER TRACE FORMULA (1971)
   Tr[e^(-iHt/ℏ)] = Σ_γ A_γ e^(iS_γ/ℏ)
   Relates spectrum to periodic orbits

8. BERRY-TABOR (1977)
   Integrable: P(s) ~ e^(-s) (Poisson)
   Level spacings are uncorrelated

9. BGS CONJECTURE (1984)
   Chaotic: P(s) ~ (πs/2) e^(-πs²/4) (Wigner)
   Level repulsion from chaos

10. COLIN DE VERDIÈRE (1985)
    Spectrum determines geometry (partially)
    "Can one hear the shape of a drum?"

11. FERMI'S GOLDEN RULE
    Γ = (2π/ℏ) |V|² ρ(E)
    Transition rate ~ density of states

KEY INSIGHTS
────────────

- Geometry encodes spectrum
- Periodic orbits = spectral oscillations
- Caustics = singularities of wave propagation
- Integrability ↔ Poisson statistics
- Chaos ↔ Wigner statistics

USAGE
─────

from symbol_visualization import visualize_symbol
import sympy as sp

x, xi = sp.symbols('x xi', real=True)
H = xi**2 + x**2  # Your Hamiltonian

fig, geodesics, orbits, spectrum = visualize_symbol(
    symbol=H,
    x_range=(-3, 3),
    xi_range=(-3, 3),
    geodesics_params=[
        (x0, xi0, t_max, color),
        ...
    ],
    E_range=(E_min, E_max),  # Optional: for spectral analysis
    hbar=1.0
)

plt.show()
    """)
