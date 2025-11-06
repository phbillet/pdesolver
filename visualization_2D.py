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
SymbolVisualizer2D: Geometric and semi-classical tool for 2D pseudo-differential operators
Unified version combining modular structure and rigorous caustic handling

Architecture:
- Modular object-oriented structure (v1)
- Full 4×4 Jacobian computation for precise caustic detection (v2)
- 18 geometric visualization panels (v1)
- Integration of semi-classical and quantum concepts (v2)
- KAM theory and topological analysis (v1)
- Monte Carlo phase space volume estimation (v2)
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import DiracDelta, Heaviside
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.signal import find_peaks
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional, Callable
import warnings
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from matplotlib.colors import LinearSegmentedColormap
warnings.filterwarnings('ignore')

# ============================================================================
# MODULAR DATA STRUCTURES (enhanced v1)
# ============================================================================

@dataclass
class Geodesic2D:

    """2D phase space geodesic trajectory with caustic analysis"""
    t: np.ndarray           # Time
    x: np.ndarray           # Position x
    y: np.ndarray           # Position y
    xi: np.ndarray          # Momentum ξ
    eta: np.ndarray         # Momentum η
    H: np.ndarray           # Energy
    J_full: np.ndarray      # Full 4x4 Jacobian (complete evolution)
    det_caustic: np.ndarray # Determinant ∂(x,y)/∂(ξ₀,η₀) for caustic detection
    caustic_indices: np.ndarray # Indices where caustics occur

    @property
    def energy(self) -> float:
        """Constant energy along the geodesic"""
        return self.H[0]

    @property
    def spatial_trajectory(self) -> np.ndarray:
        """Trajectory in configuration space (x,y)"""
        return np.column_stack([self.x, self.y])
    
    @property
    def momentum_trajectory(self) -> np.ndarray:
        """Trajectory in momentum space (ξ,η)"""
        return np.column_stack([self.xi, self.eta])
    
    @property
    def caustic_points(self) -> Tuple[np.ndarray, np.ndarray]:
        """Caustic points (x,y) along the trajectory"""
        if len(self.caustic_indices) == 0:
            return np.array([]), np.array([])
        return self.x[self.caustic_indices], self.y[self.caustic_indices]

@dataclass
class PeriodicOrbit2D:
    """2D periodic orbit in phase space with quantum characteristics"""
    x0: float
    y0: float
    xi0: float
    eta0: float
    period: float
    action: float           # Action S = ∮ (ξ dx + η dy)
    energy: float
    stability_1: float      # First Lyapunov exponent
    stability_2: float      # Second Lyapunov exponent
    x_cycle: np.ndarray
    y_cycle: np.ndarray
    xi_cycle: np.ndarray
    eta_cycle: np.ndarray
    t_cycle: np.ndarray
    maslov_index: int       # Maslov index (number of caustic crossings)

    @property
    def is_stable(self) -> bool:
        """Check if orbit is stable (KAM)"""
        return self.stability_1 < 0 and self.stability_2 < 0

    @property
    def bohr_sommerfeld_condition(self) -> float:
        """Bohr-Sommerfeld quantization condition with Maslov correction"""
        return self.action / (2 * np.pi) - self.maslov_index / 4

@dataclass
class CausticStructure:
    """Caustic structure with classification and physical properties"""
    x: np.ndarray
    y: np.ndarray
    t: float                # Time of appearance
    energy: float
    type: str               # 'fold', 'cusp', 'swallowtail'
    maslov_index: int       # Associated Maslov index
    strength: float         # Singularity intensity

# ============================================================================
# GEOMETRIC AND PHYSICAL ENGINE (merged v1 + v2)
# ============================================================================

def _sanitize(expr):
    """Remove DiracDelta, Heaviside, and undefined sign terms for numeric use."""
    expr = expr.replace(sp.DiracDelta, lambda *args: 0)
    expr = expr.replace(sp.Heaviside, lambda *args: 1)
    expr = sp.simplify(expr)
    return expr
    
class SymbolGeometry2D:
    """
    Full geometric and semi-classical analysis of a 2D symbol
    H(x, y, ξ, η) with 4D phase space and rigorous caustic treatment
    """
    def __init__(self, symbol: sp.Expr, 
                 x_sym: sp.Symbol, y_sym: sp.Symbol,
                 xi_sym: sp.Symbol, eta_sym: sp.Symbol,
                 hbar: float = 1.0):
        """
        Initialization with complete derivative computation for Jacobian evolution
        Parameters
        ----------
        symbol : sympy expression
            Hamiltonian H(x, y, ξ, η)
        x_sym, y_sym : sympy symbols
            Position coordinates
        xi_sym, eta_sym : sympy symbols
            Momentum coordinates
        hbar : float
            Reduced Planck constant (for quantum aspects)
        """
        self.H_sym = symbol
        self.x_sym = x_sym
        self.y_sym = y_sym
        self.xi_sym = xi_sym
        self.eta_sym = eta_sym
        self.hbar = hbar
            
        print(f"Initializing 2D geometry engine for H = {self.H_sym} with ℏ = {self.hbar}")
        # --- First derivatives (Hamiltonian vector field) ---
        dH_x = sp.diff(self.H_sym, self.x_sym)
        self.dH_dx_sym = _sanitize(dH_x)
        dH_y = sp.diff(self.H_sym, self.y_sym)
        self.dH_dy_sym = _sanitize(dH_y)
        dH_xi = sp.diff(self.H_sym, self.xi_sym)
        self.dH_dxi_sym = _sanitize(dH_xi)
        dH_eta = sp.diff(self.H_sym, self.eta_sym)
        self.dH_deta_sym = _sanitize(dH_eta)

        # --- Second derivatives for variational equations ---
        d2H_x2 = sp.diff(self.dH_dx_sym, self.x_sym)
        self.d2H_dx2_sym = _sanitize(d2H_x2)
        d2H_y2 = sp.diff(self.dH_dy_sym, self.y_sym)
        self.d2H_dy2_sym = _sanitize(d2H_y2)
        d2H_xi2 = sp.diff(self.dH_dxi_sym, self.xi_sym)
        self.d2H_dxi2_sym = _sanitize(d2H_xi2)
        d2H_eta2 = sp.diff(self.dH_deta_sym, self.eta_sym)
        self.d2H_deta2_sym = _sanitize(d2H_eta2)
        d2H_xy = sp.diff(self.dH_dx_sym, self.y_sym)
        self.d2H_dxdy_sym = _sanitize(d2H_xy)
        d2H_xxi = sp.diff(self.dH_dx_sym, self.xi_sym)
        self.d2H_dxdxi_sym = _sanitize(d2H_xxi)
        d2H_xeta = sp.diff(self.dH_dx_sym, self.eta_sym)
        self.d2H_dxdeta_sym = _sanitize(d2H_xeta)
        d2H_yxi = sp.diff(self.dH_dy_sym, self.xi_sym)
        self.d2H_dydxi_sym = _sanitize(d2H_yxi)
        d2H_yeta = sp.diff(self.dH_dy_sym, self.eta_sym)
        self.d2H_dyeta_sym = _sanitize(d2H_yeta)
        d2H_xieta = sp.diff(self.dH_dxi_sym, self.eta_sym)
        self.d2H_dxideta_sym = _sanitize(d2H_xieta)
        # --- Hessian for variational equations ---
        self.Hessian = sp.Matrix([
            [self.d2H_dx2_sym, self.d2H_dxdy_sym, self.d2H_dxdxi_sym, self.d2H_dxdeta_sym],
            [self.d2H_dxdy_sym, self.d2H_dy2_sym, self.d2H_dydxi_sym, self.d2H_dyeta_sym],
            [self.d2H_dxdxi_sym, self.d2H_dydxi_sym, self.d2H_dxi2_sym, self.d2H_dxideta_sym],
            [self.d2H_dxdeta_sym, self.d2H_dyeta_sym, self.d2H_dxideta_sym, self.d2H_deta2_sym]
        ])

        # --- Convert to numerical functions ---
        self._lambdify_functions()
  
    def _safe_lambdify(self, args: tuple, expr: sp.Expr) -> Callable:
        """Safe conversion of sympy expressions to numerical functions"""
        if isinstance(expr, (int, float, sp.Integer, sp.Float)):
            const_val = float(expr)
            return lambda x, y, xi, eta: np.full_like(x, const_val)
        try:
            return sp.lambdify(args, expr, modules=['numpy', 'scipy'])
        except Exception as e:
            print(f"Warning: lambdify failed for {expr}. Error: {e}")
            return lambda x, y, xi, eta: np.full_like(x, np.nan)

    def _lambdify_functions(self):
        """Convert all symbolic expressions to numerical functions"""
        args = (self.x_sym, self.y_sym, self.xi_sym, self.eta_sym)
        self.H_num = self._safe_lambdify(args, self.H_sym)
        self.dH_dx_num = self._safe_lambdify(args, self.dH_dx_sym)
        self.dH_dy_num = self._safe_lambdify(args, self.dH_dy_sym)
        self.dH_dxi_num = self._safe_lambdify(args, self.dH_dxi_sym)
        self.dH_deta_num = self._safe_lambdify(args, self.dH_deta_sym)
        # Hessian functions
        self.second_derivs_funcs = []
        for i in range(4):
            row_funcs = []
            for j in range(4):
                row_funcs.append(self._safe_lambdify(args, self.Hessian[i,j]))
            self.second_derivs_funcs.append(row_funcs)
    
    def _hamiltonian_system_augmented(self, t: float, z: np.ndarray) -> np.ndarray:
        """
        Augmented Hamiltonian system with variational equations for Jacobian evolution
        State vector z = [x, y, xi, eta, J11, J12, ..., J44] (20 dimensions)
        """
        # Extract position and momentum
        x, y, xi, eta = z[0:4]
        # Extract Jacobian matrix (4x4)
        J = z[4:].reshape((4, 4))
        try:
            # Hamilton's equations
            dx = float(self.dH_dxi_num(x, y, xi, eta))
            dy = float(self.dH_deta_num(x, y, xi, eta))
            dxi = float(-self.dH_dx_num(x, y, xi, eta))
            deta = float(-self.dH_dy_num(x, y, xi, eta))
            # Evaluate numerical Hessian
            Hessian_num = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    Hessian_num[i, j] = float(self.second_derivs_funcs[i][j](x, y, xi, eta))
            # Symplectic matrix J0
            J0 = np.array([
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [-1, 0, 0, 0],
                [0, -1, 0, 0]
            ])
            # Variational equations: dJ/dt = J @ (J0 @ Hessian)
            dJ_dt = J @ (J0 @ Hessian_num)
            # Build derivative vector
            dz = np.zeros(20)
            dz[0:4] = [dx, dy, dxi, deta]
            dz[4:] = dJ_dt.flatten()
            return dz
        except Exception as e:
            print(f"Integration error at t={t}, z={z}: {e}")
            return np.zeros(20)
    
    def compute_geodesic(self, x0: float, y0: float, 
                        xi0: float, eta0: float,
                        t_max: float, n_points: int = 500) -> Geodesic2D:
        """
        Compute a geodesic with full Jacobian evolution for caustic detection
        Parameters
        ----------
        x0, y0 : float
            Initial position
        xi0, eta0 : float
            Initial momentum
        t_max : float
            Final time
        n_points : int
            Number of sampling points
        Returns
        -------
        Geodesic2D
            Structure containing trajectory and caustic analysis
        """
        # Initial condition: position, momentum + identity Jacobian
        z0 = np.zeros(20)
        z0[0:4] = [x0, y0, xi0, eta0]
        z0[4:] = np.eye(4).flatten()
        t_eval = np.linspace(0, t_max, n_points)
        sol = solve_ivp(
            self._hamiltonian_system_augmented,
            [0, t_max], z0, t_eval=t_eval,
            method='DOP853', rtol=1e-9, atol=1e-12
        )
        if not sol.success:
            print(f"Warning: Integration failed for ({x0}, {y0}, {xi0}, {eta0})")
        # Extract trajectory data
        x_traj = sol.y[0]
        y_traj = sol.y[1]
        xi_traj = sol.y[2]
        eta_traj = sol.y[3]
        # Evaluate energy
        H_vals = self.H_num(x_traj, y_traj, xi_traj, eta_traj)
        # Extract and reshape Jacobian matrices
        J_mats = np.zeros((n_points, 4, 4))
        for i in range(n_points):
            J_mats[i] = sol.y[4:, i].reshape((4, 4))
        # Submatrix for caustic detection: ∂(x,y)/∂(ξ₀,η₀)
        caustic_matrix = J_mats[:, 0:2, 2:4]
        # Determinant for caustic detection
        det_caustic = np.zeros(n_points)
        for i in range(n_points):
            det_caustic[i] = np.linalg.det(caustic_matrix[i])
        # Detect caustic indices (sign change)
        caustic_indices = np.where(np.diff(np.sign(det_caustic)))[0]
        return Geodesic2D(
            t=sol.t,
            x=x_traj,
            y=y_traj,
            xi=xi_traj,
            eta=eta_traj,
            H=H_vals,
            J_full=J_mats,
            det_caustic=det_caustic,
            caustic_indices=caustic_indices
        )
    
    def find_periodic_orbits_2d(self, energy: float,
                               x_range: Tuple[float, float],
                               y_range: Tuple[float, float],
                               xi_range: Tuple[float, float],
                               eta_range: Tuple[float, float],
                               n_attempts: int = 30) -> List[PeriodicOrbit2D]:
        """
        Search for periodic orbits with Maslov index computation
        """
        orbits = []
        # Sample configuration space
        n_samples = int(np.sqrt(n_attempts))
        x_samples = np.linspace(x_range[0], x_range[1], n_samples)
        y_samples = np.linspace(y_range[0], y_range[1], n_samples)
        for x0 in x_samples:
            for y0 in y_samples:
                # Test different momentum directions
                angles = np.linspace(0, 2*np.pi, 8)
                for angle in angles:
                    for r in np.linspace(0.5, 3, 3):
                        xi0_guess = r * np.cos(angle)
                        eta0_guess = r * np.sin(angle)
                        try:
                            # Energy check
                            E_test = self.H_num(x0, y0, xi0_guess, eta0_guess)
                            if abs(E_test - energy) > 0.5:
                                continue
                            # Compute geodesic
                            geo = self.compute_geodesic(x0, y0, xi0_guess, eta0_guess, 15, 1500)
                            # Search for return points
                            distances = np.sqrt((geo.x - x0)**2 + (geo.y - y0)**2 +
                                              (geo.xi - xi0_guess)**2 + (geo.eta - eta0_guess)**2)
                            minima = []
                            for i in range(10, len(distances)-10):
                                if (distances[i] < distances[i-1] and
                                    distances[i] < distances[i+1] and
                                    distances[i] < 0.05):
                                    minima.append(i)
                            if minima:
                                idx = minima[0]
                                period = geo.t[idx]
                                if period > 0.2 and distances[idx] < 0.05:
                                    # Compute action
                                    x_cyc = geo.x[:idx+1]
                                    y_cyc = geo.y[:idx+1]
                                    xi_cyc = geo.xi[:idx+1]
                                    eta_cyc = geo.eta[:idx+1]
                                    t_cyc = geo.t[:idx+1]
                                    dx_dt = np.gradient(x_cyc, t_cyc)
                                    dy_dt = np.gradient(y_cyc, t_cyc)
                                    action = np.trapz(xi_cyc * dx_dt + eta_cyc * dy_dt, t_cyc)
                                    # Compute Maslov index (number of caustic crossings)
                                    maslov_index = len([i for i in geo.caustic_indices if i < idx])
                                    # Compute stability
                                    stab1 = self._compute_stability_2d(x0, y0, xi0_guess, eta0_guess, period)
                                    orbits.append(PeriodicOrbit2D(
                                        x0=x0, y0=y0,
                                        xi0=xi0_guess, eta0=eta0_guess,
                                        period=period,
                                        action=action,
                                        energy=energy,
                                        stability_1=stab1,
                                        stability_2=0.0,
                                        x_cycle=x_cyc,
                                        y_cycle=y_cyc,
                                        xi_cycle=xi_cyc,
                                        eta_cycle=eta_cyc,
                                        t_cycle=t_cyc,
                                        maslov_index=maslov_index
                                    ))
                        except Exception as e:
                            continue
        return self._remove_duplicate_orbits_2d(orbits)
    
    def _compute_stability_2d(self, x0, y0, xi0, eta0, T):
        """Compute the largest Lyapunov exponent"""
        def linearized(t, z):
            x, y, xi, eta, dx, dy, dxi, deta = z
            try:
                vx = float(self.dH_dxi_num(x, y, xi, eta))
                vy = float(self.dH_deta_num(x, y, xi, eta))
                vxi = float(-self.dH_dx_num(x, y, xi, eta))
                veta = float(-self.dH_dy_num(x, y, xi, eta))
                # Linearization (simplified)
                A13 = float(self.second_derivs_funcs[2][0](x, y, xi, eta))
                A24 = float(self.second_derivs_funcs[3][1](x, y, xi, eta))
                ddx = A13 * dxi
                ddy = A24 * deta
                ddxi = 0
                ddeta = 0
                return [vx, vy, vxi, veta, ddx, ddy, ddxi, ddeta]
            except:
                return [0]*8
        eps = 1e-6
        z0 = [x0, y0, xi0, eta0, eps, 0, 0, 0]
        sol = solve_ivp(linearized, [0, T], z0, method='DOP853', rtol=1e-10)
        if sol.success and len(sol.y[4]) > 0:
            pert = np.sqrt(sol.y[4][-1]**2 + sol.y[5][-1]**2)
            return np.log(pert / eps) / T
        return np.nan
    
    def _remove_duplicate_orbits_2d(self, orbits):
        """Remove duplicate periodic orbits"""
        unique = []
        for orb in orbits:
            is_dup = False
            for u_orb in unique:
                if (abs(orb.period - u_orb.period) < 0.2 and
                    abs(orb.action - u_orb.action) < 0.2):
                    is_dup = True
                    break
            if not is_dup:
                unique.append(orb)
        return unique
    
    def detect_caustic_structures(self, geodesics: List[Geodesic2D], 
                                 t_fixed: float) -> List[CausticStructure]:
        """
        Advanced caustic structure detection with classification
        """
        caustic_points = []
        for geo in geodesics:
            # Find closest time to t_fixed
            idx = np.argmin(np.abs(geo.t - t_fixed))
            # Check if near a caustic
            if abs(geo.det_caustic[idx]) < 0.1:
                # Classify caustic type
                caustic_type = self._classify_caustic(geo, idx)
                # Compute singularity strength
                strength = 1.0 / (abs(geo.det_caustic[idx]) + 0.01)
                caustic_points.append({
                    'x': geo.x[idx],
                    'y': geo.y[idx],
                    'energy': geo.energy,
                    'type': caustic_type,
                    'strength': strength
                })
        if len(caustic_points) < 3:
            return []
        # Cluster points into caustic structures
        caustic_structures = self._cluster_caustic_points(caustic_points, t_fixed)
        return caustic_structures
    
    def _classify_caustic(self, geo: Geodesic2D, idx: int) -> str:
        """
        Caustic classification according to catastrophe theory
        """
        # Compute curvature near caustic point
        window = 10
        start = max(0, idx - window)
        end = min(len(geo.t), idx + window + 1)
        if end - start < 5:
            return 'fold'
        # Curvature approximation
        x_window = geo.x[start:end]
        y_window = geo.y[start:end]
        dx = np.gradient(x_window)
        dy = np.gradient(y_window)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        with np.errstate(divide='ignore', invalid='ignore'):
            curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
        # Detect cusp points (high curvature)
        if np.max(curvature) > 2.0 * np.mean(curvature):
            return 'cusp'
        return 'fold'
    
    def _cluster_caustic_points(self, points: List[dict], t_fixed: float) -> List[CausticStructure]:
        """Group caustic points into coherent structures"""
        if not points:
            return []
        # Extract coordinates
        coords = np.array([[p['x'], p['y']] for p in points])
        # Simple proximity-based clustering
        clusters = []
        visited = set()
        for i, point in enumerate(points):
            if i in visited:
                continue
            # New cluster
            cluster = [point]
            visited.add(i)
            # Find nearby points
            for j, other in enumerate(points):
                if j in visited:
                    continue
                dist = np.sqrt((point['x'] - other['x'])**2 + (point['y'] - other['y'])**2)
                if dist < 0.5:  # Distance threshold
                    cluster.append(other)
                    visited.add(j)
            # Create caustic structure
            xs = np.array([p['x'] for p in cluster])
            ys = np.array([p['y'] for p in cluster])
            types = [p['type'] for p in cluster]
            strengths = [p['strength'] for p in cluster]
            # Majority type
            type_counts = {}
            for t in types:
                type_counts[t] = type_counts.get(t, 0) + 1
            dominant_type = max(type_counts.items(), key=lambda x: x[1])[0]
            # Maslov index (approximation)
            maslov_index = 1 if dominant_type == 'fold' else 2
            clusters.append(CausticStructure(
                x=xs,
                y=ys,
                t=t_fixed,
                energy=cluster[0]['energy'],
                type=dominant_type,
                maslov_index=maslov_index,
                strength=np.mean(strengths)
            ))
        return clusters
    
    def compute_phase_space_volume(self, E_max: float, x_range: tuple, y_range: tuple,
                                 xi_range: tuple, eta_range: tuple, 
                                 n_samples: int = 200000) -> float:
        """Monte Carlo estimation of phase space volume for H ≤ E_max"""
        # Generate random samples
        x_samples = np.random.uniform(x_range[0], x_range[1], n_samples)
        y_samples = np.random.uniform(y_range[0], y_range[1], n_samples)
        xi_samples = np.random.uniform(xi_range[0], xi_range[1], n_samples)
        eta_samples = np.random.uniform(eta_range[0], eta_range[1], n_samples)
        # Evaluate Hamiltonian
        H_vals = self.H_num(x_samples, y_samples, xi_samples, eta_samples)
        # Count points where H ≤ E_max
        volume_ratio = np.mean(H_vals <= E_max)
        # Total phase space volume
        total_volume = ((x_range[1]-x_range[0]) * (y_range[1]-y_range[0]) * 
                       (xi_range[1]-xi_range[0]) * (eta_range[1]-eta_range[0]))
        return volume_ratio * total_volume

# ============================================================================
# COMPLETE VISUALIZATION ENGINE (merged v1 + v2)
# ============================================================================
class SymbolVisualizer2D:
    """
    Complete visualization combining geometric and physical aspects
    """
    def __init__(self, geometry: SymbolGeometry2D):
        self.geo = geometry

    def visualize_complete(self,
                          x_range: Tuple[float, float],
                          y_range: Tuple[float, float],
                          xi_range: Tuple[float, float],
                          eta_range: Tuple[float, float],
                          geodesics_params: List[Tuple],
                          E_range: Optional[Tuple[float, float]] = None,
                          hbar: float = 1.0,
                          resolution: int = 50) -> Tuple:
        """
        Create a complete 18-panel visualization combining geometry and physics
        Parameters
        ----------
        x_range, y_range : tuple
            Configuration space domain
        xi_range, eta_range : tuple
            Momentum space domain
        geodesics_params : list
            Geodesic parameters: (x0, y0, xi0, eta0, t_max, color)
        E_range : tuple, optional
            Energy interval for spectral analysis
        hbar : float
            Reduced Planck constant
        resolution : int
            Grid resolution
        Returns
        -------
        fig, geodesics, periodic_orbits, caustics
        """
        # Compute geodesics with caustic detection
        geodesics = self._compute_geodesics(geodesics_params)
        # Search for periodic orbits
        periodic_orbits = []
        if E_range:
            energies = np.linspace(E_range[0], E_range[1], 5)
            for E in energies:
                orbits = self.geo.find_periodic_orbits_2d(
                    E, x_range, y_range, xi_range, eta_range, n_attempts=20
                )
                periodic_orbits.extend(orbits)
        # Detect caustic structures
        caustics = []
        if geodesics:
            t_samples = np.linspace(0, geodesics[0].t[-1], 5)
            for t in t_samples:
                caustics.extend(self.geo.detect_caustic_structures(geodesics, t))
        # Create full figure
        fig = self._create_complete_figure(
            E_range, x_range, y_range, xi_range, eta_range,
            geodesics, periodic_orbits, caustics, hbar, resolution
        )
        return fig, geodesics, periodic_orbits, caustics
    
    def _compute_geodesics(self, params):
        """Compute geodesics with caustic detection"""
        geodesics = []
        for p in params:
            x0, y0, xi0, eta0, t_max = p[:5]
            geo = self.geo.compute_geodesic(x0, y0, xi0, eta0, t_max)
            geo.color = p[5] if len(p) > 5 else 'blue'
            geodesics.append(geo)
        return geodesics

    
    def _create_complete_figure(self, E_range, x_range, y_range, xi_range, eta_range,
                               geodesics, periodic_orbits, caustics, hbar, resolution):
        """Creates an adaptive multi-panel figure: only relevant panels are displayed."""
        
        # --- List of panels with explicit call signatures ---
        panels_to_plot = []
    
        # Always safe to plot if data exists
        if geodesics:
            panels_to_plot.append(lambda ax_spec: self._plot_energy_surface_2d(fig, ax_spec, x_range, y_range, geodesics, resolution))
            panels_to_plot.append(lambda ax_spec: self._plot_configuration_space(fig, ax_spec, geodesics, caustics))
            panels_to_plot.append(lambda ax_spec: self._plot_phase_projection_x(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_phase_projection_y(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_momentum_space(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_vector_field_2d(fig, ax_spec, x_range, y_range, geodesics, resolution))
            panels_to_plot.append(lambda ax_spec: self._plot_group_velocity_2d(fig, ax_spec, x_range, y_range, geodesics, resolution))
            panels_to_plot.append(lambda ax_spec: self._plot_caustic_curves_2d(fig, ax_spec, geodesics, caustics))
            panels_to_plot.append(lambda ax_spec: self._plot_jacobian_evolution(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_energy_conservation_2d(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_poincare_x(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_poincare_y(fig, ax_spec, geodesics))
            panels_to_plot.append(lambda ax_spec: self._plot_caustic_network(fig, ax_spec, x_range, y_range, geodesics))
    
        if geodesics and caustics:
            pass  # already handled above
    
        if periodic_orbits:
            panels_to_plot.append(lambda ax_spec: self._plot_periodic_orbits_3d(fig, ax_spec, periodic_orbits))
            panels_to_plot.append(lambda ax_spec: self._plot_action_energy_2d(fig, ax_spec, periodic_orbits))
            panels_to_plot.append(lambda ax_spec: self._plot_torus_quantization(fig, ax_spec, periodic_orbits, hbar))
            if len(periodic_orbits) > 2:
                panels_to_plot.append(lambda ax_spec: self._plot_level_spacing_2d(fig, ax_spec, periodic_orbits))
    
        if periodic_orbits and E_range:
            panels_to_plot.append(lambda ax_spec: self._plot_spectral_density_with_caustics(fig, ax_spec, periodic_orbits, E_range))
    
        # Always plot Maslov (demo)
        panels_to_plot.append(lambda ax_spec: self._plot_maslov_index_phase_shifts(fig, ax_spec, geodesics, caustics))
    
        if E_range:
            panels_to_plot.append(lambda ax_spec: self._plot_phase_space_volume(fig, ax_spec, E_range, x_range, y_range, xi_range, eta_range))
    
        # --- Handle empty case ---
        if not panels_to_plot:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, "No panels to display for this Hamiltonian.",
                    ha='center', va='center', fontsize=16, transform=ax.transAxes)
            ax.set_axis_off()
            return fig
    
        # --- Dynamic layout ---
        n = len(panels_to_plot)
        if n <= 5:
            cols, rows = n, 1
        elif n <= 10:
            cols, rows = 5, 2
        elif n <= 15:
            cols, rows = 5, 3
        else:
            cols, rows = 5, (n + 4) // 5
    
        figsize = (4.8 * cols, 4.0 * rows)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(rows, cols, figure=fig, hspace=0.5, wspace=0.3)
        plt.suptitle(f'Geometric and Semiclassical Atlas: H = {self.geo.H_sym} (ℏ={hbar})',
                     fontsize=18, fontweight='bold', y=0.98)
    
        # --- Plot all panels ---
        for idx, plot_cmd in enumerate(panels_to_plot):
            if idx >= rows * cols:
                break
            row = idx // cols
            col = idx % cols
            subplot_spec = gs[row, col]
            try:
                plot_cmd(subplot_spec)
            except Exception as e:
                ax = fig.add_subplot(subplot_spec)
                ax.text(0.5, 0.5, f"[Error]\n{type(e).__name__}", ha='center', va='center', color='red')
                ax.set_axis_off()
    
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        return fig

    # ======== DETAILED VISUALIZATION METHODS ========
    def _plot_energy_surface_2d(self, fig, subplot_spec, x_range, y_range, geodesics, res):
        """Energy surface H(x,y) at fixed (ξ,η)"""
        ax = fig.add_subplot(subplot_spec, projection='3d')
        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        # Evaluate at reference momentum
        xi_ref, eta_ref = 1.0, 1.0
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i,j] = self.geo.H_num(X[i,j], Y[i,j], xi_ref, eta_ref)
                except:
                    Z[i,j] = np.nan
        # Surface with transparency to see geodesics
        ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, edgecolor='none')
        # Geodesics on the surface
        for geo in geodesics[:5]:
            H_geo = np.array([self.geo.H_num(geo.x[i], geo.y[i], xi_ref, eta_ref)
                             for i in range(len(geo.t))])
            color = getattr(geo, 'color', 'red')
            ax.plot(geo.x, geo.y, H_geo, color=color, linewidth=2.5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('H')
        ax.set_title('Energy Surface\nH(x,y,ξ₀,η₀)', fontweight='bold', fontsize=10)
        ax.view_init(elev=25, azim=-45)
    
    def _plot_configuration_space(self, fig, subplot_spec, geodesics, caustics):
        """Configuration space (x,y) with trajectories and caustics"""
        ax = fig.add_subplot(subplot_spec)
        
        # Trajectories - use thinner lines and lighter colors for better visibility
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.y, color=color, linewidth=1.5, alpha=0.7, zorder=5)
            ax.scatter([geo.x[0]], [geo.y[0]], color=color, s=80, 
                      marker='o', edgecolors='black', linewidths=1.5, zorder=10)
        
        # Caustic points on trajectories - keep as stars but reduce size slightly
        for geo in geodesics:
            caust_x, caust_y = geo.caustic_points
            if len(caust_x) > 0:
                ax.scatter(caust_x, caust_y, c='red', s=80, marker='*',  # Reduced from 120
                          edgecolors='darkred', linewidths=1.0, zorder=15,
                          label='Caustic points')
        
        # Caustic structures - use smaller, more subtle markers
        for caust in caustics:
            color_map = {'fold': 'red', 'cusp': 'magenta', 'swallowtail': 'orange'}
            color = color_map.get(caust.type, 'red')
            # Use a small circle or dot instead of a large X
            marker = 'o'  # You can also try '.' for even smaller dots
            # Reduce size significantly and increase transparency
            size = 30  # Fixed size for clarity, or use: max(15, min(50, 80 * caust.strength / 2))
            alpha_val = 0.5  # More transparent to avoid obscuring trajectories
            
            ax.scatter(caust.x, caust.y, c=color, s=size, marker=marker,
                      edgecolors='none',  # Remove edge for cleaner look
                      linewidths=0, alpha=alpha_val, zorder=12,  # zorder between traj and points
                      label=f'Caustic {caust.type} (μ={caust.maslov_index})')
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Configuration Space\n★ = caustics', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')
    
    def _plot_jacobian_evolution(self, fig, subplot_spec, geodesics):
        """Evolution of Jacobian determinant with caustic detection"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.t, geo.det_caustic, color=color, linewidth=2.5, alpha=0.9,
                   label=f'E={geo.energy:.2f}')
            # Mark caustic points
            for idx in geo.caustic_indices:
                ax.scatter(geo.t[idx], geo.det_caustic[idx], s=100, marker='*',
                          color='red', edgecolor='darkred', zorder=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.set_xlabel('Time t')
        ax.set_ylabel('det(∂(x,y)/∂(ξ₀,η₀))')
        ax.set_title('Jacobian Determinant\nZeros = caustics', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_maslov_index_phase_shifts(self, fig, subplot_spec, geodesics, caustics):
        """Visualization of phase shifts due to Maslov index"""
        ax = fig.add_subplot(subplot_spec)
        # Simulate wavefunction crossing caustics
        x_demo = np.linspace(-4, 4, 1000)
        k = 2.0  # Wavenumber
        # Free wavefunction (before caustic)
        psi_free = np.exp(1j * k * x_demo**2 / 2)
        # Simulate phase shifts at caustics
        caustic_positions = [-2.0, 0.0, 2.0]  # Caustic positions
        maslov_indices = [1, 2, 1]  # Maslov index for each caustic
        psi_with_shifts = np.zeros_like(psi_free, dtype=complex)
        current_phase = 0.0
        for i, x in enumerate(x_demo):
            # Check if crossing a caustic
            for j, caust_x in enumerate(caustic_positions):
                if abs(x - caust_x) < 0.05:
                    current_phase -= maslov_indices[j] * np.pi / 2
            psi_with_shifts[i] = psi_free[i] * np.exp(1j * current_phase)
        # Plot real parts
        ax.plot(x_demo, np.real(psi_free), 'b-', alpha=0.8, linewidth=2, 
                label='Re[ψ] before caustics')
        ax.plot(x_demo, np.real(psi_with_shifts), 'r-', alpha=0.8, linewidth=2, 
                label='Re[ψ] after caustics')
        # Mark caustic positions
        for i, caust_x in enumerate(caustic_positions):
            ax.axvline(caust_x, color='k', linestyle='--', alpha=0.7,
                      label=f'Caustic μ={maslov_indices[i]}')
        ax.set_xlabel('Position x')
        ax.set_ylabel('Re[ψ(x)]')
        ax.set_title('Maslov Index\nPhase shifts at caustics', fontweight='bold', fontsize=10)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    def _plot_spectral_density_with_caustics(self, fig, subplot_spec, periodic_orbits, E_range):
        """Spectral density with caustic corrections"""
        ax = fig.add_subplot(subplot_spec)
        if not periodic_orbits:
            ax.text(0.5, 0.5, 'No periodic orbits', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        # Sort orbits by energy
        orbits_sorted = sorted(periodic_orbits, key=lambda x: x.energy)
        energies = np.array([orb.energy for orb in orbits_sorted])
        periods = np.array([orb.period for orb in orbits_sorted])
        # Compute state density ρ(E) = T(E)/(2π) for integrable systems
        if len(energies) > 1:
            dE = np.diff(energies)
            dT = np.diff(periods)
            rho_E = np.zeros_like(energies)
            rho_E[1:-1] = (periods[2:] - periods[:-2]) / (energies[2:] - energies[:-2])
            if len(rho_E) > 2:
                rho_E[0] = (periods[1] - periods[0]) / (energies[1] - energies[0])
                rho_E[-1] = (periods[-1] - periods[-2]) / (energies[-1] - energies[-2])
            rho_E = np.maximum(rho_E, 0)  # Avoid negative values
            # Caustic correction (oscillatory terms)
            rho_osc = np.zeros_like(rho_E)
            for orb in orbits_sorted:
                # Amplitude depending on Maslov index
                amp = 0.3 * np.exp(-orb.maslov_index/2) * orb.period
                phase = orb.action / self.geo.hbar - np.pi * orb.maslov_index / 2
                idx = np.argmin(np.abs(energies - orb.energy))
                if 0 <= idx < len(rho_osc):
                    rho_osc[idx] += amp * np.cos(phase)
            # Smooth curve
            E_fine = np.linspace(E_range[0], E_range[1], 500)
            from scipy.interpolate import interp1d
            try:
                interp_rho = interp1d(energies, rho_E, kind='cubic', fill_value="extrapolate")
                interp_osc = interp1d(energies, rho_osc, kind='cubic', fill_value="extrapolate")
                rho_smooth = np.maximum(0, interp_rho(E_fine))
                rho_osc_smooth = interp_osc(E_fine)
                # Plot components
                ax.plot(E_fine, rho_smooth, 'k-', linewidth=2.5, 
                       label='Smooth (Weyl)')
                ax.plot(E_fine, rho_smooth + rho_osc_smooth, 'b-', linewidth=2,
                       label='Total with caustics')
                ax.fill_between(E_fine, rho_smooth, rho_smooth + rho_osc_smooth, 
                               where=rho_osc_smooth>0, color='#ff9999', alpha=0.4,
                               label='Caustic corrections')
            except:
                ax.plot(energies, rho_E, 'b-o', linewidth=2, label='State density ρ(E)')
        ax.set_xlabel('Energy E')
        ax.set_ylabel('ρ(E)')
        ax.set_title('Spectral Density\nwith caustic corrections', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_phase_space_volume(self, fig, subplot_spec, E_range, x_range, y_range, xi_range, eta_range):
        """Phase space volume via Monte Carlo"""
        ax = fig.add_subplot(subplot_spec)
        # Compute volume for different energies
        E_vals = np.linspace(E_range[0], E_range[1], 8)
        volumes = []
        print("Computing phase space volume (Monte Carlo)...")
        for E in E_vals:
            vol = self.geo.compute_phase_space_volume(E, x_range, y_range, xi_range, eta_range, n_samples=50000)
            volumes.append(vol)
            print(f"  E={E:.2f}, Volume={vol:.4f}")
        # Weyl law: N(E) ~ Vol/(2πℏ)²
        d = 2  # Dimension
        weyl_constant = (2 * np.pi * self.geo.hbar) ** d
        N_weyl = np.array(volumes) / weyl_constant
        ax.plot(E_vals, N_weyl, 'b-o', linewidth=2.5, markersize=8, 
                label=f'Weyl law: N(E) ~ Vol/(2πℏ)²', color='#1f77b4')
        # Conceptual caustic correction
        if len(E_vals) > 3:
            oscillation_freq = 5 / (E_range[1] - E_range[0])
            correction = 0.15 * N_weyl * np.sin(2 * np.pi * oscillation_freq * (E_vals - E_vals[0]) + 0.7)
            N_corrected = N_weyl + correction
            from scipy.ndimage import gaussian_filter1d
            N_corrected_smooth = gaussian_filter1d(N_corrected, sigma=1.0)
            ax.plot(E_vals, N_corrected_smooth, 'r--', linewidth=2, 
                   label="With caustic corrections", alpha=0.9)
        ax.set_xlabel('Energy E')
        ax.set_ylabel('N(E) (Number of states)')
        ax.set_title('Phase Space Volume\n(Monte Carlo)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    def _plot_caustic_network(self, fig, subplot_spec, x_range, y_range, geodesics):
        """Caustic network with multiple initial conditions"""
        ax = fig.add_subplot(subplot_spec)
        if not geodesics:
            ax.text(0.5, 0.5, 'No geodesics', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        # Use first geodesic as reference
        E_ref = geodesics[0].energy
        t_max = geodesics[0].t[-1]
        # Generate trajectory family
        n_family = 15
        x0_vals = np.linspace(x_range[0], x_range[1], n_family)
        caustic_points = []
        for x0 in x0_vals:
            try:
                # Solve for y0, xi0, eta0 keeping energy constant
                def energy_eq(vars):
                    y_val, xi_val, eta_val = vars
                    return self.geo.H_num(x0, y_val, xi_val, eta_val) - E_ref
                # Use initial values of first geodesic as guess
                y0_guess = geodesics[0].y[0]
                xi0_guess = geodesics[0].xi[0]
                eta0_guess = geodesics[0].eta[0]
                sol = fsolve(energy_eq, [y0_guess, xi0_guess, eta0_guess])
                if np.all(np.isfinite(sol)):
                    y0_new, xi0_new, eta0_new = sol
                    # Compute trajectory
                    geo = self.geo.compute_geodesic(x0, y0_new, xi0_new, eta0_new, t_max, n_points=300)
                    # Plot trajectory
                    ax.plot(geo.x, geo.y, color='blue', alpha=0.3, linewidth=1)
                    # Collect caustic points
                    caust_x, caust_y = geo.caustic_points
                    for i in range(len(caust_x)):
                        caustic_points.append((caust_x[i], caust_y[i]))
            except Exception as e:
                continue
        # Plot caustic points
        if caustic_points:
            caustic_points = np.array(caustic_points)
            ax.scatter(caustic_points[:, 0], caustic_points[:, 1], 
                      s=30, c='red', alpha=0.8, edgecolor='none',
                      label='Caustic points')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Caustic Network\n(Multiple initial conditions)', fontweight='bold', fontsize=10)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    # ======== STANDARD VISUALIZATION METHODS (similar to v1) ========
    # Following methods are similar to v1 but enhanced
    # to integrate caustics and new data structures
    def _plot_phase_projection_x(self, fig, subplot_spec, geodesics):
        """Phase space projection (x,ξ)"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.x, geo.xi, color=color, linewidth=2, alpha=0.8)
            ax.scatter([geo.x[0]], [geo.xi[0]], color=color, s=80,
                      marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Phase Space (x,ξ)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_phase_projection_y(self, fig, subplot_spec, geodesics):
        """Phase space projection (y,η)"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.y, geo.eta, color=color, linewidth=2, alpha=0.8)
            ax.scatter([geo.y[0]], [geo.eta[0]], color=color, s=80,
                      marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('y')
        ax.set_ylabel('η')
        ax.set_title('Phase Space (y,η)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_momentum_space(self, fig, subplot_spec, geodesics):
        """Momentum space (ξ,η)"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            ax.plot(geo.xi, geo.eta, color=color, linewidth=2, alpha=0.8)
            ax.scatter([geo.xi[0]], [geo.eta[0]], color=color, s=80,
                      marker='o', edgecolors='black', linewidths=1.5)
        ax.set_xlabel('ξ')
        ax.set_ylabel('η')
        ax.set_title('Momentum Space\n(ξ,η)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    def _plot_vector_field_2d(self, fig, subplot_spec, x_range, y_range, geodesics, res):
        """Vector field in configuration space"""
        ax = fig.add_subplot(subplot_spec)
        x = np.linspace(x_range[0], x_range[1], res//2)
        y = np.linspace(y_range[0], y_range[1], res//2)
        X, Y = np.meshgrid(x, y)
        # Evaluate vector field at reference momentum
        xi_ref, eta_ref = 1.0, 1.0
        VX = np.zeros_like(X)
        VY = np.zeros_like(Y)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    VX[i,j] = self.geo.dH_dxi_num(X[i,j], Y[i,j], xi_ref, eta_ref)
                    VY[i,j] = self.geo.dH_deta_num(X[i,j], Y[i,j], xi_ref, eta_ref)
                except:
                    VX[i,j] = VY[i,j] = np.nan
        # Magnitude for coloring
        magnitude = np.sqrt(VX**2 + VY**2)
        magnitude[magnitude == 0] = 1
        # Normalized vector field
        ax.quiver(X, Y, VX/magnitude, VY/magnitude, magnitude, 
                 cmap='plasma', alpha=0.7, scale=30)
        # Overlay geodesics
        for geo in geodesics[:5]:
            color = getattr(geo, 'color', 'white')
            ax.plot(geo.x, geo.y, color=color, linewidth=2.5, alpha=0.9)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Vector Field\nFlow in configuration space', fontweight='bold', fontsize=10)
        ax.set_aspect('equal')
    
    def _plot_group_velocity_2d(self, fig, subplot_spec, x_range, y_range, geodesics, res):
        """Group velocity magnitude |∇_p H|"""
        ax = fig.add_subplot(subplot_spec)
        x = np.linspace(x_range[0], x_range[1], res)
        y = np.linspace(y_range[0], y_range[1], res)
        X, Y = np.meshgrid(x, y)
        # Group velocity at reference momentum
        xi_ref, eta_ref = 1.0, 1.0
        V_mag = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    vx = self.geo.dH_dxi_num(X[i,j], Y[i,j], xi_ref, eta_ref)
                    vy = self.geo.dH_deta_num(X[i,j], Y[i,j], xi_ref, eta_ref)
                    V_mag[i,j] = np.sqrt(vx**2 + vy**2)
                except:
                    V_mag[i,j] = np.nan
        # Heatmap
        im = ax.contourf(X, Y, V_mag, levels=20, cmap='hot')
        plt.colorbar(im, ax=ax, label='|v_g|')
        # Geodesics
        for geo in geodesics[:5]:
            ax.plot(geo.x, geo.y, 'cyan', linewidth=2, alpha=0.8)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Group Velocity\n|∇_p H|', fontweight='bold', fontsize=10)
        ax.set_aspect('equal')
    
    def _plot_caustic_curves_2d(self, fig, subplot_spec, geodesics, caustics):
        """Caustic curves in (x,y) space"""
        ax = fig.add_subplot(subplot_spec)
        # All geodesics
        for geo in geodesics:
            color = getattr(geo, 'color', 'lightblue')
            ax.plot(geo.x, geo.y, color=color, linewidth=1.5, alpha=0.5)
            # Caustic points on each geodesic
            caust_x, caust_y = geo.caustic_points
            if len(caust_x) > 0:
                ax.scatter(caust_x, caust_y, c='red', s=80, marker='*', 
                          edgecolors='darkred', linewidths=1.5, zorder=10)
        # Complete caustic structures
        for caust in caustics:
            color_map = {'fold': 'red', 'cusp': 'magenta', 'swallowtail': 'orange'}
            color = color_map.get(caust.type, 'red')
            # If enough points, plot smoothed curve
            if len(caust.x) > 3:
                ax.plot(caust.x, caust.y, color=color, linewidth=3, 
                       label=f'Caustic {caust.type} (μ={caust.maslov_index})')
            else:
                ax.scatter(caust.x, caust.y, c=color, s=100, marker='X',
                          edgecolors='black', linewidths=1.5,
                          label=f'Caustic {caust.type}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('Caustic Curves\n★ = points on geodesics', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        # Legend without duplicates
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8)
    
    def _plot_energy_conservation_2d(self, fig, subplot_spec, geodesics):
        """Energy conservation verification"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            color = getattr(geo, 'color', 'blue')
            H_var = (geo.H - geo.H[0]) / (np.abs(geo.H[0]) + 1e-10)
            ax.semilogy(geo.t, np.abs(H_var) + 1e-16,
                       color=color, linewidth=2, label=f'E={geo.H[0]:.2f}')
        ax.set_xlabel('Time t')
        ax.set_ylabel('|ΔH/H₀|')
        ax.set_title('Energy Conservation\nNumerical quality', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_poincare_x(self, fig, subplot_spec, geodesics):
        """Poincaré section (x,ξ) at y=0"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            # Find y=0 crossings
            crossings_x = []
            crossings_xi = []
            for i in range(len(geo.y)-1):
                if geo.y[i] * geo.y[i+1] < 0:  # Sign change
                    alpha = -geo.y[i] / (geo.y[i+1] - geo.y[i])
                    x_cross = geo.x[i] + alpha * (geo.x[i+1] - geo.x[i])
                    xi_cross = geo.xi[i] + alpha * (geo.xi[i+1] - geo.xi[i])
                    crossings_x.append(x_cross)
                    crossings_xi.append(xi_cross)
            if crossings_x:
                color = getattr(geo, 'color', 'blue')
                ax.scatter(crossings_x, crossings_xi, c=color, s=50, alpha=0.7)
        ax.set_xlabel('x')
        ax.set_ylabel('ξ')
        ax.set_title('Poincaré Section\n(x,ξ) at y=0', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_poincare_y(self, fig, subplot_spec, geodesics):
        """Poincaré section (y,η) at x=0"""
        ax = fig.add_subplot(subplot_spec)
        for geo in geodesics:
            # Find x=0 crossings
            crossings_y = []
            crossings_eta = []
            for i in range(len(geo.x)-1):
                if geo.x[i] * geo.x[i+1] < 0:
                    alpha = -geo.x[i] / (geo.x[i+1] - geo.x[i])
                    y_cross = geo.y[i] + alpha * (geo.y[i+1] - geo.y[i])
                    eta_cross = geo.eta[i] + alpha * (geo.eta[i+1] - geo.eta[i])
                    crossings_y.append(y_cross)
                    crossings_eta.append(eta_cross)
            if crossings_y:
                color = getattr(geo, 'color', 'blue')
                ax.scatter(crossings_y, crossings_eta, c=color, s=50, alpha=0.7)
        ax.set_xlabel('y')
        ax.set_ylabel('η')
        ax.set_title('Poincaré Section\n(y,η) at x=0', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_periodic_orbits_3d(self, fig, subplot_spec, periodic_orbits):
        """Periodic orbits in 3D (x,y,t)"""
        ax = fig.add_subplot(subplot_spec, projection='3d')
        colors = plt.cm.rainbow(np.linspace(0, 1, min(10, len(periodic_orbits))))
        for idx, orb in enumerate(periodic_orbits[:10]):  # Limit for clarity
            ax.plot(orb.x_cycle, orb.y_cycle, orb.t_cycle,
                   color=colors[idx], linewidth=2.5, alpha=0.8)
            ax.scatter([orb.x0], [orb.y0], [0], color=colors[idx],
                      s=100, marker='o', edgecolors='black', linewidths=2)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('t')
        ax.set_title('Periodic Orbits\nSpace-time view', fontweight='bold', fontsize=10)
    
    def _plot_action_energy_2d(self, fig, subplot_spec, periodic_orbits):
        """Action vs Energy"""
        ax = fig.add_subplot(subplot_spec)
        E_orb = [orb.energy for orb in periodic_orbits]
        S_orb = [orb.action for orb in periodic_orbits]
        T_orb = [orb.period for orb in periodic_orbits]
        scatter = ax.scatter(E_orb, S_orb, c=T_orb, s=150,
                           cmap='plasma', edgecolors='black', linewidths=1.5)
        plt.colorbar(scatter, ax=ax, label='Period T')
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Action S')
        ax.set_title('Action-Energy\nS(E)', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    def _plot_torus_quantization(self, fig, subplot_spec, periodic_orbits, hbar):
        """Torus quantization (KAM theory)"""
        ax = fig.add_subplot(subplot_spec)
        E_orb = [orb.energy for orb in periodic_orbits]
        S_orb = [orb.action for orb in periodic_orbits]
        scatter = ax.scatter(E_orb, S_orb, s=150, c='blue',
                           edgecolors='black', linewidths=1.5, label='Orbits')
        # EBK quantization for 2D: S_i = 2πℏ(n_i + α_i)
        # Simplified for one dimension
        E_max = max(E_orb) if E_orb else 10
        for n in range(20):
            S_quant = 2 * np.pi * hbar * (n + 0.5)
            if S_quant < max(S_orb) if S_orb else 10:
                ax.axhline(S_quant, color='red', linestyle='--', alpha=0.3)
                ax.text(min(E_orb) if E_orb else 0, S_quant, 
                       f'n={n}', fontsize=7, color='red')
        ax.set_xlabel('Energy E')
        ax.set_ylabel('Action S')
        ax.set_title('Torus Quantization\nKAM theory', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _plot_level_spacing_2d(self, fig, subplot_spec, periodic_orbits):
        """Level spacing distribution"""
        ax = fig.add_subplot(subplot_spec)
        # Extract unique energies
        energies = sorted(set(orb.energy for orb in periodic_orbits))
        if len(energies) > 2:
            spacings = np.diff(energies)
            # Normalize
            s_mean = np.mean(spacings)
            s_norm = spacings / s_mean
            # Histogram
            ax.hist(s_norm, bins=15, density=True, alpha=0.7,
                   color='blue', edgecolor='black', label='Data')
            # Theoretical curves
            s = np.linspace(0, np.max(s_norm), 100)
            # Poisson (integrable systems)
            poisson = np.exp(-s)
            ax.plot(s, poisson, 'g--', linewidth=2, label='Poisson (Integrable)')
            # Wigner (chaotic systems)
            wigner = (np.pi * s / 2) * np.exp(-np.pi * s**2 / 4)
            ax.plot(s, wigner, 'r-', linewidth=2, label='Wigner (Chaotic)')
            ax.set_xlabel('Normalized spacing s')
            ax.set_ylabel('P(s)')
            ax.set_title('Level Spacing\nIntegrable vs Chaotic', fontweight='bold', fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

# ============================================================================
# MAIN INTERFACE
# ============================================================================
def visualize_symbol_2d(symbol: sp.Expr,
                       x_range: Tuple[float, float],
                       y_range: Tuple[float, float],
                       xi_range: Tuple[float, float],
                       eta_range: Tuple[float, float],
                       geodesics_params: List[Tuple],
                       E_range: Optional[Tuple[float, float]] = None,
                       hbar: float = 1.0,
                       resolution: int = 50,
                       x_sym: Optional[sp.Symbol] = None,
                       y_sym: Optional[sp.Symbol] = None,
                       xi_sym: Optional[sp.Symbol] = None,
                       eta_sym: Optional[sp.Symbol] = None) -> Tuple:
    """
    Main interface: full 2D visualization
    Parameters
    ----------
    symbol : sympy expression
        Hamiltonian H(x, y, ξ, η)
    x_range, y_range : tuple
        Configuration space domain
    xi_range, eta_range : tuple
        Momentum space domain
    geodesics_params : list of tuples
        Each tuple: (x0, y0, ξ0, η0, t_max) or (x0, y0, ξ0, η0, t_max, color)
    E_range : tuple, optional
        Energy interval for spectral analysis
    hbar : float
        Reduced Planck constant
    resolution : int
        Grid resolution
    x_sym, y_sym, xi_sym, eta_sym : sympy symbols, optional
        Variable symbols
    Returns
    -------
    fig, geodesics, periodic_orbits, caustics
    Example
    -------
    >>> x, y = sp.symbols('x y', real=True)
    >>> xi, eta = sp.symbols('xi eta', real=True)
    >>> H = xi**2 + eta**2 + x**2 + y**2 + 0.1*x**4  # 2D anharmonic oscillator
    >>> fig, geos, orbits, caust = visualize_symbol_2d(
    ...     H,
    ...     x_range=(-3, 3), y_range=(-3, 3),
    ...     xi_range=(-3, 3), eta_range=(-3, 3),
    ...     geodesics_params=[
    ...         (1, 0, 0, 1, 2*np.pi, 'red'),
    ...         (0, 1, 1, 0, 2*np.pi, 'blue'),
    ...         (0.5, 0.5, 0.5, -0.5, 4*np.pi, 'green')
    ...     ],
    ...     E_range=(0.5, 5),
    ...     hbar=0.1
    ... )
    >>> plt.show()
    """
    # Define default symbols if needed
    if x_sym is None:
        x_sym = sp.symbols('x', real=True)
    if y_sym is None:
        y_sym = sp.symbols('y', real=True)
    if xi_sym is None:
        xi_sym = sp.symbols('xi', real=True)
    if eta_sym is None:
        eta_sym = sp.symbols('eta', real=True)
    # Initialize geometry
    geometry = SymbolGeometry2D(symbol, x_sym, y_sym, xi_sym, eta_sym, hbar)
    # Create visualizer
    visualizer = SymbolVisualizer2D(geometry)
    # Generate visualization
    results = visualizer.visualize_complete(
        x_range=x_range,
        y_range=y_range,
        xi_range=xi_range,
        eta_range=eta_range,
        geodesics_params=geodesics_params,
        E_range=E_range,
        hbar=hbar,
        resolution=resolution
    )
    return results

# ============================================================================
# ADDITIONAL UTILITIES
# ============================================================================
class Utilities2D:
    """Additional analysis tools for 2D systems"""
    @staticmethod
    def compute_winding_number(geo: Geodesic2D) -> float:
        """
        Compute winding number around origin
        """
        angles = np.arctan2(geo.y, geo.x)
        angles_unwrapped = np.unwrap(angles)
        winding = (angles_unwrapped[-1] - angles_unwrapped[0]) / (2 * np.pi)
        return winding

    @staticmethod
    def compute_rotation_numbers(geo: Geodesic2D) -> Tuple[float, float]:
        """
        Compute rotation numbers (ω_x, ω_y)
        """
        theta_x = np.arctan2(geo.xi, geo.x)
        theta_y = np.arctan2(geo.eta, geo.y)
        theta_x = np.unwrap(theta_x)
        theta_y = np.unwrap(theta_y)
        omega_x = (theta_x[-1] - theta_x[0]) / (geo.t[-1] - geo.t[0])
        omega_y = (theta_y[-1] - theta_y[0]) / (geo.t[-1] - geo.t[0])
        return omega_x / (2*np.pi), omega_y / (2*np.pi)
    
    @staticmethod
    def detect_kam_tori(periodic_orbits: List[PeriodicOrbit2D],
                       tolerance: float = 0.1) -> Dict:
        """
        Detect KAM tori from periodic orbits
        """
        if not periodic_orbits:
            return {'n_tori': 0, 'tori': []}
        actions = np.array([orb.action for orb in periodic_orbits])
        # Cluster by action
        if len(actions) > 1:
            Z = linkage(actions.reshape(-1, 1), method='ward')
            clusters = fcluster(Z, t=tolerance, criterion='distance')
            n_tori = len(np.unique(clusters))
        else:
            n_tori = 1
            clusters = [1]
        # Analyze each torus
        tori = []
        for torus_id in np.unique(clusters):
            orbits_in_torus = [orb for i, orb in enumerate(periodic_orbits) 
                              if clusters[i] == torus_id]
            mean_action = np.mean([orb.action for orb in orbits_in_torus])
            mean_energy = np.mean([orb.energy for orb in orbits_in_torus])
            mean_period = np.mean([orb.period for orb in orbits_in_torus])
            stabilities = [orb.stability_1 for orb in orbits_in_torus]
            is_stable = np.mean(stabilities) < 0
            tori.append({
                'id': int(torus_id),
                'n_orbits': len(orbits_in_torus),
                'action': mean_action,
                'energy': mean_energy,
                'period': mean_period,
                'stable': is_stable
            })
        return {
            'n_tori': n_tori,
            'tori': tori
        }

# ============================================================================
# THEORETICAL DOCUMENTATION
# ============================================================================
def print_theory_summary():
    """Print a theoretical summary of key concepts"""
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║        GEOMETRIC AND SEMI-CLASSICAL THEORY FOR 2D SYSTEMS            ║
╚══════════════════════════════════════════════════════════════════════╝
PHASE SPACE AND SYMPLECTIC GEOMETRY
────────────────────────────────────
- Phase space: T*ℝ² ≅ ℝ⁴ with coordinates (x,y,ξ,η)
- Symplectic form: ω = dx∧dξ + dy∧dη
- Hamiltonian flow: dx/dt = ∂H/∂ξ, dy/dt = ∂H/∂η, dξ/dt = -∂H/∂x, dη/dt = -∂H/∂y
- Lagrangian manifolds: maximal submanifolds where ω = 0
CAUSTICS AND CATASTROPHE THEORY
───────────────────────────────
- Caustics: envelopes of trajectories where Jacobian vanishes
- Arnold classification:
  * Fold: generic singularity, Maslov index μ=1
  * Cusp: higher-order singularity, Maslov index μ=2
- Full 4×4 Jacobian for rigorous detection
- Phase shift upon crossing: -μπ/2
KAM THEORY AND INTEGRABLE SYSTEMS
────────────────────────────────
- Integrable systems: two independent first integrals (H, L)
- Phase space foliated by 2D tori
- KAM theorem: persistence of tori under small perturbations
- Rotation numbers (ω₁, ω₂): frequencies on torus
- EBK quantization: S_i = 2πℏ(n_i + α_i) with Maslov corrections
SEMI-CLASSICAL QUANTUM ANALYSIS
──────────────────────────────
- Weyl law: N(E) ~ Vol({H≤E}) / (2πℏ)²
- Spectral expansion: N(E) = N_smooth(E) + N_osc(E)
- Oscillatory terms: contributions from periodic orbits and caustics
- Level spacing distribution:
  * Poisson e^(-s): integrable systems
  * Wigner (πs/2)e^(-πs²/4): chaotic systems
PRACTICAL APPLICATIONS
──────────────────────
1. Geometrical optics and light caustics
2. Semi-classical quantum mechanics
3. Quantum chaos theory
4. Spectral analysis of pseudo-differential operators
5. Visualization of classical-quantum transition
USAGE
─────
>>> from symbol_visualizer_2d import visualize_symbol_2d, print_theory_summary
>>> x, y = sp.symbols('x y', real=True)
>>> xi, eta = sp.symbols('xi eta', real=True)
>>> H = xi**2 + eta**2 + x**2*y**2 + 0.1*(x**2 - 1)**2  # Complex example
>>> fig, geos, orbits, caustics = visualize_symbol_2d(
...     symbol=H,
...     x_range=(-2, 2), y_range=(-2, 2),
...     xi_range=(-2, 2), eta_range=(-2, 2),
...     geodesics_params=[
...         (0.5, 0.5, 0.5, 0.5, 10, 'red'),
...         (-0.5, 0.5, -0.5, 0.5, 10, 'blue'),
...         (0.0, 1.0, 1.0, 0.0, 15, 'green')
...     ],
...     E_range=(0.1, 5),
...     hbar=0.05,
...     resolution=60
... )
>>> plt.show()
""")