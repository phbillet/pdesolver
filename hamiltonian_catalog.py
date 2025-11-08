"""
hamiltonian_catalog.py
────────────────────────────────
Extended catalog of 1D and 2D Hamiltonians for pseudodifferential
and semiclassical analysis, visualization, and symbolic exploration.

Each entry:
    key : {
        "expr"        : sympy.Expr,
        "dim"         : 1 or 2,
        "category"    : str,
        "description" : str
    }

Usage:
    from hamiltonian_catalog_extended import get_hamiltonian
    H, vars, meta = get_hamiltonian("henon_heiles")
"""

import sympy as sp
from collections import Counter
import itertools
import os
import json

# ---------------------------------------------------------------------
# SYMBOLS (shared)
# ---------------------------------------------------------------------
# Real variables (real=True)
x, y, xi, eta = sp.symbols("x y xi eta", real=True)
mu1, mu2, sigma1, sigma2, zeta, z = sp.symbols("mu1 mu2 sigma1 sigma2 zeta z", real=True)
T = sp.symbols("T", real=True)

# Real and positive variables(real=True, positive=True)
m, k, alpha, beta, gamma, delta, omega, B, g, eps, A, V0, lambda_param, theta = sp.symbols(
    "m k alpha beta gamma delta omega B g eps A V0 lambda theta", real=True, positive=True
)
R, Delta, mu, golden_ratio, A_param, B_param, C_param, f_param, g_param, L_z = sp.symbols(
    "R Delta mu golden_ratio A_param B_param C_param f_param g_param L_z", real=True, positive=True
)

# ========================================================== #
# N.B. : This list of Hamiltonians was generated using LLMs. #
# ========================================================== #

# =====================================================================
# 1. INTEGRABLE / POLYNOMIAL SYSTEMS
# =====================================================================
H_INTEGRABLE = {
    "free_particle": {
        "expr": xi**2 / (2*m),
        "dim": 1,
        "category": "integrable",
        "description": "Free particle — straight trajectories, trivial flow.",
    },
    "harmonic_oscillator": {
        "expr": xi**2/(2*m) + k*x**2/2,
        "dim": 1,
        "category": "integrable",
        "description": "1D harmonic oscillator, closed circular trajectories.",
    },
    "anharmonic_oscillator": {
        "expr": xi**2/(2*m) + alpha*x**4/4,
        "dim": 1,
        "category": "integrable",
        "description": "Quartic oscillator — stiffer potential, still integrable.",
    },
    "double_well": {
        "expr": xi**2/(2*m) + alpha*(x**2 - 1)**2,
        "dim": 1,
        "category": "integrable",
        "description": "Symmetric double-well with two minima — tunneling prototype.",
    },
    "kepler": {
        "expr": (xi**2 + eta**2)/(2*m) - k/sp.sqrt(x**2 + y**2 + eps),
        "dim": 2,
        "category": "integrable",
        "description": "Kepler problem: inverse-square central potential (elliptic orbits).",
    },
    "isotropic_oscillator": {
        "expr": (xi**2 + eta**2)/(2*m) + k*(x**2 + y**2)/2,
        "dim": 2,
        "category": "integrable",
        "description": "2D isotropic oscillator — circular orbits, conserved angular momentum.",
    },
    "anisotropic_oscillator": {
        "expr": (xi**2 + eta**2)/(2*m) + 0.5*(x**2 + 2*y**2),
        "dim": 2,
        "category": "integrable",
        "description": "Anisotropic oscillator with rational frequency ratio — Lissajous figures.",
    },
    "mexican_hat": {
        "expr": (xi**2 + eta**2)/2 + (x**2 + y**2 - 1)**2,
        "dim": 2,
        "category": "integrable",
        "description": "Mexican-hat potential — ring of stable equilibria.",
    },
    "toda_pair": {
        "expr": xi**2/2 + alpha*sp.exp(-(x - y)),
        "dim": 2,
        "category": "integrable",
        "description": "Two-particle Toda lattice — exponential repulsion.",
    },
    "calogero_moser": {
        "expr": (xi**2 + eta**2)/2 + g/((x - y)**2 + eps),
        "dim": 2,
        "category": "integrable",
        "description": "Calogero–Moser model with inverse-square interaction.",
    },
    "sextic_oscillator": {
        "expr": xi**2/(2*m) + alpha*x**6/6,
        "dim": 1,
        "category": "integrable",
        "description": "Sextic potential — higher-order polynomial confinement.",
    },
    "linear_potential": {
        "expr": xi**2/(2*m) + g*x,
        "dim": 1,
        "category": "integrable",
        "description": "Particle in constant force field (gravity).",
    },
    "cubic_potential": {
        "expr": xi**2/(2*m) + alpha*x**3/3,
        "dim": 1,
        "category": "integrable",
        "description": "Cubic potential — asymmetric unbounded system.",
    },
    "quartic_2d": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha*(x**4 + y**4)/4,
        "dim": 2,
        "category": "integrable",
        "description": "Separable 2D quartic oscillator.",
    },
    "radial_power": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha*(x**2 + y**2)**2,
        "dim": 2,
        "category": "integrable",
        "description": "Radially symmetric quartic potential.",
    },
}

# =====================================================================
# 2. NONLINEAR & CHAOTIC SYSTEMS
# =====================================================================
H_CHAOTIC = {
    "henon_heiles": {
        "expr": (xi**2 + eta**2)/2 + (x**2 + y**2)/2 + alpha*(x**2*y - y**3/3),
        "dim": 2,
        "category": "chaotic",
        "description": "Hénon–Heiles: benchmark for mixed regular/chaotic motion.",
    },
    "quartic_coupled": {
        "expr": (xi**2 + eta**2)/2 + 0.25*(x**4 + y**4 + alpha*x**2*y**2),
        "dim": 2,
        "category": "chaotic",
        "description": "Quartic coupled oscillator — chaotic for large coupling α.",
    },
    "double_pendulum_reduced": {
        "expr": (xi**2 + eta**2 + xi*eta*sp.cos(x-y))/(2*(1 + sp.sin(x-y)**2)) + (sp.cos(x) + sp.cos(y)),
        "dim": 2,
        "category": "chaotic",
        "description": "Reduced double pendulum — strongly nonlinear and chaotic.",
    },
    "duffing": {
        "expr": xi**2/2 + 0.5*x**2 + 0.25*beta*x**4,
        "dim": 1,
        "category": "nonlinear",
        "description": "Duffing oscillator — bistable potential, nonlinear dynamics.",
    },
    "driven_pendulum": {
        "expr": xi**2/2 + (1 - sp.cos(x)) + alpha*x*sp.cos(omega),
        "dim": 1,
        "category": "chaotic",
        "description": "Driven pendulum — time-dependent forcing, chaotic response.",
    },
    "standard_map_like": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.cos(x)*sp.cos(y),
        "dim": 2,
        "category": "chaotic",
        "description": "Continuous analogue of the standard map — separatrix chaos.",
    },
    "quartic_mixed": {
        "expr": xi**2/2 + 0.25*x**4 - x,
        "dim": 1,
        "category": "nonlinear",
        "description": "Asymmetric quartic potential — metastability and bifurcation.",
    },
    "hill_potential": {
        "expr": xi**2/2 - alpha*x**2 + beta*x**4,
        "dim": 1,
        "category": "nonlinear",
        "description": "Hill potential — used in celestial and accelerator dynamics.",
    },
    "henon_heiles_variant": {
        "expr": (xi**2 + eta**2)/2 + 0.5*(x**2 + y**2) + alpha*x**2*y - beta*y**3,
        "dim": 2,
        "category": "chaotic",
        "description": "Modified Hénon–Heiles with adjustable nonlinearity.",
    },
    "yang_mills_reduced": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 - y**2)**2,
        "dim": 2,
        "category": "chaotic",
        "description": "Reduced Yang–Mills model — gauge theory analog.",
    },
    "stadium_billiard_smooth": {
        "expr": (xi**2 + eta**2)/2 + V0/(1 + sp.exp(-alpha*(sp.sqrt(x**2 + y**2) - 1))),
        "dim": 2,
        "category": "chaotic",
        "description": "Smoothed stadium billiard potential — chaotic scattering.",
    },
    "sinai_billiard_smooth": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.exp(-alpha*(x**2 + y**2)),
        "dim": 2,
        "category": "chaotic",
        "description": "Smooth approximation to Sinai billiard with circular scatterer.",
    },
    "poincare_surface": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "chaotic",
        "description": "Poincaré surface of section model — periodic modulation.",
    },
    "coupled_morse": {
        "expr": xi**2/2 + eta**2/2 + alpha*(sp.exp(-2*x) - 2*sp.exp(-x)) + beta*(sp.exp(-2*y) - 2*sp.exp(-y)) + gamma*x*y,
        "dim": 2,
        "category": "chaotic",
        "description": "Coupled Morse oscillators — molecular vibrational chaos.",
    },
    "van_der_pol": {
        "expr": xi**2/2 + y**2/2 + alpha*(x**2 - 1)*x*xi,
        "dim": 2,
        "category": "nonlinear",
        "description": "Van der Pol oscillator — limit cycle dynamics.",
    },
}

# =====================================================================
# 3. MAGNETIC & ROTATING SYSTEMS
# =====================================================================
H_MAGNETIC = {
    "landau_levels": {
        "expr": ((xi - B*y/2)**2 + (eta + B*x/2)**2)/(2*m),
        "dim": 2,
        "category": "magnetic",
        "description": "Charged particle in uniform B field — Landau quantization.",
    },
    "fock_darwin": {
        "expr": ((xi - B*y/2)**2 + (eta + B*x/2)**2)/(2*m) + 0.5*k*(x**2 + y**2),
        "dim": 2,
        "category": "magnetic",
        "description": "Oscillator + magnetic field — Fock–Darwin states.",
    },
    "coriolis": {
        "expr": 0.5*(xi**2 + eta**2) - omega*(x*eta - y*xi),
        "dim": 2,
        "category": "rotating",
        "description": "Coriolis Hamiltonian — dynamics in a rotating frame.",
    },
    "charged_potential": {
        "expr": ((xi - A*y)**2 + (eta + A*x)**2)/2 + alpha*(x**2 + y**2),
        "dim": 2,
        "category": "magnetic",
        "description": "Generic magnetic oscillator with vector potential.",
    },
    "aharonov_bohm": {
        "expr": (xi**2 + eta**2)/(2*m) + (A/(x**2 + y**2 + eps))**2,
        "dim": 2,
        "category": "magnetic",
        "description": "Aharonov–Bohm effect — topological phase from magnetic flux.",
    },
    "hall_effect": {
        "expr": ((xi - B*y)**2 + eta**2)/(2*m) + V0*x,
        "dim": 2,
        "category": "magnetic",
        "description": "Hall effect geometry — drift in crossed E and B fields.",
    },
    "cyclotron_resonance": {
        "expr": ((xi - omega*y)**2 + (eta + omega*x)**2)/(2*m) + alpha*sp.cos(omega),
        "dim": 2,
        "category": "magnetic",
        "description": "Cyclotron resonance with time-periodic drive.",
    },
    "penning_trap": {
        "expr": ((xi - B*y/2)**2 + (eta + B*x/2)**2)/(2*m) + alpha*(x**2 + y**2 - 2*x**2),
        "dim": 2,
        "category": "magnetic",
        "description": "Penning trap — quadrupole electric + magnetic confinement.",
    },
    "magnetic_bottle": {
        "expr": (xi**2 + eta**2)/(2*m) + B*(1 + alpha*x**2)*(x**2 + y**2)/2,
        "dim": 2,
        "category": "magnetic",
        "description": "Magnetic bottle trap — inhomogeneous field confinement.",
    },
}

# =====================================================================
# 4. OPTICAL / REFRACTIVE SYSTEMS
# =====================================================================
H_OPTICAL = {
    "graded_index": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*(x**2 + y**2))),
        "dim": 2,
        "category": "optical",
        "description": "Gradient-index fiber — geodesics bend toward the axis.",
    },
    "photonic_crystal": {
        "expr": (xi**2 + eta**2)/2 + alpha*(sp.cos(x) + sp.cos(y)),
        "dim": 2,
        "category": "optical",
        "description": "Periodic photonic lattice — band-structure analog.",
    },
    "anisotropic_medium": {
        "expr": 0.5*((1+alpha*x**2)*xi**2 + (1+beta*y**2)*eta**2),
        "dim": 2,
        "category": "optical",
        "description": "Anisotropic refractive medium — direction-dependent propagation.",
    },
    "waveguide_bent": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2 + beta*(y - 0.2*x**2)**2,
        "dim": 2,
        "category": "optical",
        "description": "Bent optical waveguide — model for ray focusing.",
    },
    "kerr_nonlinearity": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2)**2,
        "dim": 2,
        "category": "optical",
        "description": "Kerr nonlinearity — self-focusing in optical media.",
    },
    "bragg_grating": {
        "expr": xi**2/(2*m) + V0*sp.cos(2*alpha*x),
        "dim": 1,
        "category": "optical",
        "description": "Bragg grating — periodic refractive index modulation.",
    },
    "fiber_coupler": {
        "expr": (xi**2 + eta**2)/2 + 0.5*(x**2 + y**2) + alpha*x*y,
        "dim": 2,
        "category": "optical",
        "description": "Optical fiber coupler — evanescent wave coupling.",
    },
    "soliton_potential": {
        "expr": xi**2/2 - V0/sp.cosh(alpha*x)**2,
        "dim": 1,
        "category": "optical",
        "description": "Soliton potential — nonlinear wave localization.",
    },
    "photonic_waveguide_array": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + alpha*sp.cos(x + y)),
        "dim": 2,
        "category": "optical",
        "description": "Coupled waveguide array with diagonal coupling.",
    },
}

# =====================================================================
# 5. RELATIVISTIC & SEMICLASSICAL
# =====================================================================
H_RELATIVISTIC = {
    "relativistic_free": {
        "expr": sp.sqrt(xi**2 + eta**2 + m**2),
        "dim": 2,
        "category": "relativistic",
        "description": "Relativistic free particle, energy–momentum relation.",
    },
    "klein_gordon": {
        "expr": sp.sqrt(xi**2 + m**2) + 0.5*k*x**2,
        "dim": 1,
        "category": "relativistic",
        "description": "Klein–Gordon Hamiltonian with harmonic confinement.",
    },
    "dirac_radial": {
        "expr": sp.sqrt(xi**2 + (alpha*x)**2 + m**2),
        "dim": 1,
        "category": "relativistic",
        "description": "1D Dirac-type dispersion with position-dependent mass term.",
    },
    "semi_classical": {
        "expr": xi**2/2 + alpha*x**4/4 + eps*xi**4,
        "dim": 1,
        "category": "semiclassical",
        "description": "Schrödinger-like with small semiclassical correction in ξ.",
    },
    "relativistic_oscillator": {
        "expr": sp.sqrt(xi**2 + eta**2 + m**2) + k*(x**2 + y**2)/2,
        "dim": 2,
        "category": "relativistic",
        "description": "Relativistic harmonic oscillator.",
    },
    "dirac_coulomb": {
        "expr": sp.sqrt(xi**2 + m**2) - alpha/sp.sqrt(x**2 + eps),
        "dim": 1,
        "category": "relativistic",
        "description": "Dirac equation with Coulomb potential.",
    },
    "relativistic_kepler": {
        "expr": sp.sqrt(xi**2 + eta**2 + m**2) - k/sp.sqrt(x**2 + y**2 + eps),
        "dim": 2,
        "category": "relativistic",
        "description": "Relativistic Kepler problem — perihelion precession.",
    },
}

# =====================================================================
# 6. ATOMIC / MOLECULAR POTENTIALS
# =====================================================================
H_POTENTIALS = {
    "morse": {
        "expr": xi**2/(2*m) + alpha*(sp.exp(-2*x) - 2*sp.exp(-x)),
        "dim": 1,
        "category": "atomic",
        "description": "Morse potential — bound vibrational states, dissociation limit.",
    },
    "yukawa": {
        "expr": xi**2/2 - g*sp.exp(-alpha*sp.sqrt(x**2 + y**2))/(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "molecular",
        "description": "Yukawa potential — screened Coulomb interaction.",
    },
    "lennard_jones": {
        "expr": xi**2/2 + 4*alpha*((beta/x)**12 - (beta/x)**6),
        "dim": 1,
        "category": "molecular",
        "description": "Lennard–Jones — molecular bonding and repulsion.",
    },
    "gaussian_barrier": {
        "expr": xi**2/2 + alpha*sp.exp(-x**2),
        "dim": 1,
        "category": "scattering",
        "description": "Gaussian barrier — simple tunneling benchmark.",
    },
    "coulomb_2d": {
        "expr": (xi**2 + eta**2)/2 - 1/sp.sqrt(x**2 + y**2 + eps),
        "dim": 2,
        "category": "atomic",
        "description": "2D Coulomb potential — hydrogen-like bound states.",
    },
    "poschl_teller": {
        "expr": xi**2/(2*m) - V0/sp.cosh(alpha*x)**2,
        "dim": 1,
        "category": "atomic",
        "description": "Pöschl–Teller potential — exactly solvable quantum well.",
    },
    "eckart_barrier": {
        "expr": xi**2/(2*m) + V0/(sp.cosh(alpha*x)**2),
        "dim": 1,
        "category": "scattering",
        "description": "Eckart barrier — tunneling and reflection coefficient.",
    },
    "rosen_morse": {
        "expr": xi**2/(2*m) - V0*sp.tanh(alpha*x) + V0,
        "dim": 1,
        "category": "atomic",
        "description": "Rosen–Morse potential — asymmetric molecular interaction.",
    },
    "woods_saxon": {
        "expr": xi**2/(2*m) - V0/(1 + sp.exp((x - beta)/alpha)),
        "dim": 1,
        "category": "nuclear",
        "description": "Woods–Saxon potential — nuclear mean field approximation.",
    },
    "hulthen": {
        "expr": xi**2/(2*m) - alpha*sp.exp(-x)/(1 - sp.exp(-x)),
        "dim": 1,
        "category": "atomic",
        "description": "Hulthén potential — screened Coulomb for atomic screening.",
    },
    "manning_rosen": {
        "expr": xi**2/(2*m) - V0/sp.sinh(alpha*x)**2,
        "dim": 1,
        "category": "molecular",
        "description": "Manning–Rosen potential — molecular bond model.",
    },
    "buckingham": {
        "expr": xi**2/(2*m) + alpha*sp.exp(-beta*x) - gamma/x**6,
        "dim": 1,
        "category": "molecular",
        "description": "Buckingham potential — exp-6 molecular interaction.",
    },
}

# =====================================================================
# 7. GEOMETRIC & FLUID-INSPIRED FLOWS
# =====================================================================
H_GEOMETRIC = {
    "geodesic_plane": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*(x**2 + y**2))),
        "dim": 2,
        "category": "geometric",
        "description": "Geodesic flow on curved metric g=(1+αr²) — defocusing curvature.",
    },
    "vortex_pair": {
        "expr": -sp.log(sp.sqrt((x-y)**2 + eps)),
        "dim": 2,
        "category": "fluid",
        "description": "Simplified vortex interaction energy in 2D Euler flow.",
    },
    "shallow_water": {
        "expr": xi**2/2 + g*x,
        "dim": 1,
        "category": "fluid",
        "description": "Reduced shallow-water Hamiltonian — slope-induced motion.",
    },
    "magnetic_geodesic": {
        "expr": ((xi - B*y)**2 + (eta + B*x)**2)/2,
        "dim": 2,
        "category": "geometric",
        "description": "Geodesic flow under magnetic field (twisted symplectic form).",
    },
    "schwarzschild_radial": {
        "expr": (1 - 2*m/x)*xi**2/2 + alpha**2/(2*x**2),
        "dim": 1,
        "category": "geometric",
        "description": "Schwarzschild radial geodesic — general relativity orbit.",
    },
    "hyperbolic_geodesic": {
        "expr": (xi**2 + eta**2)/(2*y**2),
        "dim": 2,
        "category": "geometric",
        "description": "Geodesic flow on hyperbolic plane (Poincaré half-plane).",
    },
    "point_vortex_3": {
        "expr": -sp.log(sp.sqrt(x**2 + y**2 + eps)) - sp.log(sp.sqrt((x-1)**2 + y**2 + eps)),
        "dim": 2,
        "category": "fluid",
        "description": "Three-vortex interaction in 2D ideal fluid.",
    },
    "rossby_wave": {
        "expr": (xi**2 + eta**2)/2 + beta*x*y,
        "dim": 2,
        "category": "fluid",
        "description": "Rossby wave Hamiltonian — atmospheric/oceanic dynamics.",
    },
}

# =====================================================================
# 8. QUANTUM & CONDENSED MATTER SYSTEMS
# =====================================================================
H_QUANTUM = {
    "harmonic_spin_orbit": {
        "expr": (xi**2 + eta**2)/(2*m) + k*(x**2 + y**2)/2 + alpha*(x*eta - y*xi),
        "dim": 2,
        "category": "quantum",
        "description": "Harmonic oscillator with spin-orbit coupling.",
    },
    "rashba_hamiltonian": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha*(x*eta - y*xi) + beta*(x**2 + y**2),
        "dim": 2,
        "category": "quantum",
        "description": "Rashba spin-orbit interaction in 2DEG.",
    },
    "jaynes_cummings": {
        "expr": omega*xi + alpha*(x*xi + x**2),
        "dim": 1,
        "category": "quantum",
        "description": "Jaynes–Cummings model — atom-cavity interaction.",
    },
    "hofstadter_butterfly": {
        "expr": (xi**2 + eta**2)/2 + alpha*(sp.cos(x) + sp.cos(y)) - B*(x*eta - y*xi),
        "dim": 2,
        "category": "quantum",
        "description": "Hofstadter model — fractal energy spectrum in magnetic field.",
    },
    "bose_hubbard_continuum": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y)) + alpha*(x**2 + y**2)**2,
        "dim": 2,
        "category": "quantum",
        "description": "Continuum limit of Bose–Hubbard model.",
    },
    "superconducting_pairing": {
        "expr": (xi**2 + eta**2)/(2*m) - delta*(x*y + xi*eta),
        "dim": 2,
        "category": "quantum",
        "description": "BCS pairing Hamiltonian — superconductivity.",
    },
    "gross_pitaevskii": {
        "expr": xi**2/(2*m) + V0*sp.sin(x)**2 + g*x**2,
        "dim": 1,
        "category": "quantum",
        "description": "Gross–Pitaevskii equation for BEC — mean field theory.",
    },
}

# =====================================================================
# 9. ASTROPHYSICAL & GRAVITATIONAL SYSTEMS
# =====================================================================
H_ASTROPHYSICS = {
    "schwarzschild_orbit": {
        "expr": (xi**2 + eta**2)/(2*m) - k/sp.sqrt(x**2 + y**2) + alpha/(x**2 + y**2),
        "dim": 2,
        "category": "astrophysics",
        "description": "Schwarzschild effective potential — GR corrections to orbits.",
    },
    "three_body_restricted": {
        "expr": (xi**2 + eta**2)/2 - 1/sp.sqrt((x+alpha)**2 + y**2 + eps) - alpha/sp.sqrt((x-1)**2 + y**2 + eps),
        "dim": 2,
        "category": "astrophysics",
        "description": "Restricted three-body problem — Lagrange points.",
    },
    "tidal_force": {
        "expr": xi**2/(2*m) + k*x**2/2 - alpha*x**3,
        "dim": 1,
        "category": "astrophysics",
        "description": "Tidal force approximation near massive body.",
    },
    "galactic_rotation": {
        "expr": (xi**2 + eta**2)/(2*m) - k/sp.sqrt(x**2 + y**2 + eps) + omega*(x*eta - y*xi),
        "dim": 2,
        "category": "astrophysics",
        "description": "Galactic disk rotation with dark matter halo.",
    },
    "planetary_ring": {
        "expr": (xi**2 + eta**2)/2 - 1/sp.sqrt(x**2 + y**2 + eps) + omega**2*(x**2 + y**2)/2,
        "dim": 2,
        "category": "astrophysics",
        "description": "Particle dynamics in planetary ring system.",
    },
}

# =====================================================================
# 10. LATTICE & PERIODIC SYSTEMS
# =====================================================================
H_LATTICE = {
    "kronig_penney": {
        "expr": xi**2/(2*m) + V0*sum([sp.DiracDelta(x - n) for n in range(-3, 4)]),
        "dim": 1,
        "category": "lattice",
        "description": "Kronig–Penney model — periodic delta potentials.",
    },
    "mathieu": {
        "expr": xi**2/(2*m) + V0*sp.cos(2*x),
        "dim": 1,
        "category": "lattice",
        "description": "Mathieu equation — parametric resonance in periodic systems.",
    },
    "tight_binding": {
        "expr": (xi**2 + eta**2)/2 + 2*V0*(sp.cos(x) + sp.cos(y)),
        "dim": 2,
        "category": "lattice",
        "description": "Tight-binding approximation on square lattice.",
    },
    "harper_model": {
        "expr": xi**2/2 + 2*V0*sp.cos(x + alpha*y),
        "dim": 2,
        "category": "lattice",
        "description": "Harper model — quasiperiodic potential, fractal spectrum.",
    },
    "aubry_andre": {
        "expr": xi**2/(2*m) + V0*sp.cos(2*sp.pi*alpha*x),
        "dim": 1,
        "category": "lattice",
        "description": "Aubry–André model — metal-insulator transition.",
    },
    "wannier_stark": {
        "expr": xi**2/(2*m) + V0*sp.cos(x) + g*x,
        "dim": 1,
        "category": "lattice",
        "description": "Wannier–Stark ladder — Bloch oscillations in tilted lattice.",
    },
    "kagome_lattice": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + sp.cos(x-y)),
        "dim": 2,
        "category": "lattice",
        "description": "Kagome lattice geometry — frustrated magnetic systems.",
    },
    "hexagonal_lattice": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + sp.cos(x+y)),
        "dim": 2,
        "category": "lattice",
        "description": "Hexagonal (graphene-like) lattice structure.",
    },
}

# =====================================================================
# 11. STOCHASTIC & DISSIPATIVE SYSTEMS
# =====================================================================
H_DISSIPATIVE = {
    "damped_oscillator": {
        "expr": xi**2/(2*m) + k*x**2/2 + gamma*x*xi,
        "dim": 1,
        "category": "dissipative",
        "description": "Damped harmonic oscillator — energy dissipation.",
    },
    "caldirola_kanai": {
        "expr": sp.exp(-gamma)*xi**2/(2*m) + sp.exp(gamma)*k*x**2/2,
        "dim": 1,
        "category": "dissipative",
        "description": "Caldirola–Kanai Hamiltonian — time-dependent damping.",
    },
    "rayleigh_oscillator": {
        "expr": xi**2/2 + x**2/2 + alpha*xi*(xi**2 - 1),
        "dim": 1,
        "category": "dissipative",
        "description": "Rayleigh oscillator — nonlinear damping model.",
    },
    "fokker_planck": {
        "expr": xi**2/2 - gamma*sp.log(1 + x**2),
        "dim": 1,
        "category": "dissipative",
        "description": "Fokker–Planck effective Hamiltonian.",
    },
}

# =====================================================================
# 12. BIOPHYSICAL & CHEMICAL SYSTEMS
# =====================================================================
H_BIOPHYSICS = {
    "protein_folding": {
        "expr": xi**2/(2*m) + alpha*(1 - sp.cos(x))**2 + beta*(1 - sp.cos(y))**2 + gamma*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "biophysics",
        "description": "Simplified protein dihedral angle dynamics.",
    },
    "dna_twist": {
        "expr": xi**2/(2*m) + k*x**2/2 + alpha*sp.cos(beta*x),
        "dim": 1,
        "category": "biophysics",
        "description": "DNA torsional dynamics — supercoiling model.",
    },
    "michaelis_menten": {
        "expr": xi**2/2 + V0*x/(k + x),
        "dim": 1,
        "category": "biophysics",
        "description": "Michaelis–Menten enzyme kinetics effective potential.",
    },
    "hodgkin_huxley_reduced": {
        "expr": xi**2/2 + alpha*x**3 - beta*x,
        "dim": 1,
        "category": "biophysics",
        "description": "Reduced Hodgkin–Huxley — neural spike dynamics.",
    },
    "fitzhugh_nagumo": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**3/3 - x) + y,
        "dim": 2,
        "category": "biophysics",
        "description": "FitzHugh–Nagumo model — excitable media.",
    },
    "chemotaxis": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.log(1 + x**2 + y**2),
        "dim": 2,
        "category": "biophysics",
        "description": "Chemotactic cell migration — logarithmic attraction.",
    },
}

# =====================================================================
# 13. PLASMA & ELECTROMAGNETIC SYSTEMS
# =====================================================================
H_PLASMA = {
    "plasma_wave": {
        "expr": (xi**2 + eta**2)/2 + omega**2*(x**2 + y**2)/2 + alpha*x*y,
        "dim": 2,
        "category": "plasma",
        "description": "Plasma oscillation with wave coupling.",
    },
    "vlasov_poisson": {
        "expr": xi**2/2 + alpha*sp.sin(x),
        "dim": 1,
        "category": "plasma",
        "description": "Vlasov–Poisson system — plasma collective effects.",
    },
    "debye_shielding": {
        "expr": xi**2/2 - g*sp.exp(-alpha*sp.sqrt(x**2 + eps))/(sp.sqrt(x**2 + eps)),
        "dim": 1,
        "category": "plasma",
        "description": "Debye-screened Coulomb potential in plasma.",
    },
    "tokamak_particle": {
        "expr": ((xi - B*y)**2 + eta**2)/2 + V0*(1 - sp.cos(x)),
        "dim": 2,
        "category": "plasma",
        "description": "Charged particle in tokamak — toroidal confinement.",
    },
    "cyclotron_maser": {
        "expr": ((xi - omega*y)**2 + (eta + omega*x)**2)/2 + alpha*sp.cos(x),
        "dim": 2,
        "category": "plasma",
        "description": "Cyclotron maser instability — wave-particle resonance.",
    },
}

# =====================================================================
# 14. ACCELERATOR & BEAM PHYSICS
# =====================================================================
H_ACCELERATOR = {
    "rf_cavity": {
        "expr": xi**2/(2*m) + V0*sp.sin(omega*x),
        "dim": 1,
        "category": "accelerator",
        "description": "RF cavity acceleration — synchrotron motion.",
    },
    "betatron_oscillation": {
        "expr": xi**2/(2*m) + k*(1 + alpha*sp.cos(y))*x**2/2,
        "dim": 2,
        "category": "accelerator",
        "description": "Betatron oscillations in circular accelerator.",
    },
    "synchrotron_radiation": {
        "expr": xi**2/(2*m) - gamma*xi + k*x**2/2,
        "dim": 1,
        "category": "accelerator",
        "description": "Energy loss from synchrotron radiation.",
    },
    "space_charge": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha*sp.log(x**2 + y**2 + eps),
        "dim": 2,
        "category": "accelerator",
        "description": "Space charge effects in particle beams.",
    },
    "chromaticity": {
        "expr": xi**2/(2*m) + k*(1 + alpha*xi)*x**2/2,
        "dim": 1,
        "category": "accelerator",
        "description": "Chromatic aberration in beam optics.",
    },
}

# =====================================================================
# 15. EXOTIC & ADVANCED SYSTEMS
# =====================================================================
H_EXOTIC = {
    "fractal_potential": {
        "expr": xi**2/2 + V0*sp.sin(x)*sp.sin(alpha*x)*sp.sin(alpha**2*x),
        "dim": 1,
        "category": "exotic",
        "description": "Multi-scale fractal potential — self-similar structure.",
    },
    "random_matrix": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 - y**2) + beta*x*y,
        "dim": 2,
        "category": "exotic",
        "description": "Random matrix ensemble Hamiltonian.",
    },
    "supersymmetric_qm": {
        "expr": xi**2/2 + (sp.diff(V0*sp.tanh(x), x))**2 - sp.diff(sp.diff(V0*sp.tanh(x), x), x),
        "dim": 1,
        "category": "exotic",
        "description": "Supersymmetric quantum mechanics partner potential.",
    },
    "pt_symmetric": {
        "expr": xi**2/(2*m) + sp.I*V0*x**3,
        "dim": 1,
        "category": "exotic",
        "description": "PT-symmetric (non-Hermitian) Hamiltonian.",
    },
    "anyonic_oscillator": {
        "expr": (xi**2 + eta**2)/2 + k*(x**2 + y**2)/2 + alpha*(x*eta - y*xi)**2,
        "dim": 2,
        "category": "exotic",
        "description": "Anyon oscillator — fractional statistics.",
    },
    "noncommutative_space": {
        "expr": (xi**2 + eta**2)/2 + k*(x**2 + y**2)/2 + theta*(x*eta - y*xi),
        "dim": 2,
        "category": "exotic",
        "description": "Noncommutative geometry — quantum space structure.",
    },
    "q_deformed": {
        "expr": (sp.exp(xi) - sp.exp(-xi))/(2*sp.sinh(alpha)) + k*x**2/2,
        "dim": 1,
        "category": "exotic",
        "description": "q-deformed oscillator — quantum group symmetry.",
    },
    "levy_flight": {
        "expr": sp.Abs(xi)**alpha + V0*x**2/2,
        "dim": 1,
        "category": "exotic",
        "description": "Lévy flight dynamics — superdiffusion anomalous transport.",
    },
}

# =====================================================================
# 16. ADDITIONAL CLASSICAL MECHANICS
# =====================================================================
H_CLASSICAL_EXTENDED = {
    "spherical_pendulum": {
        "expr": (xi**2 + eta**2)/(2*m) + m*g*(1 - sp.cos(x))*sp.sin(y)**2,
        "dim": 2,
        "category": "classical",
        "description": "Spherical pendulum — 3D pendulum motion.",
    },
    "spinning_top": {
        "expr": (xi**2 + eta**2)/(2*m) + omega*(x*eta - y*xi) + m*g*x,
        "dim": 2,
        "category": "classical",
        "description": "Spinning top (simplified) — precession dynamics.",
    },
    "wilberforce_spring": {
        "expr": xi**2/(2*m) + eta**2/(2*m) + k*x**2/2 + k*y**2/2 + alpha*x*y,
        "dim": 2,
        "category": "classical",
        "description": "Wilberforce pendulum — coupled translation-rotation.",
    },
    "atwood_machine": {
        "expr": xi**2/(2*m) - m*g*x,
        "dim": 1,
        "category": "classical",
        "description": "Atwood machine — constrained pulley system.",
    },
    "coupled_pendula": {
        "expr": (xi**2 + eta**2)/(2*m) + g*(sp.cos(x) + sp.cos(y)) + k*(x - y)**2/2,
        "dim": 2,
        "category": "classical",
        "description": "Two coupled pendula — normal mode analysis.",
    },
    "roller_coaster": {
        "expr": xi**2/(2*m) + m*g*sp.sin(alpha*x),
        "dim": 1,
        "category": "classical",
        "description": "Roller coaster dynamics — gravity on curved track.",
    },
    "brachistochrone": {
        "expr": sp.sqrt(1 + sp.diff(y, x)**2)*sp.sqrt(2*g*y),
        "dim": 1,
        "category": "classical",
        "description": "Brachistochrone problem — fastest descent curve.",
    },
}

# =====================================================================
# 17. TOPOLOGICAL & GAUGE SYSTEMS
# =====================================================================
H_TOPOLOGICAL = {
    "chern_insulator": {
        "expr": (xi**2 + eta**2)/2 + alpha*(sp.cos(x) + sp.cos(y)) + beta*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "topological",
        "description": "Chern insulator — topological band structure.",
    },
    "haldane_model": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + sp.cos(x+y)) + alpha*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "topological",
        "description": "Haldane model — quantum Hall effect without Landau levels.",
    },
    "su_schrieffer_heeger": {
        "expr": xi**2/(2*m) + V0*(sp.cos(x) + alpha*sp.cos(x/2)),
        "dim": 1,
        "category": "topological",
        "description": "SSH model — topological edge states.",
    },
    "berry_phase": {
        "expr": (xi**2 + eta**2)/2 + omega*(x*eta - y*xi) + V0*sp.cos(sp.atan2(y, x)),
        "dim": 2,
        "category": "topological",
        "description": "System with geometric Berry phase.",
    },
    "monopole_field": {
        "expr": (xi**2 + eta**2)/(2*m) + g*sp.atan2(y, x),
        "dim": 2,
        "category": "topological",
        "description": "Dirac magnetic monopole — topological magnetic charge.",
    },
}

# =====================================================================
# 18. NONLINEAR OPTICS & SOLITONS
# =====================================================================
H_NONLINEAR_OPTICS = {
    "nls_cubic": {
        "expr": xi**2/(2*m) + alpha*x**4,
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "Cubic nonlinear Schrödinger equation — bright solitons.",
    },
    "nls_quintic": {
        "expr": xi**2/(2*m) + alpha*x**4 - beta*x**6,
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "Quintic NLS — soliton stability and collapse.",
    },
    "derivative_nls": {
        "expr": xi**2/(2*m) + alpha*x**2*xi,
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "Derivative NLS — Alfvén waves, plasma physics.",
    },
    "davey_stewartson": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2)**2 + beta*x*y,
        "dim": 2,
        "category": "nonlinear_optics",
        "description": "Davey–Stewartson equation — 2D soliton interactions.",
    },
    "sine_gordon": {
        "expr": xi**2/(2*m) + V0*(1 - sp.cos(x)),
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "Sine-Gordon equation — kinks and breathers.",
    },
    "phi4_theory": {
        "expr": xi**2/(2*m) + 0.5*m**2*x**2 + lambda_param*x**4/4,
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "φ⁴ field theory — domain walls in phase transitions.",
    },
    "ablowitz_ladik": {
        "expr": xi**2/2 + alpha*sp.sin(x)/(1 + beta*sp.cos(x)),
        "dim": 1,
        "category": "nonlinear_optics",
        "description": "Ablowitz–Ladik lattice — integrable discrete NLS.",
    },
    "manakov_system": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**4 + y**4 + 2*x**2*y**2),
        "dim": 2,
        "category": "nonlinear_optics",
        "description": "Manakov system — vector solitons, polarization coupling.",
    },
    "kadomtsev_petviashvili": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**3 + beta*x*y,
        "dim": 2,
        "category": "nonlinear_optics",
        "description": "KP equation — 2D shallow water waves.",
    },
}

# =====================================================================
# 19. SPIN SYSTEMS & MAGNETIC MODELS
# =====================================================================
H_SPIN_SYSTEMS = {
    "heisenberg_classical": {
        "expr": -alpha*(sp.cos(x-y)) - beta*(x**2 + y**2),
        "dim": 2,
        "category": "spin_systems",
        "description": "Classical Heisenberg model — spin exchange interaction.",
    },
    "ising_transverse": {
        "expr": -alpha*x - beta*sp.tanh(x),
        "dim": 1,
        "category": "spin_systems",
        "description": "Transverse field Ising model — quantum phase transition.",
    },
    "xy_model": {
        "expr": -alpha*sp.cos(x - y),
        "dim": 2,
        "category": "spin_systems",
        "description": "XY model — planar spins, Kosterlitz-Thouless transition.",
    },
    "dzyaloshinskii_moriya": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*eta - y*xi) - beta*sp.cos(x)*sp.cos(y),
        "dim": 2,
        "category": "spin_systems",
        "description": "Dzyaloshinskii–Moriya interaction — chiral magnetism.",
    },
    "landau_lifshitz": {
        "expr": -alpha*x*y - beta*(x**2 + y**2)/2,
        "dim": 2,
        "category": "spin_systems",
        "description": "Landau–Lifshitz equation — magnetization dynamics.",
    },
    "skyrmion": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - x**2 - y**2)**2 + beta*(x*eta - y*xi),
        "dim": 2,
        "category": "spin_systems",
        "description": "Skyrmion texture — topological magnetic soliton.",
    },
}

# =====================================================================
# 20. REACTION-DIFFUSION & PATTERN FORMATION
# =====================================================================
H_REACTION_DIFFUSION = {
    "fisher_kpp": {
        "expr": xi**2/(2*m) - alpha*x*(1 - x),
        "dim": 1,
        "category": "reaction_diffusion",
        "description": "Fisher–KPP equation — population dynamics, traveling waves.",
    },
    "allen_cahn": {
        "expr": xi**2/(2*m) + alpha*(x**2 - 1)**2,
        "dim": 1,
        "category": "reaction_diffusion",
        "description": "Allen–Cahn equation — phase separation dynamics.",
    },
    "cahn_hilliard": {
        "expr": xi**2/2 + alpha*(x**2 - 1)**2 - beta*xi**2,
        "dim": 1,
        "category": "reaction_diffusion",
        "description": "Cahn–Hilliard equation — spinodal decomposition.",
    },
    "brusselator": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2*y - beta*x,
        "dim": 2,
        "category": "reaction_diffusion",
        "description": "Brusselator model — chemical oscillations, Turing patterns.",
    },
    "schnakenberg": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2*y - beta*(x + y),
        "dim": 2,
        "category": "reaction_diffusion",
        "description": "Schnakenberg model — autocatalytic reactions.",
    },
    "gierer_meinhardt": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2/y - beta*x,
        "dim": 2,
        "category": "reaction_diffusion",
        "description": "Gierer–Meinhardt model — biological morphogenesis.",
    },
}

# =====================================================================
# 21. ELASTICITY & CONTINUUM MECHANICS
# =====================================================================
H_ELASTICITY = {
    "euler_bernoulli_beam": {
        "expr": xi**2/(2*m) + k*x**4/4,
        "dim": 1,
        "category": "elasticity",
        "description": "Euler–Bernoulli beam — bending energy (polynomial approximation).",
    },
    "timoshenko_beam": {
        "expr": xi**2/(2*m) + eta**2/(2*m) + k*x**2/2 + alpha*(x - y)**2,
        "dim": 2,
        "category": "elasticity",
        "description": "Timoshenko beam — coupled bending-shear modes.",
    },
    "kirchhoff_plate": {
        "expr": (xi**2 + eta**2)/2 + k*(x**2 + y**2)**2,
        "dim": 2,
        "category": "elasticity",
        "description": "Kirchhoff plate theory — thin plate bending (simplified).",
    },
    "neo_hookean": {
        "expr": xi**2/(2*m) + alpha*(x**2 + 1/x**2),
        "dim": 1,
        "category": "elasticity",
        "description": "Neo-Hookean material — nonlinear elasticity.",
    },
    "mooney_rivlin": {
        "expr": xi**2/(2*m) + alpha*(x**2 - 1) + beta*(1/x**2 - 1),
        "dim": 1,
        "category": "elasticity",
        "description": "Mooney–Rivlin model — rubber elasticity.",
    },
}


# =====================================================================
# 22. INFORMATION THEORY & STATISTICAL MECHANICS
# =====================================================================
H_STATISTICAL = {
    "maxwell_boltzmann": {
        "expr": xi**2/(2*m) - k*sp.log(1 + sp.exp(-x)),
        "dim": 1,
        "category": "statistical",
        "description": "Maxwell–Boltzmann distribution effective potential.",
    },
    "fermi_dirac": {
        "expr": xi**2/(2*m) - k*sp.log(1 + sp.exp(-x/k)),
        "dim": 1,
        "category": "statistical",
        "description": "Fermi–Dirac statistics — electron gas.",
    },
    "bose_einstein": {
        "expr": xi**2/(2*m) + k*sp.log(1 - sp.exp(-x/k)),
        "dim": 1,
        "category": "statistical",
        "description": "Bose–Einstein condensation effective Hamiltonian.",
    },
    "ising_mean_field": {
        "expr": -alpha*x**2 + beta*x**4,
        "dim": 1,
        "category": "statistical",
        "description": "Ising model mean-field free energy.",
    },
    "potts_model": {
        "expr": -alpha*sp.cos(2*sp.pi*x/3) - beta*sp.cos(4*sp.pi*x/3),
        "dim": 1,
        "category": "statistical",
        "description": "q-state Potts model — generalized Ising.",
    },
}

# =====================================================================
# 23. NEUROSCIENCE & NEURAL NETWORKS
# =====================================================================
H_NEUROSCIENCE = {
    "hopfield_network": {
        "expr": -(xi**2 + eta**2)/2 - alpha*x*y,
        "dim": 2,
        "category": "neuroscience",
        "description": "Hopfield network — associative memory energy.",
    },
    "wilson_cowan": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.tanh(x) - beta*sp.tanh(y),
        "dim": 2,
        "category": "neuroscience",
        "description": "Wilson–Cowan model — neural population dynamics.",
    },
    "izhikevich": {
        "expr": 0.5*xi**2 + 0.04*x**2 + 5*x - y,
        "dim": 2,
        "category": "neuroscience",
        "description": "Izhikevich neuron — efficient spiking model.",
    },
    "hindmarsh_rose": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**3 - beta*x - y,
        "dim": 2,
        "category": "neuroscience",
        "description": "Hindmarsh–Rose neuron — bursting behavior.",
    },
    "morris_lecar": {
        "expr": xi**2/2 + eta**2/2 + alpha*x*(x - beta)*(x - 1) - y,
        "dim": 2,
        "category": "neuroscience",
        "description": "Morris–Lecar model — barnacle muscle fiber.",
    },
}

# =====================================================================
# 24. ECONOPHYSICS & SOCIAL DYNAMICS
# =====================================================================
H_ECONOPHYSICS = {
    "black_scholes": {
        "expr": xi**2*x**2/(2*m) + alpha*x*xi,
        "dim": 1,
        "category": "econophysics",
        "description": "Black–Scholes as Hamiltonian — option pricing.",
    },
    "heston_model": {
        "expr": xi**2/2 + eta**2/2 + alpha*y*(x**2 - beta),
        "dim": 2,
        "category": "econophysics",
        "description": "Heston stochastic volatility model.",
    },
    "ising_market": {
        "expr": -alpha*sp.tanh(x)*sp.tanh(y),
        "dim": 2,
        "category": "econophysics",
        "description": "Ising-like market interaction — herding behavior.",
    },
    "voter_model": {
        "expr": -alpha*x*y,
        "dim": 2,
        "category": "econophysics",
        "description": "Voter model — opinion dynamics.",
    },
}

# =====================================================================
# 25. QUANTUM FIELD THEORY INSPIRED
# =====================================================================
H_QFT = {
    "schwinger_model": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2) + beta*x*y,
        "dim": 2,
        "category": "qft",
        "description": "Schwinger model — QED in 1+1 dimensions.",
    },
    "thirring_model": {
        "expr": xi**2/(2*m) + alpha*x**4,
        "dim": 1,
        "category": "qft",
        "description": "Thirring model — self-interacting fermions.",
    },
    "gross_neveu": {
        "expr": xi**2/(2*m) + alpha*x**2 + beta*x**4,
        "dim": 1,
        "category": "qft",
        "description": "Gross–Neveu model — asymptotic freedom.",
    },
    "coleman_weinberg": {
        "expr": xi**2/(2*m) + alpha*x**4*sp.log(x**2/beta),
        "dim": 1,
        "category": "qft",
        "description": "Coleman–Weinberg potential — radiative corrections.",
    },
    "higgs_mexican_hat": {
        "expr": (xi**2 + eta**2)/2 - alpha*(x**2 + y**2) + beta*(x**2 + y**2)**2,
        "dim": 2,
        "category": "qft",
        "description": "Higgs potential — spontaneous symmetry breaking.",
    },
}

# =====================================================================
# 26. ADDITIONAL EXOTIC & MATHEMATICAL
# =====================================================================
H_MATHEMATICAL = {
    "painleve_transcendent": {
        "expr": xi**2/2 + alpha*x**3 + beta*x,
        "dim": 1,
        "category": "mathematical",
        "description": "Painlevé transcendent — special function dynamics.",
    },
    "weierstrass": {
        "expr": xi**2/(2*m) + alpha*sp.elliptic_k(x),
        "dim": 1,
        "category": "mathematical",
        "description": "Weierstrass elliptic function potential.",
    },
#    "jacobi_elliptic": {
#        "expr": xi**2/(2*m) + alpha*sp.jacobi_sn(x, m)**2,
#        "dim": 1,
#        "category": "mathematical",
#        "description": "Jacobi elliptic function potential.",
#    },
    "hypergeometric": {
        "expr": xi**2/(2*m) + alpha*sp.hyper((1, 1), (2,), x),
        "dim": 1,
        "category": "mathematical",
        "description": "Hypergeometric function potential.",
    },
    "lambert_w": {
        "expr": xi**2/(2*m) + alpha*x*sp.exp(x),
        "dim": 1,
        "category": "mathematical",
        "description": "Lambert W function related potential.",
    },
    "zeta_potential": {
        "expr": xi**2/(2*m) + alpha/x**2,
        "dim": 1,
        "category": "mathematical",
        "description": "Riemann zeta related inverse square potential.",
    },
}

# =====================================================================
# 27. COSMOLOGY & EARLY UNIVERSE
# =====================================================================
H_COSMOLOGY = {
    "inflaton_chaotic": {
        "expr": xi**2/(2*m) + 0.5*m**2*x**2,
        "dim": 1,
        "category": "cosmology",
        "description": "Chaotic inflation — quadratic potential.",
    },
    "inflaton_starobinsky": {
        "expr": xi**2/(2*m) + alpha*(1 - sp.exp(-beta*x)),
        "dim": 1,
        "category": "cosmology",
        "description": "Starobinsky inflation — R² gravity.",
    },
    "quintessence": {
        "expr": xi**2/(2*m) + V0*sp.exp(-alpha*x),
        "dim": 1,
        "category": "cosmology",
        "description": "Quintessence dark energy — exponential potential.",
    },
    "ekpyrotic": {
        "expr": -xi**2/(2*m) + V0*sp.exp(alpha*x),
        "dim": 1,
        "category": "cosmology",
        "description": "Ekpyrotic universe — negative kinetic energy.",
    },
}

# =====================================================================
# 28. TURBULENCE & FLUID DYNAMICS AVANCÉE
# =====================================================================
H_TURBULENCE = {
    "kolmogorov_flow": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.sin(y),
        "dim": 2,
        "category": "turbulence",
        "description": "Kolmogorov flow — forced 2D turbulence.",
    },
    "rayleigh_benard": {
        "expr": (xi**2 + eta**2)/2 + alpha*x*y - beta*y**2,
        "dim": 2,
        "category": "turbulence",
        "description": "Rayleigh–Bénard convection — thermal instability.",
    },
    "taylor_couette": {
        "expr": (xi**2 + eta**2)/2 - omega*(x*eta - y*xi) + alpha*x**2,
        "dim": 2,
        "category": "turbulence",
        "description": "Taylor–Couette flow — rotating cylinders.",
    },
    "karman_vortex": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x)*sp.sinh(y)),
        "dim": 2,
        "category": "turbulence",
        "description": "von Kármán vortex street — cylinder wake.",
    },
    "burgers_potential": {
        "expr": xi**2/2 + alpha*x**3/3,
        "dim": 1,
        "category": "turbulence",
        "description": "Burgers equation potential — shock formation.",
    },
    "kdv_equation": {
        "expr": xi**2/2 + alpha*x**3,
        "dim": 1,
        "category": "turbulence",
        "description": "Korteweg–de Vries (simplified) — shallow water solitons.",
    },
    "navier_stokes_2d": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*y)**2,
        "dim": 2,
        "category": "turbulence",
        "description": "2D Navier–Stokes enstrophy (simplified).",
    },
    "hasegawa_mima": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.log(1 + x**2 + y**2),
        "dim": 2,
        "category": "turbulence",
        "description": "Hasegawa–Mima plasma turbulence.",
    },
}

# =====================================================================
# 29. GRANULAR MATTER & SOFT MATTER
# =====================================================================
H_GRANULAR = {
    "hertz_contact": {
        "expr": xi**2/(2*m) + alpha*sp.Abs(x)**(3/2),
        "dim": 1,
        "category": "granular",
        "description": "Hertzian contact — elastic collision of spheres.",
    },
    "durian_foam": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.Abs(sp.sqrt(x**2 + y**2) - 1)**(5/2),
        "dim": 2,
        "category": "granular",
        "description": "Durian foam model — soft sphere packing.",
    },
    "jamming_transition": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.Heaviside(1 - sp.sqrt(x**2 + y**2))*sp.Abs(1 - sp.sqrt(x**2 + y**2))**(3/2),
        "dim": 2,
        "category": "granular",
        "description": "Jamming transition — athermal soft spheres.",
    },
    "frenkel_kontorova": {
        "expr": xi**2/(2*m) + k*(x - sp.sin(x))**2/2,
        "dim": 1,
        "category": "granular",
        "description": "Frenkel–Kontorova model — dislocation dynamics.",
    },
    "stick_slip": {
        "expr": xi**2/(2*m) + k*x**2/2 - alpha*sp.sign(xi)*sp.exp(-beta*sp.Abs(xi)),
        "dim": 1,
        "category": "granular",
        "description": "Stick-slip friction — rate-and-state friction.",
    },
    "sandpile": {
        "expr": (xi**2 + eta**2)/2 + g*sp.sqrt(x**2 + y**2),
        "dim": 2,
        "category": "granular",
        "description": "Sandpile model — self-organized criticality.",
    },
}

# =====================================================================
# 30. ACTIVE MATTER & LIVING SYSTEMS
# =====================================================================
H_ACTIVE_MATTER = {
    "vicsek_model": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*xi + y*eta)/sp.sqrt(x**2 + y**2 + eps),
        "dim": 2,
        "category": "active_matter",
        "description": "Vicsek model — collective motion of self-propelled particles.",
    },
    "active_brownian": {
        "expr": (xi**2 + eta**2)/2 - V0*(x*sp.cos(sp.atan2(eta, xi)) + y*sp.sin(sp.atan2(eta, xi))),
        "dim": 2,
        "category": "active_matter",
        "description": "Active Brownian particle — self-propulsion.",
    },
    "toner_tu": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2) - beta*(x*xi + y*eta),
        "dim": 2,
        "category": "active_matter",
        "description": "Toner–Tu theory — flocking hydrodynamics.",
    },
    "bacterial_swarm": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.log(1 + x**2 + y**2) + beta*(x*xi + y*eta),
        "dim": 2,
        "category": "active_matter",
        "description": "Bacterial swarming — chemotaxis and active flow.",
    },
    "run_and_tumble": {
        "expr": xi**2/(2*m) - V0*sp.sign(x) + alpha*sp.exp(-beta*sp.Abs(x)),
        "dim": 1,
        "category": "active_matter",
        "description": "Run-and-tumble dynamics — bacterial locomotion.",
    },
}

# =====================================================================
# 31. METAMATERIALS & PHONONIC CRYSTALS
# =====================================================================
H_METAMATERIALS = {
    "negative_refraction": {
        "expr": -sp.sqrt(xi**2 + eta**2) + V0*(sp.cos(x) + sp.cos(y)),
        "dim": 2,
        "category": "metamaterials",
        "description": "Negative refraction — backward wave propagation.",
    },
    "dirac_cone": {
        "expr": sp.sqrt(xi**2 + eta**2) + V0*(sp.cos(x) + sp.cos(y) + sp.cos(x-y)),
        "dim": 2,
        "category": "metamaterials",
        "description": "Dirac cone dispersion — graphene-like.",
    },
    "topological_insulator": {
        "expr": (xi**2 + eta**2)/2 + alpha*(sp.sin(x)**2 + sp.sin(y)**2) + beta*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "metamaterials",
        "description": "Topological insulator — edge state protection.",
    },
    "phononic_bandgap": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(2*x) + sp.cos(2*y)),
        "dim": 2,
        "category": "metamaterials",
        "description": "Phononic crystal — elastic wave bandgap.",
    },
    "pentamode_material": {
        "expr": (xi**2 + eta**2 + 2*alpha*xi*eta)/2 + beta*(x**2 + y**2),
        "dim": 2,
        "category": "metamaterials",
        "description": "Pentamode metamaterial — acoustic cloaking.",
    },
    "hyperbolic_metamaterial": {
        "expr": xi**2/(2*m) - eta**2/(2*m) + V0*(x**2 + y**2),
        "dim": 2,
        "category": "metamaterials",
        "description": "Hyperbolic dispersion — extreme anisotropy.",
    },
}

# =====================================================================
# 32. QUANTUM COMPUTING & INFORMATION
# =====================================================================
H_QUANTUM_INFO = {
    "transverse_ising_chain": {
        "expr": -alpha*sp.cos(x) - beta*x,
        "dim": 1,
        "category": "quantum_info",
        "description": "Transverse Ising — quantum phase transition.",
    },
    "xyz_spin_chain": {
        "expr": -alpha*sp.cos(x) - beta*sp.cos(y) - gamma*sp.cos(x-y),
        "dim": 2,
        "category": "quantum_info",
        "description": "XYZ model — fully anisotropic spin chain.",
    },
    "kitaev_chain": {
        "expr": xi**2/(2*m) + delta*sp.cos(x) + alpha*sp.sin(x),
        "dim": 1,
        "category": "quantum_info",
        "description": "Kitaev chain — Majorana fermions.",
    },
    "toric_code": {
        "expr": -(sp.cos(x) + sp.cos(y) + sp.cos(x+y)),
        "dim": 2,
        "category": "quantum_info",
        "description": "Toric code — topological quantum error correction.",
    },
    "cluster_state": {
        "expr": -sp.cos(x)*sp.cos(y),
        "dim": 2,
        "category": "quantum_info",
        "description": "Cluster state — measurement-based quantum computing.",
    },
    "rydberg_blockade": {
        "expr": (xi**2 + eta**2)/(2*m) + V0/(sp.Abs(x-y)**6 + eps),
        "dim": 2,
        "category": "quantum_info",
        "description": "Rydberg blockade — quantum simulation.",
    },
}

# =====================================================================
# 33. GEOPHYSICS & PLANETARY SCIENCE
# =====================================================================
H_GEOPHYSICS = {
    "seismic_wave": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*sp.exp(-beta*y))),
        "dim": 2,
        "category": "geophysics",
        "description": "Seismic wave in stratified medium.",
    },
    "mantle_convection": {
        "expr": (xi**2 + eta**2)/2 + alpha*y*(sp.exp(-x**2) - 1),
        "dim": 2,
        "category": "geophysics",
        "description": "Mantle convection — thermal plumes.",
    },
    "core_oscillation": {
        "expr": (xi**2 + eta**2)/2 - g/sp.sqrt(x**2 + y**2 + eps) + omega**2*(x**2 + y**2),
        "dim": 2,
        "category": "geophysics",
        "description": "Earth's core free oscillation.",
    },
    "tsunami_wave": {
        "expr": xi**2/(2*m) + g*sp.sqrt(x),
        "dim": 1,
        "category": "geophysics",
        "description": "Tsunami propagation — shallow water approximation.",
    },
    "dynamo_effect": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*eta - y*xi) - beta*sp.cos(x)*sp.cos(y),
        "dim": 2,
        "category": "geophysics",
        "description": "Geodynamo — planetary magnetic field generation.",
    },
}

# =====================================================================
# 34. CLIMATE & ATMOSPHERIC DYNAMICS
# =====================================================================
H_CLIMATE = {
    "lorenz63": {
        "expr": (xi**2 + eta**2)/2 + alpha*x*y - beta*y,
        "dim": 2,
        "category": "climate",
        "description": "Lorenz-63 system — deterministic chaos in convection.",
    },
    "lorenz96": {
        "expr": xi**2/2 + (x - sp.sin(2*sp.pi*alpha))**2,
        "dim": 1,
        "category": "climate",
        "description": "Lorenz-96 — atmospheric predictability model.",
    },
    "hadley_cell": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.sin(y)*x,
        "dim": 2,
        "category": "climate",
        "description": "Hadley cell circulation — tropical convection.",
    },
    "enso_oscillation": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.tanh(beta*x)*y,
        "dim": 2,
        "category": "climate",
        "description": "El Niño Southern Oscillation — coupled ocean-atmosphere.",
    },
    "ice_sheet_flow": {
        "expr": xi**2/(2*m) + g*x - alpha*x**3,
        "dim": 1,
        "category": "climate",
        "description": "Ice sheet dynamics — plastic flow model.",
    },
}

# =====================================================================
# 35. QUANTUM OPTICS & CAVITY QED
# =====================================================================
H_CAVITY_QED = {
    "rabi_oscillation": {
        "expr": omega*xi/2 + alpha*x*sp.cos(omega),
        "dim": 1,
        "category": "cavity_qed",
        "description": "Rabi oscillations — two-level system in field.",
    },
    "tavis_cummings": {
        "expr": omega*(xi + eta) + alpha*(x + y)*sp.cos(omega),
        "dim": 2,
        "category": "cavity_qed",
        "description": "Tavis–Cummings — multiple atoms in cavity.",
    },
    "dicke_model": {
        "expr": omega*xi + alpha*x*sp.sqrt(xi),
        "dim": 1,
        "category": "cavity_qed",
        "description": "Dicke superradiance — collective atom-light coupling.",
    },
    "purcell_effect": {
        "expr": (xi**2 + eta**2)/2 + omega*(x**2 + y**2)/2 + alpha*x*y,
        "dim": 2,
        "category": "cavity_qed",
        "description": "Purcell effect — cavity-enhanced emission.",
    },
    "optomechanics": {
        "expr": omega*xi + k*x**2/2 + alpha*x*xi,
        "dim": 1,
        "category": "cavity_qed",
        "description": "Cavity optomechanics — light-matter coupling.",
    },
}

# =====================================================================
# 36. CRYSTAL DEFECTS & SOLID STATE
# =====================================================================
H_DEFECTS = {
    "edge_dislocation": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.atan2(y, x),
        "dim": 2,
        "category": "defects",
        "description": "Edge dislocation — topological defect in crystal.",
    },
    "screw_dislocation": {
        "expr": (xi**2 + eta**2)/2 + alpha*y*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "defects",
        "description": "Screw dislocation — helical lattice defect.",
    },
    "vacancy_diffusion": {
        "expr": xi**2/(2*m) + V0*sum([sp.exp(-alpha*(x - n)**2) for n in range(-3, 4)]),
        "dim": 1,
        "category": "defects",
        "description": "Vacancy hopping in crystal lattice.",
    },
    "peierls_nabarro": {
        "expr": xi**2/(2*m) + V0*sp.sin(2*sp.pi*x)**2,
        "dim": 1,
        "category": "defects",
        "description": "Peierls–Nabarro potential — dislocation core.",
    },
    "twin_boundary": {
        "expr": xi**2/(2*m) + alpha*sp.tanh(beta*x)**2,
        "dim": 1,
        "category": "defects",
        "description": "Twin boundary — grain boundary energy.",
    },
}

# =====================================================================
# 37. ULTRA-COLD ATOMS & BEC
# =====================================================================
H_ULTRACOLD = {
    "optical_lattice_bec": {
        "expr": xi**2/(2*m) + V0*sp.sin(x)**2 + g*x**2,
        "dim": 1,
        "category": "ultracold",
        "description": "BEC in optical lattice — Bloch oscillations.",
    },
    "josephson_junction_bec": {
        "expr": -alpha*sp.cos(x) + beta*x**2,
        "dim": 1,
        "category": "ultracold",
        "description": "BEC Josephson junction — macroscopic tunneling.",
    },
    "feshbach_resonance": {
        "expr": xi**2/(2*m) + alpha*(x**2 - beta)**2/(x**2 + gamma),
        "dim": 1,
        "category": "ultracold",
        "description": "Feshbach resonance — tunable interactions.",
    },
    "raman_coupling": {
        "expr": (xi**2 + eta**2)/(2*m) + omega*y + alpha*x*y,
        "dim": 2,
        "category": "ultracold",
        "description": "Raman-coupled BEC — spin-orbit coupling.",
    },
    "vortex_lattice_bec": {
        "expr": (xi**2 + eta**2)/2 - omega*(x*eta - y*xi) + alpha*(x**2 + y**2),
        "dim": 2,
        "category": "ultracold",
        "description": "Vortex lattice in rotating BEC.",
    },
}

# =====================================================================
# 38. STOCHASTIC PROCESSES & LÉVY FLIGHTS
# =====================================================================
H_STOCHASTIC = {
    "ornstein_uhlenbeck": {
        "expr": xi**2/(2*m) + alpha*x**2/2 - gamma*x*xi,
        "dim": 1,
        "category": "stochastic",
        "description": "Ornstein–Uhlenbeck process — mean-reverting noise.",
    },
    "levy_stable": {
        "expr": sp.Abs(xi)**alpha/alpha + V0*x**2/2,
        "dim": 1,
        "category": "stochastic",
        "description": "Lévy stable process — heavy-tailed jumps.",
    },
    "fractional_brownian": {
        "expr": sp.Abs(xi)**(2*alpha) + V0*x**2/2,
        "dim": 1,
        "category": "stochastic",
        "description": "Fractional Brownian motion — long-range correlations.",
    },
    "continuous_time_random_walk": {
        "expr": xi**2/(2*m) + V0*sp.exp(-alpha*sp.Abs(x)),
        "dim": 1,
        "category": "stochastic",
        "description": "CTRW — anomalous diffusion.",
    },
}

# =====================================================================
# 39. THÉORIE DES CORDES & GRAVITÉ QUANTIQUE
# =====================================================================
H_STRING_THEORY = {
    "nambu_goto": {
        "expr": sp.sqrt((xi**2 + eta**2) * (1 + alpha*(x - y)**2)),
        "dim": 2,
        "category": "string_theory",
        "description": "Discrete approximation of Nambu–Goto action using two coupled points on the string.",
    },
    
    "polyakov": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x - y)**2,
        "dim": 2,
        "category": "string_theory",
        "description": "Discretized Polyakov string as coupled oscillators with tension term.",
    },
    "ads_cft_particle": {
        "expr": sp.sqrt(xi**2 + eta**2)/y + V0*y**(-delta),
        "dim": 2,
        "category": "string_theory",
        "description": "Particle in AdS space — holographic correspondence.",
    },
    "brane_fluctuation": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2) + beta/(x**2 + y**2 + eps),
        "dim": 2,
        "category": "string_theory",
        "description": "D-brane fluctuations — open string endpoints.",
    },
    "randall_sundrum": {
        "expr": sp.exp(-2*alpha*sp.Abs(y))*xi**2/(2*m) + V0*sp.exp(-alpha*sp.Abs(y)),
        "dim": 2,
        "category": "string_theory",
        "description": "Randall–Sundrum warped geometry.",
    },
    "dvali_gabadadze": {
        "expr": (xi**2 + eta**2)/2 - 1/(sp.sqrt(x**2 + y**2) + eps) + alpha*sp.sqrt(x**2 + y**2),
        "dim": 2,
        "category": "string_theory",
        "description": "DGP model — massive gravity modification.",
    },
    "regge_trajectory": {
        "expr": alpha*xi**2 + beta*x**2,
        "dim": 1,
        "category": "string_theory",
        "description": "Regge trajectory — rotating string spectrum.",
    },
}

# =====================================================================
# 40. PHYSIQUE DES PARTICULES & QCD
# =====================================================================
H_PARTICLE_PHYSICS = {
    "quark_confinement": {
        "expr": xi**2/(2*m) + alpha*sp.Abs(x),
        "dim": 1,
        "category": "particle_physics",
        "description": "Linear confinement potential — QCD string.",
    },
    "cornell_potential": {
        "expr": xi**2/(2*m) - alpha/sp.Abs(x) + beta*sp.Abs(x),
        "dim": 1,
        "category": "particle_physics",
        "description": "Cornell potential — quarkonium (charmonium, bottomonium).",
    },
    "instanton": {
        "expr": (xi**2 + eta**2)/2 + V0/(1 + alpha*(x**2 + y**2))**2,
        "dim": 2,
        "category": "particle_physics",
        "description": "Instanton solution — quantum tunneling in QFT.",
    },
    "sphaleron": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.sin(x)**2*sp.sin(y)**2/(sp.sin(x)**2 + sp.sin(y)**2 + eps),
        "dim": 2,
        "category": "particle_physics",
        "description": "Sphaleron — baryon number violation.",
    },
    "skyrme_model": {
        "expr": (xi**2 + eta**2)/2 + alpha*(xi**2 + eta**2) + beta*(x**2 + y**2)**2,
        "dim": 2,
        "category": "particle_physics",
        "description": "Skyrme model (simplified) — topological solitons as baryons.",
    },
    "electroweak_phase": {
        "expr": xi**2/(2*m) - alpha*x**2 + beta*x**4 + gamma*x**3,
        "dim": 1,
        "category": "particle_physics",
        "description": "Electroweak phase transition potential.",
    },
    "glueball": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.Abs(x*y) + beta*(x**2 + y**2),
        "dim": 2,
        "category": "particle_physics",
        "description": "Glueball — bound state of gluons.",
    },
    "parton_distribution": {
        "expr": xi**2/(2*m) + V0*x**(alpha)*(1-x)**beta,
        "dim": 1,
        "category": "particle_physics",
        "description": "Parton distribution function — deep inelastic scattering.",
    },
}

# =====================================================================
# 41. GRAVITÉ QUANTIQUE & LOOP QUANTUM GRAVITY
# =====================================================================
H_QUANTUM_GRAVITY = {
    "wheeler_dewitt": {
        "expr": xi**2/(2*m) + V0*sp.exp(3*x),
        "dim": 1,
        "category": "quantum_gravity",
        "description": "Wheeler–DeWitt equation — quantum cosmology (minisuperspace).",
    },
    "ashtekar_variable": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*eta - y*xi)**2,
        "dim": 2,
        "category": "quantum_gravity",
        "description": "Ashtekar variables — loop quantum gravity.",
    },
    "spin_network": {
        "expr": -alpha*sp.sqrt(x*(x+1)) - beta*sp.sqrt(y*(y+1)),
        "dim": 2,
        "category": "quantum_gravity",
        "description": "Spin network dynamics — discrete quantum geometry.",
    },
    "causal_set": {
        "expr": xi**2/(2*m) + sum([V0*sp.Heaviside(x - n) for n in range(-5, 6)]),
        "dim": 1,
        "category": "quantum_gravity",
        "description": "Causal set — discrete spacetime structure.",
    },
    "horava_lifshitz": {
        "expr": xi**2/(2*m) + alpha*xi**4 + beta*x**2,
        "dim": 1,
        "category": "quantum_gravity",
        "description": "Hořava–Lifshitz gravity — anisotropic scaling with higher derivatives.",
    },
}

# =====================================================================
# 42. SYSTÈMES INTÉGRABLES CLASSIQUES AVANCÉS
# =====================================================================
H_INTEGRABLE_ADVANCED = {
    "kowalevski_top": {
        "expr": (xi**2 + 2*eta**2)/2 + m*g*x,
        "dim": 2,
        "category": "integrable_advanced",
        "description": "Kowalevski top — integrable spinning top.",
    },
    "goryachev_chaplygin": {
        "expr": (xi**2 + eta**2 + 4*xi*eta)/2 + alpha*x,
        "dim": 2,
        "category": "integrable_advanced",
        "description": "Goryachev–Chaplygin top — another integrable case.",
    },
    "garnier_system": {
        "expr": (xi**2 + eta**2)/2 + alpha/(x - y) + beta/(x + y),
        "dim": 2,
        "category": "integrable_advanced",
        "description": "Garnier system — higher-order Painlevé.",
    },
    "schlesinger_system": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.log(sp.Abs(x - y)) + beta*sp.log(sp.Abs(x + y)),
        "dim": 2,
        "category": "integrable_advanced",
        "description": "Schlesinger system — isomonodromic deformations.",
    },
    "bogoyavlensky_lattice": {
        "expr": xi**2/(2*m) + alpha*sp.exp(x - sp.sin(x)),
        "dim": 1,
        "category": "integrable_advanced",
        "description": "Bogoyavlensky–Toda lattice — integrable discretization.",
    },
    "ruijsenaars_schneider": {
        "expr": sp.Product(sp.sinh((xi - eta + k)/2)/sp.sinh(k/2), (k, 1, alpha)),
        "dim": 2,
        "category": "integrable_advanced",
        "description": "Ruijsenaars–Schneider model — relativistic Calogero.",
    },
}

# =====================================================================
# 43. SYSTÈMES HORS ÉQUILIBRE & THERMODYNAMIQUE
# =====================================================================
H_NON_EQUILIBRIUM = {
    "jarzynski_work": {
        "expr": xi**2/(2*m) + k*(x - lambda_param)**2/2,
        "dim": 1,
        "category": "non_equilibrium",
        "description": "Jarzynski equality setup — nonequilibrium work.",
    },
    "crooks_fluctuation": {
        "expr": xi**2/(2*m) + V0*(sp.tanh(alpha*x) + 1),
        "dim": 1,
        "category": "non_equilibrium",
        "description": "Crooks fluctuation theorem — time-reversal asymmetry.",
    },
    "mpemba_effect": {
        "expr": xi**2/(2*m) + alpha*x**2*(1 - sp.exp(-beta*x**2)),
        "dim": 1,
        "category": "non_equilibrium",
        "description": "Mpemba effect model — anomalous cooling.",
    },
    "heat_engine": {
        "expr": xi**2/(2*m) + k*x**2/2 - alpha*x*sp.cos(omega),
        "dim": 1,
        "category": "non_equilibrium",
        "description": "Quantum heat engine — Carnot-like cycle.",
    },
    "loschmidt_echo": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.cos(x)*sp.cos(y) + eps*sp.sin(x)*sp.sin(y),
        "dim": 2,
        "category": "non_equilibrium",
        "description": "Loschmidt echo — quantum irreversibility.",
    },
    "kibble_zurek": {
        "expr": xi**2/(2*m) - alpha*(1 - lambda_param)*x**2 + beta*x**4,
        "dim": 1,
        "category": "non_equilibrium",
        "description": "Kibble–Zurek mechanism — defect formation in phase transitions.",
    },
}

# =====================================================================
# 44. THÉORIE DES TWISTEURS & GÉOMÉTRIE COMPLEXE
# =====================================================================
H_TWISTOR = {
    "penrose_twistor": {
        "expr": (xi**2 + eta**2)/2 + sp.I*alpha*(x*eta - y*xi),
        "dim": 2,
        "category": "twistor",
        "description": "Penrose twistor space — complex spacetime geometry.",
    },
    "hitchin_system": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.log(sp.Abs(x**2 + y**2)),
        "dim": 2,
        "category": "twistor",
        "description": "Hitchin system — integrable gauge theory.",
    },
    "calabi_yau_geodesic": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*(x**2 + y**2))**2),
        "dim": 2,
        "category": "twistor",
        "description": "Geodesic on Calabi–Yau manifold.",
    },
    "kaehler_potential": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.log(1 + x**2 + y**2),
        "dim": 2,
        "category": "twistor",
        "description": "Kähler geometry — complex differential geometry.",
    },
}

# =====================================================================
# 45. SUPERSYMÉTRIE & SUPERGRAVITÉ
# =====================================================================
H_SUPERSYMMETRY = {
    "susy_harmonic": {
        "expr": xi**2/(2*m) + k*x**2/2 + alpha*xi,
        "dim": 1,
        "category": "supersymmetry",
        "description": "Supersymmetric harmonic oscillator (simplified bosonic sector).",
    },
    "witten_index": {
        "expr": xi**2/(2*m) + (2*V0*x)**2/2 - V0,
        "dim": 1,
        "category": "supersymmetry",
        "description": "Witten index — SUSY partner potential.",
    },
    "n2_susy": {
        "expr": (xi**2 + eta**2)/2 + V0*(x**2 + y**2) + alpha*(xi**2 + eta**2)*(x**2 + y**2),
        "dim": 2,
        "category": "supersymmetry",
        "description": "N=2 supersymmetry — extended SUSY with coupling.",
    },
    "seiberg_witten": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 - y**2)**2 + beta*x*y,
        "dim": 2,
        "category": "supersymmetry",
        "description": "Seiberg–Witten theory — N=2 gauge theory.",
    },
    "sugra_scalar": {
        "expr": xi**2/(2*m) - alpha*sp.exp(beta*x) + gamma*sp.exp(2*beta*x),
        "dim": 1,
        "category": "supersymmetry",
        "description": "Supergravity scalar potential.",
    },
}

# =====================================================================
# 46. MATIÈRE NOIRE & ÉNERGIE NOIRE
# =====================================================================
H_DARK_SECTOR = {
    "wimp_scattering": {
        "expr": xi**2/(2*m) + g*sp.exp(-alpha*x**2)/x**2,
        "dim": 1,
        "category": "dark_sector",
        "description": "WIMP–nucleon scattering — dark matter detection.",
    },
    "axion_field": {
        "expr": xi**2/(2*m) + V0*(1 - sp.cos(x/alpha)),
        "dim": 1,
        "category": "dark_sector",
        "description": "Axion field — dark matter candidate.",
    },
    "fuzzy_dark_matter": {
        "expr": (xi**2 + eta**2)/(2*m) + g*(x**2 + y**2)**2,
        "dim": 2,
        "category": "dark_sector",
        "description": "Fuzzy dark matter — ultralight bosons.",
    },
    "chameleon_field": {
        "expr": xi**2/(2*m) + V0*sp.exp(alpha*x) + beta/x**4,
        "dim": 1,
        "category": "dark_sector",
        "description": "Chameleon mechanism — screened fifth force.",
    },
    "dark_photon": {
        "expr": (xi**2 + eta**2)/(2*m) - eps*alpha/(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "dark_sector",
        "description": "Dark photon — hidden sector U(1).",
    },
    "phantom_energy": {
        "expr": -xi**2/(2*m) + V0*x**2,
        "dim": 1,
        "category": "dark_sector",
        "description": "Phantom dark energy — w < -1 equation of state.",
    },
}

# =====================================================================
# 47. PHYSIQUE DES NEUTRINOS
# =====================================================================
H_NEUTRINO = {
    "neutrino_oscillation": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha*sp.sin(2*sp.atan2(y, x)),
        "dim": 2,
        "category": "neutrino",
        "description": "Neutrino flavor oscillations — mixing matrix.",
    },
    "msw_effect": {
        "expr": xi**2/(2*m) + V0*x + alpha*sp.cos(2*beta),
        "dim": 1,
        "category": "neutrino",
        "description": "Mikheyev–Smirnov–Wolfenstein effect — matter enhancement.",
    },
    "majorana_mass": {
        "expr": xi**2/(2*m) + alpha*x**2 + beta*xi**2,
        "dim": 1,
        "category": "neutrino",
        "description": "Majorana neutrino mass term.",
    },
    "sterile_neutrino": {
        "expr": (xi**2 + eta**2)/(2*m) + eps*(x - y)**2,
        "dim": 2,
        "category": "neutrino",
        "description": "Sterile neutrino mixing — dark sector coupling.",
    },
}

# =====================================================================
# 48. MATIÈRE ÉTRANGE & ÉTATS EXOTIQUES
# =====================================================================
H_EXOTIC_MATTER = {
    "quark_gluon_plasma": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2)*sp.log(x**2 + y**2 + eps),
        "dim": 2,
        "category": "exotic_matter",
        "description": "Quark–gluon plasma — deconfined QCD matter.",
    },
    "color_superconductor": {
        "expr": (xi**2 + eta**2)/(2*m) - delta*sp.cos(x - y),
        "dim": 2,
        "category": "exotic_matter",
        "description": "Color superconductivity — quark Cooper pairs.",
    },
    "strangelets": {
        "expr": xi**2/(2*m) + alpha*sp.Abs(x)**(4/3),
        "dim": 1,
        "category": "exotic_matter",
        "description": "Strangelet — hypothetical strange quark matter.",
    },
    "pentaquark": {
        "expr": (xi**2 + eta**2)/2 + V0/(sp.Abs(x - y) + eps) + alpha*(x**2 + y**2),
        "dim": 2,
        "category": "exotic_matter",
        "description": "Pentaquark state — exotic hadron.",
    },
    "tetraquark": {
        "expr": (xi**2 + eta**2)/2 - alpha/sp.Abs(x - y) + beta*sp.Abs(x - y),
        "dim": 2,
        "category": "exotic_matter",
        "description": "Tetraquark — four-quark bound state.",
    },
}

# =====================================================================
# 49. INFORMATION QUANTIQUE AVANCÉE
# =====================================================================
H_QUANTUM_INFO_ADVANCED = {
    "quantum_discord": {
        "expr": -(x**2 + y**2)*sp.log(x**2 + y**2 + eps)/2,
        "dim": 2,
        "category": "quantum_info_advanced",
        "description": "Quantum discord — beyond-entanglement correlations.",
    },
    "measurement_back_action": {
        "expr": xi**2/(2*m) + k*x**2/2 + gamma*xi**2*x**2,
        "dim": 1,
        "category": "quantum_info_advanced",
        "description": "Measurement back-action — Heisenberg uncertainty.",
    },
    "quantum_zeno": {
        "expr": xi**2/(2*m) + V0*sp.exp(-alpha*x**2)*(1 - sp.exp(-beta)),
        "dim": 1,
        "category": "quantum_info_advanced",
        "description": "Quantum Zeno effect — inhibition by measurement.",
    },
    "contextuality": {
        "expr": -(sp.cos(x) + sp.cos(y) + sp.cos(x + y)),
        "dim": 2,
        "category": "quantum_info_advanced",
        "description": "Quantum contextuality — Kochen–Specker theorem.",
    },
    "entanglement_swapping": {
        "expr": (xi**2 + eta**2)/2 - alpha*(x*y + xi*eta),
        "dim": 2,
        "category": "quantum_info_advanced",
        "description": "Entanglement swapping protocol.",
    },
}

# =====================================================================
# 50. SYSTÈMES PUREMENT MATHÉMATIQUES & ABSTRAITS
# =====================================================================
H_PURE_MATH = {
    "riemann_hypothesis": {
        "expr": xi**2/(2*m) + sp.re(sp.zeta(0.5 + sp.I*x)),
        "dim": 1,
        "category": "pure_math",
        "description": "Riemann zeta on critical line — analytic number theory.",
    },
    "modular_form": {
        "expr": (xi**2 + eta**2)/(2*(sp.im(x + sp.I*y))**2),
        "dim": 2,
        "category": "pure_math",
        "description": "Modular form geodesic — automorphic functions.",
    },
    "fibonacci_potential": {
        "expr": xi**2/(2*m) + V0*sp.cos(2*sp.pi*x/(1 + sp.sqrt(5))/2),
        "dim": 1,
        "category": "pure_math",
        "description": "Fibonacci quasicrystal — golden ratio modulation.",
    },
    "cantor_set": {
        "expr": xi**2/(2*m) + sum([V0*sp.exp(-alpha*(x - 3**(-n))**2) for n in range(1, 8)]),
        "dim": 1,
        "category": "pure_math",
        "description": "Cantor set potential — fractal energy landscape.",
    },
    "mandelbrot_escape": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "pure_math",
        "description": "Mandelbrot set escape dynamics.",
    },
    "julia_set": {
        "expr": (xi**2 + eta**2)/2 + V0/(1 + (x**2 + y**2)**2),
        "dim": 2,
        "category": "pure_math",
        "description": "Julia set — complex dynamics.",
    },
}

# =====================================================================
# 51. PHÉNOMÉNOLOGIE AU-DELÀ DU MODÈLE STANDARD
# =====================================================================
H_BSM = {
    "little_higgs": {
        "expr": xi**2/(2*m) + V0*(x**2 + y**2) - alpha*(x**2 + y**2)**2 + beta*(x**4 + y**4),
        "dim": 2,
        "category": "bsm",
        "description": "Little Higgs models — natural EWSB.",
    },
    "composite_higgs": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.sin(sp.sqrt(x**2 + y**2)/alpha)**2,
        "dim": 2,
        "category": "bsm",
        "description": "Composite Higgs — strongly coupled dynamics.",
    },
    "extra_dimension_kk": {
        "expr": (xi**2 + eta**2)/(2*m) + sum([alpha*n**2/(x**2 + y**2 + eps) for n in range(1, 6)]),
        "dim": 2,
        "category": "bsm",
        "description": "Kaluza–Klein tower — extra dimensions.",
    },
    "technicolor": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.cos(x/alpha)*sp.cos(y/beta),
        "dim": 2,
        "category": "bsm",
        "description": "Technicolor — dynamical electroweak symmetry breaking.",
    },
    "leptoquark": {
        "expr": (xi**2 + eta**2)/(2*m) + alpha/(x**2 + y**2 + eps) + beta*(x + y),
        "dim": 2,
        "category": "bsm",
        "description": "Leptoquark interaction — lepton-quark unification.",
    },
}

# =====================================================================
# 52. PHYSIQUE DES TROUS NOIRS
# =====================================================================
H_BLACK_HOLES = {
    "schwarzschild_particle": {
        "expr": (1 - 2*m/sp.sqrt(x**2 + y**2 + eps))*xi**2/2 + eta**2/(2*(1 - 2*m/sp.sqrt(x**2 + y**2 + eps))),
        "dim": 2,
        "category": "black_holes",
        "description": "Particle in Schwarzschild spacetime.",
    },
    "kerr_geodesic": {
        "expr": xi**2/2 + eta**2/(2*(1 + alpha*sp.cos(y)**2)) + beta*(x*eta)/(1 + alpha*sp.cos(y)**2),
        "dim": 2,
        "category": "black_holes",
        "description": "Kerr black hole geodesic — rotating BH.",
    },
    "hawking_radiation": {
        "expr": xi**2/(2*m) - alpha/x + beta*sp.exp(-gamma*x),
        "dim": 1,
        "category": "black_holes",
        "description": "Hawking radiation effective potential.",
    },
    "ads_black_hole": {
        "expr": (1 - m/x**2 - x**2)*xi**2/2 + V0*x**2,
        "dim": 1,
        "category": "black_holes",
        "description": "Anti-de Sitter black hole.",
    },
    "information_paradox": {
        "expr": xi**2/(2*m) + alpha*x**2 - beta*sp.log(x**2 + eps),
        "dim": 1,
        "category": "black_holes",
        "description": "Black hole information paradox model.",
    },
}

# =====================================================================
# FIELD THEORY CORRECTED
# =====================================================================
H_FIELD_THEORY_PROPER = {
    "klein_gordon_field": {
        "expr": (xi**2 + eta**2)/2 + m**2*(x**2 + y**2)/2 + lambda_param*(x**2 + y**2)**2/4,
        "dim": 2,
        "category": "field_theory",
        "description": "Klein–Gordon field — scalar field with self-interaction.",
    },
    "sine_gordon_field": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x))*(1 - sp.cos(y)),
        "dim": 2,
        "category": "field_theory",
        "description": "Sine-Gordon field — 2D discretized field theory.",
    },
    "phi_fourth_field": {
        "expr": (xi**2 + eta**2)/2 + m**2*(x**2 + y**2)/2 + lambda_param*(x**4 + y**4)/4,
        "dim": 2,
        "category": "field_theory",
        "description": "φ⁴ theory — quartic self-interaction.",
    },
    "coupled_oscillator_chain": {
        "expr": (xi**2 + eta**2)/(2*m) + k*(x**2 + y**2)/2 + alpha*(x - y)**2/2,
        "dim": 2,
        "category": "field_theory",
        "description": "Coupled oscillator chain — lattice field theory approximation.",
    },
}

# =====================================================================
# 53. CONTROL THEORY & OPTIMIZATION
# =====================================================================
H_CONTROL_THEORY = {
    "lqr_problem": {
        "expr": alpha*x*xi + k*x**2/2 - beta*xi**2/2,
        "dim": 1,
        "category": "control_theory",
        "description": "LQR Problem (Linear Quadratic Regulator) 1D — Pontryagin Hamiltonian after control optimization.",
    },
    "bang_bang_control": {
        "expr": xi*y + alpha*sp.Abs(eta),
        "dim": 2,
        "category": "control_theory",
        "description": "Bang-Bang Control (double integrator) — Pontryagin Hamiltonian for a minimum-time problem with bounded control.",
    },
    "optimal_braking": {
        "expr": xi*y + (eta**2)/(4*alpha) - k*x,
        "dim": 2,
        "category": "control_theory",
        "description": "Optimal Braking Problem (quadratic cost on control) — H = p*v + V(x) + u^2/(2a).",
    },
    "fuller_problem": {
        "expr": xi*y + eta*sp.sign(x),
        "dim": 2,
        "category": "control_theory",
        "description": "Fuller Problem — Double integrator system with a cost $x^2$.",
    },
}
# =====================================================================
# 54. ACOUSTICS & WAVE DYNAMICS
# =====================================================================
H_ACOUSTICS = {
    "helmholtz_homogeneous": {
        "expr": alpha * sp.sqrt(xi**2 + eta**2),
        "dim": 2,
        "category": "acoustics",
        "description": "Helmholtz Equation (homogeneous medium) — Dispersion relation $\omega = c|k|$ (geometrical acoustics).",
    },
    "acoustic_gradient": {
        "expr": (xi**2 + eta**2)/2 - (1 + alpha*x)**2 / 2,
        "dim": 2,
        "category": "acoustics",
        "description": "Acoustic wave in a medium with a refractive index gradient $n(x) = 1 + \alpha x$ — Ray refraction.",
    },
    "acoustic_waveguide": {
        "expr": (xi**2 + eta**2)/2 - V0*sp.exp(-alpha*x**2),
        "dim": 2,
        "category": "acoustics",
        "description": "Acoustic waveguide (SOFAR channel type) — Gaussian index profile creating a potential well.",
    },
    "paraxial_wave": {
        "expr": xi**2 / (2*m) + eta**2 / (2*m) + k*x,
        "dim": 2,
        "category": "acoustics",
        "description": "Paraxial wave equation (Schrödinger analog) — Acoustic beam propagation.",
    },
}

# =====================================================================
# 55. COMPLEX NETWORK DYNAMICS
# =====================================================================
H_NETWORK_DYNAMICS = {
    "kuramoto_hamiltonian": {
        "expr": (xi**2 + eta**2)/(2*m) - k*sp.cos(x - y),
        "dim": 2,
        "category": "network_dynamics",
        "description": "Kuramoto model (N=2) formulated as a Hamiltonian (XY model analog) — Study of synchronization.",
    },
    "network_consensus": {
        "expr": (xi**2 + eta**2)/(2*m) + k*(x - y)**2/2,
        "dim": 2,
        "category": "network_dynamics",
        "description": "Consensus dynamics (N=2) — Diffusion on a graph (quadratic harmonic potential).",
    },
    "kuramoto_chain_3": {
        "expr": (xi**2 + eta**2)/2 + k*(sp.cos(x - y) + sp.cos(y - x)), # Note: just an example, needs 3 vars
        "dim": 2, # Note: This is simplified for 2D
        "category": "network_dynamics",
        "description": "Chain of Kuramoto oscillators (simplified N=3) — Phase interaction potential.",
    },
    "hopfield_potential": {
        "expr": -(xi**2 + eta**2)/2 + alpha*(x**2 - 1)**2 + beta*(y**2 - 1)**2 - k*x*y,
        "dim": 2,
        "category": "network_dynamics",
        "description": "Hopfield network potential (N=2) — Dynamics of an associative memory.",
    },
}

# =====================================================================
# 56. SPIN GLASSES & DISORDERED SYSTEMS
# =====================================================================
H_SPIN_GLASS = {
    "sk_model_potential": {
        "expr": -alpha*(x**2 + y**2) + beta*(x**4 + y**4) + gamma*(x*y)**2,
        "dim": 2,
        "category": "spin_glass",
        "description": "Sherrington-Kirkpatrick (SK) potential — Free energy (2-spin replicas) showing a complex energy landscape.",
    },
    "edwards_anderson_pheno": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + alpha*sp.cos(x - y + beta)),
        "dim": 2,
        "category": "spin_glass",
        "description": "Edwards-Anderson model (phenomenological) — Frustrated and disordered periodic potential.",
    },
    "p_spin_potential": {
        "expr": -alpha*(x**3 + y**3) - beta*x*y,
        "dim": 2,
        "category": "spin_glass",
        "description": "'p-spin' model (p=3, N=2) — Complex energy landscape, model for the glass transition.",
    },
    "random_field_ising": {
        "expr": xi**2/(2*m) - k*sp.cos(x) - alpha*x,
        "dim": 1,
        "category": "spin_glass",
        "description": "Random Field Ising Model (RFIM) — Ferromagnetic interaction + disordered field (here constant).",
    },
}
# =====================================================================
# 57. MESOSCOPIC PHYSICS
# =====================================================================
H_MESOSCOPIC = {
    "coulomb_blockade": {
        "expr": alpha*(x - beta)**2 + k*xi**2,
        "dim": 1,
        "category": "mesoscopic",
        "description": "Coulomb blockade (Quantum Dot) — Charging energy $E_C(N-N_g)^2$ (x=Charge, xi=Phase).",
    },
    "caldeira_leggett": {
        "expr": xi**2/(2*m) + alpha*x**2/2 + eta**2/2 + k*y**2/2 + gamma*x*y,
        "dim": 2,
        "category": "mesoscopic",
        "description": "Caldeira-Leggett model (1 mode) — Decoherence of a quantum oscillator (x) coupled to a bath (y).",
    },
    "luttinger_liquid_2mode": {
        "expr": (xi**2 + eta**2)/2 + k*(x**2 + y**2)/2 + alpha*(x - y)**2,
        "dim": 2,
        "category": "mesoscopic",
        "description": "Luttinger liquid (2-mode approximation) — Bosonization of 1D fermions (plasmon/spin modes).",
    },
    "aharonov_bohm_ring": {
        "expr": (xi - alpha)**2/(2*m) + k*x**2,
        "dim": 1,
        "category": "mesoscopic",
        "description": "Aharonov-Bohm ring (1D) — Oscillator with magnetic flux $\\alpha$ shifting the canonical momentum.",
    },
}
# =====================================================================
# 58. POLYMER PHYSICS
# =====================================================================
H_POLYMERS = {
    "edwards_polymer": {
        "expr": xi**2/(2*m) + k*x**2/2 + alpha*x**4,
        "dim": 1,
        "category": "polymers",
        "description": "Edwards model (φ⁴ field theory) — Polymer chain with excluded volume interaction.",
    },
    "flory_huggins_energy": {
        "expr": k*(x*sp.log(x + eps) + (1-x)*sp.log(1-x + eps)) + alpha*x*(1-x),
        "dim": 1,
        "category": "polymers",
        "description": "Flory-Huggins free energy — Polymer mixture theory (x = volume fraction).",
    },
    "fjc_potential": {
        "expr": xi**2/(2*m) - alpha*sp.log(sp.sinh(k*x + eps)/(k*x + eps)),
        "dim": 1,
        "category": "polymers",
        "description": "Freely Jointed Chain (FJC) — Effective potential (Langevin approximation) for stretching.",
    },
    "worm_like_chain": {
        "expr": xi**2/(2*m) + k*x**2/(2*(1-x/alpha)),
        "dim": 1,
        "category": "polymers",
        "description": "Worm-like Chain (WLC) model — Elasticity for a semi-flexible polymer (x=extension, α=max length).",
    },
}

# =====================================================================
# 59. TOPOLOGICAL FIELD THEORIES (TFT)
# =====================================================================
H_TFT = {
    "chern_simons_abelian": {
        "expr": (x*eta - y*xi),
        "dim": 2,
        "category": "tft",
        "description": "Abelian Chern–Simons term — topological action for quantum Hall effect.",
    },
    "bf_model": {
        "expr": x*eta - y*xi + V0*(x**2 + y**2),
        "dim": 2,
        "category": "tft",
        "description": "BF theory in 2+1D (simplified phase-space form) — topological gravity analog.",
    },
    "wdvv_potential": {
        "expr": xi**2/(2*m) + sp.log(sp.Abs(sp.diff(V0*sp.exp(-x**2), x, 3))),
        "dim": 1,
        "category": "tft",
        "description": "WDVV equation from Frobenius manifold — enumerative geometry link.",
    },
    "achiral_tqft": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x*eta - y*xi)**2,
        "dim": 2,
        "category": "tft",
        "description": "Effective Hamiltonian for achiral topological quantum field theory.",
    },
}

# =====================================================================
# 60. ADVANCED GEOMETRIC HAMILTONIANS
# =====================================================================
H_GEOMETRIC_ADVANCED = {
    "sphere_geodesic": {
        "expr": (xi**2 + eta**2)/(2*(sp.cos(y)**2 + eps)),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "Geodesic flow on a sphere (latitude/longitude coordinates).",
    },
    "torus_geodesic": {
        "expr": (xi**2 + eta**2)/(2*(1 + 0.5*sp.cos(y))),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "Geodesic on a torus with major/minor radii ratio = 2.",
    },
    "ellipsoid_geodesic": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*sp.cos(x)**2 + beta*sp.sin(y)**2)),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "Geodesic on ellipsoid — non-constant curvature.",
    },
    "variable_curvature": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*sp.cos(x)**2)),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "2D metric with spatially varying Gaussian curvature.",
    },
    "neumann_oscillator": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2 + beta*y**2,
        "dim": 2,
        "category": "geometric_advanced",
        "description": "Neumann system — particle on sphere with quadratic potential.",
    },
    "calogero_sutherland": {
        "expr": (xi**2 + eta**2)/2 + g/(sp.sin((x - y)/2)**2 + eps),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "Calogero–Sutherland model — integrable with trigonometric interaction.",
    },
    "birkhoff_normal_form": {
        "expr": xi**2/2 + eta**2/2 + alpha*(x**2 + y**2)**2 + beta*(x**4 - y**4),
        "dim": 2,
        "category": "geometric_advanced",
        "description": "4th-order Birkhoff normal form near elliptic equilibrium.",
    },
}

# =====================================================================
# 61. SYMMETRIES AND HAMILTONIAN REDUCTIONS
# =====================================================================
H_SYMMETRY_REDUCED = {
    "particle_on_sphere": {
        "expr": (xi**2 + eta**2)/(2*m) + lambda_param*(x**2 + y**2 - R**2),
        "dim": 2,
        "category": "symmetry_reduced",
        "description": "Particle constrained to sphere via Lagrange multiplier (Dirac formalism).",
    },
    "reduced_rotor": {
        "expr": xi**2/(2*sp.I) + omega*L_z,
        "dim": 1,
        "category": "symmetry_reduced",
        "description": "Symplectic reduction of rotating rigid body (L_z = const).",
    },
    "gauge_invariant_oscillator": {
        "expr": ((xi - A*y)**2 + (eta + A*x)**2)/(2*m),
        "dim": 2,
        "category": "symmetry_reduced",
        "description": "U(1)-gauge invariant oscillator — conserved angular momentum.",
    },
    "magnetic_monopole_reduced": {
        "expr": (xi**2 + eta**2)/(2*m) + g*sp.acos(y/sp.sqrt(x**2 + y**2 + eps)),
        "dim": 2,
        "category": "symmetry_reduced",
        "description": "Dirac monopole with azimuthal symmetry reduction.",
    },
}

# =====================================================================
# 62. EXTENDED QUANTUM TOPOLOGICAL & RELATIVISTIC HAMILTONIANS
# =====================================================================
H_QUANTUM_TOPOLOGICAL_EXTENDED = {
    "dirac_2d_nonuniform_B": {
        "expr": sp.sqrt((xi - A*sp.exp(-x**2)*y)**2 + (eta + A*x*sp.exp(-y**2))**2 + m**2),
        "dim": 2,
        "category": "quantum_topological_extended",
        "description": "2D Dirac with Gaussian magnetic field — Landau levels + edge states.",
    },
    "weyl_semimetal": {
        "expr": sp.sqrt(xi**2 + eta**2) + alpha*(x*eta - y*xi),
        "dim": 2,
        "category": "quantum_topological_extended",
        "description": "Weyl semimetal Hamiltonian — linear dispersion + spin-momentum locking.",
    },
    "graphene_dirac": {
        "expr": xi*sp.cos(2*sp.pi/3) + eta*sp.sin(2*sp.pi/3) + V0*(sp.cos(x) + sp.cos(y)),
        "dim": 2,
        "category": "quantum_topological_extended",
        "description": "Continuum graphene Dirac Hamiltonian near K point.",
    },
    "majorana_wire": {
        "expr": sp.sqrt(xi**2 + Delta**2*sp.sin(x)**2) - mu,
        "dim": 1,
        "category": "quantum_topological_extended",
        "description": "Kitaev chain effective Hamiltonian for Majorana modes.",
    },
}

# =====================================================================
# 63. CONTINUUM FIELDS & MULTI-D SOLITONS
# =====================================================================
H_CONTINUUM_SOLITONS = {
    "nls_2d_radial": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2)**2,
        "dim": 2,
        "category": "continuum_solitons",
        "description": "2D cubic NLS — collapsing/vortex solitons.",
    },
    "sine_gordon_2d": {
        "expr": (xi**2 + eta**2)/2 + V0*(1 - sp.cos(sp.sqrt(x**2 + y**2))),
        "dim": 2,
        "category": "continuum_solitons",
        "description": "Radially symmetric 2D sine-Gordon — breather analogs.",
    },
    "kp_ii_equation": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**3 + beta*x*eta**2,
        "dim": 2,
        "category": "continuum_solitons",
        "description": "Kadomtsev–Petviashvili II — weakly 2D KdV.",
    },
    "benjamin_ono": {
        "expr": xi**2/2 + alpha*xi*sp.Abs(xi) + beta*x**2,
        "dim": 1,
        "category": "continuum_solitons",
        "description": "Benjamin–Ono equation — Hilbert transform dispersion.",
    },
    "camassa_holm": {
        "expr": xi**2/2 + alpha*x*xi**2 + beta*x**3,
        "dim": 1,
        "category": "continuum_solitons",
        "description": "Camassa–Holm — peakon solutions.",
    },
}

# =====================================================================
# 64. ADVANCED STOCHASTIC & DISSIPATIVE SYSTEMS
# =====================================================================
H_STOCHASTIC_ADVANCED = {
    "multiplicative_noise": {
        "expr": xi**2/2 + alpha*x**2 + beta*x*xi + gamma*xi**2*x,
        "dim": 1,
        "category": "stochastic_advanced",
        "description": "Hamiltonian with multiplicative noise coupling.",
    },
    "memory_kernel_effective": {
        "expr": xi**2/(2*m) + alpha*x**2/2 + beta*sp.exp(-gamma*sp.Abs(xi)),
        "dim": 1,
        "category": "stochastic_advanced",
        "description": "Effective Hamiltonian with memory (non-Markovian bath).",
    },
    "fokker_planck_nonquadratic": {
        "expr": sp.Abs(xi)**3/3 + V0*x**4,
        "dim": 1,
        "category": "stochastic_advanced",
        "description": "Non-quadratic kinetic term from anomalous diffusion.",
    },
}

# =====================================================================
# 65. MULTI-SCALE HYBRIDS & CHAOS
# =====================================================================
H_MULTI_SCALE_CHAOS = {
    "kam_perturbation": {
        "expr": xi**2/2 + eta**2/2 + eps*sp.cos(x)*sp.cos(y),
        "dim": 2,
        "category": "multi_scale_chaos",
        "description": "Near-integrable KAM system — weakly perturbed torus.",
    },
#    "chirikov_continuous": {
#        "expr": (xi**2 + eta**2)/2 + alpha*sp.cos(x)*sp.cos(omega*t),
#        "dim": 2,
#        "category": "multi_scale_chaos",
#        "description": "Continuous analog of Chirikov standard map.",
#    },
#    "two_timescale": {
#        "expr": xi**2/2 + V0*sp.cos(x)*sp.cos(omega*t),
#        "dim": 1,
#        "category": "multi_scale_chaos",
#        "description": "Explicitly time-dependent Hamiltonian (t treated as parameter).",
#    },
    "quasiperiodic_coupling": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.cos(x + golden_ratio*y),
        "dim": 2,
        "category": "multi_scale_chaos",
        "description": "Incommensurate coupling — Aubry transition precursor.",
    },
}

# =====================================================================
# 66. MODERN & COMPUTATIONAL EXTENSIONS
# =====================================================================
H_MODERN_EXTENSIONS = {
    "pt_symmetric_2d": {
        "expr": xi**2 + eta**2 + sp.I*(x**3 + y**3),
        "dim": 2,
        "category": "modern_extensions",
        "description": "2D PT-symmetric non-Hermitian Hamiltonian.",
    },
    "quantum_adiabatic": {
        "expr": A_param*(xi**2 + x**2) + B_param*(eta**2 + y**2) + C_param*x*y,
        "dim": 2,
        "category": "modern_extensions",
        "description": "Interpolated Hamiltonian for adiabatic quantum computing.",
    },
    "learned_hamiltonian": {
        "expr": xi**2/2 + alpha*sp.tanh(beta*x) + gamma*sp.sin(delta*x),
        "dim": 1,
        "category": "modern_extensions",
        "description": "Symbolic surrogate from Hamiltonian neural ODE.",
    },
}

# =====================================================================
# 67. GAME THEORY & EVOLUTIONARY DYNAMICS
# =====================================================================
H_GAME_DYNAMICS = {
    "replicator_hamiltonian": {
        "expr": xi**2/2 + alpha*x*(1 - x)*(x - beta),
        "dim": 1,
        "category": "game_dynamics",
        "description": "Hamiltonian form of replicator dynamics in 2-strategy evolutionary game.",
    },
    "hawk_dove_game": {
        "expr": xi**2/2 + V0*x*(1 - x) - C_param*x**2/2,
        "dim": 1,
        "category": "game_dynamics",
        "description": "Hawk-Dove game payoff encoded as effective potential.",
    },
    "prisoner_dilemma_potential": {
        "expr": (xi**2 + eta**2)/2 - alpha*(x*y + (1 - x)*(1 - y)),
        "dim": 2,
        "category": "game_dynamics",
        "description": "Potential encoding mutual cooperation vs betrayal in Prisoner's Dilemma.",
    },
}

# =====================================================================
# 68. OPTIMIZATION & MACHINE LEARNING
# =====================================================================
H_OPTIMIZATION = {
    "nesterov_ode": {
        "expr": xi**2/2 + k*x**2/2 + gamma*x*xi,
        "dim": 1,
        "category": "optimization",
        "description": "Continuous-time limit of Nesterov's accelerated gradient descent.",
    },
    "symplectic_sgd": {
        "expr": xi**2/2 + V0*sp.tanh(x)**2,
        "dim": 1,
        "category": "optimization",
        "description": "Symplectic stochastic gradient flow for nonconvex optimization.",
    },
    "primal_dual_hamiltonian": {
        "expr": xi*y - f_param*x - g_param*y,
        "dim": 2,
        "category": "optimization",
        "description": "Hamiltonian formulation of primal-dual optimization (f, g convex).",
    },
}

# =====================================================================
# 69. QUANTITATIVE FINANCE
# =====================================================================
H_QUANT_FINANCE = {
    "martingale_hamiltonian": {
        "expr": xi**2/2 - alpha*sp.log(x + eps),
        "dim": 1,
        "category": "quant_finance",
        "description": "Martingale constraint encoded as Hamiltonian potential (geometric Brownian motion).",
    },
    "portfolio_optimization": {
        "expr": (xi**2 + eta**2)/2 - alpha*x - beta*y + gamma*(x - y)**2,
        "dim": 2,
        "category": "quant_finance",
        "description": "Mean-variance portfolio selection as Hamiltonian system.",
    },
    "risk_measure_flow": {
        "expr": xi**2/2 + V0*sp.exp(-x**2) + alpha*x**4,
        "dim": 1,
        "category": "quant_finance",
        "description": "Dynamic risk measure (e.g., entropic risk) as potential.",
    },
}

# =====================================================================
# 70. SYMBOLIC COMPUTATION & REVERSIBLE LOGIC
# =====================================================================
H_SYMBOLIC_COMPUTATION = {
    "reversible_automaton": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.Mod(x + y, 2),
        "dim": 2,
        "category": "symbolic",
        "description": "Hamiltonian encoding of a reversible cellular automaton rule.",
    },
    "logical_gate_potential": {
        "expr": xi**2/2 + V0*(x - sp.Piecewise((0, x < 0.5), (1, True)))**2,
        "dim": 1,
        "category": "symbolic",
        "description": "Energy landscape enforcing binary logic behavior (e.g., step function).",
    },
}

# =====================================================================
# 71. GENERATIVE DESIGN & MORPHOGENESIS
# =====================================================================
H_GENERATIVE_DESIGN = {
    "growth_potential": {
        "expr": xi**2/2 + alpha*sp.exp(-x**2) * sp.cos(beta*x),
        "dim": 1,
        "category": "generative",
        "description": "Morphogenetic potential for procedural branching in design.",
    },
    "turing_pattern_design": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2*y - beta*x,
        "dim": 2,
        "category": "generative",
        "description": "Hamiltonian derived from Turing reaction-diffusion for generative art.",
    },
}

# =====================================================================
# 72. EPIDEMIOLOGY & POPULATION DYNAMICS
# =====================================================================
H_EPIDEMIOLOGY = {
    "sir_hamiltonian": {
        "expr": (xi**2 + eta**2)/2 + alpha*x*y - beta*y,
        "dim": 2,
        "category": "epidemiology",
        "description": "Hamiltonian form of SIR model — susceptible-infected-recovered flow.",
    },
    "seir_potential": {
        "expr": (xi**2 + eta**2 + zeta**2)/2 + alpha*x*y - beta*y - gamma*z,
        "dim": 2,  # Note: zeta, z not in global vars → simplified to 2D
        "category": "epidemiology",
        "description": "Reduced SEIR dynamics as effective 2D Hamiltonian (latent variable integrated).",
    },
    "epidemic_wave": {
        "expr": xi**2/2 + alpha*sp.exp(-x**2)*y,
        "dim": 2,
        "category": "epidemiology",
        "description": "Traveling epidemic wave — spatial spread with Gaussian kernel.",
    },
    "vaccination_game": {
        "expr": xi**2/2 + V0*x*(1 - x) - C_param*x,
        "dim": 1,
        "category": "epidemiology",
        "description": "Vaccination decision dynamics — cost-benefit in epidemic risk.",
    },
}

# =====================================================================
# 73. LINGUISTICS & SEMIOTIC SYSTEMS
# =====================================================================
H_LINGUISTICS = {
    "language_drift": {
        "expr": xi**2/2 + alpha*sp.cos(beta*x),
        "dim": 1,
        "category": "linguistics",
        "description": "Phonemic drift as particle in periodic potential — language evolution.",
    },
    "grammar_potential": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "linguistics",
        "description": "Syntactic alignment — energy landscape for grammatical agreement.",
    },
    "word_embedding_flow": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.exp(-alpha*(x**2 + y**2)),
        "dim": 2,
        "category": "linguistics",
        "description": "Effective Hamiltonian for word embedding dynamics in semantic space.",
    },
    "zipf_law_potential": {
        "expr": xi**2/2 - alpha*sp.log(x + eps),
        "dim": 1,
        "category": "linguistics",
        "description": "Zipf’s law as logarithmic potential — frequency vs rank in language.",
    },
}

# =====================================================================
# 74. ECOLOGY & ECOSYSTEM NETWORKS
# =====================================================================
H_ECOLOGY = {
    "lotka_volterra_hamiltonian": {
        "expr": (xi**2 + eta**2)/2 + alpha*x*y - beta*x - gamma*y,
        "dim": 2,
        "category": "ecology",
        "description": "Hamiltonian formulation of Lotka–Volterra predator-prey dynamics.",
    },
    "competitive_exclusion": {
        "expr": (xi**2 + eta**2)/2 - alpha*x**2 - beta*y**2 + gamma*x*y,
        "dim": 2,
        "category": "ecology",
        "description": "Competition model — niche overlap and exclusion principle.",
    },
    "mutualism_potential": {
        "expr": xi**2/2 + eta**2/2 - alpha*sp.log(1 + x) - beta*sp.log(1 + y) + gamma*x*y,
        "dim": 2,
        "category": "ecology",
        "description": "Mutualistic interaction — cooperative species benefit.",
    },
    "trophic_cascade": {
        "expr": (xi**2 + eta**2)/2 + alpha*x - beta*x*y + gamma*y*z,  # z not defined → use 2D proxy
        "dim": 2,
        "category": "ecology",
        "description": "Simplified trophic cascade (top-down control) as 2D effective Hamiltonian.",
    },
}

# =====================================================================
# 75. MACHINE LEARNING & PROBABILISTIC INFERENCE
# =====================================================================
H_INFERENCE = {
    "variational_free_energy": {
        "expr": xi**2/2 + alpha*x**2/2 + beta*sp.log(sp.cosh(x)),
        "dim": 1,
        "category": "inference",
        "description": "Variational free energy — inference as energy minimization.",
    },
    "expectation_propagation": {
        "expr": (xi**2 + eta**2)/2 + 0.5*(x - mu1)**2/sigma1**2 + 0.5*(y - mu2)**2/sigma2**2 - alpha*x*y,
        "dim": 2,
        "category": "inference",
        "description": "Expectation propagation — Gaussian message passing as coupled oscillators.",
    },
    "information_geometry": {
        "expr": (xi**2 + eta**2)/(2*(1 + alpha*x**2)),
        "dim": 2,
        "category": "inference",
        "description": "Fisher–Rao metric as curved Hamiltonian phase space.",
    },
    "diffusion_inference": {
        "expr": sp.Abs(xi)**alpha + V0*x**2/2,
        "dim": 1,
        "category": "inference",
        "description": "Score-based diffusion models — Lévy-driven inference dynamics.",
    },
}

# =====================================================================
# 76. URBAN DYNAMICS
# =====================================================================
H_URBAN = {
    "traffic_flow": {
        "expr": xi**2/2 + V0*x*(1 - x),
        "dim": 1,
        "category": "urban",
        "description": "Lighthill–Whitham traffic model — density-dependent flow potential.",
    },
    "land_use_competition": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2*(1 - x) - beta*y**2 + gamma*x*y,
        "dim": 2,
        "category": "urban",
        "description": "Competition between residential (x) and commercial (y) land use.",
    },
    "urban_heat_island": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.exp(-beta*(x**2 + y**2)),
        "dim": 2,
        "category": "urban",
        "description": "Urban heat island effect — temperature gradient as potential well.",
    },
    "pedestrian_evacuation": {
        "expr": xi**2/(2*m) - alpha*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "urban",
        "description": "Pedestrian escape dynamics — logarithmic attraction to exit.",
    },
}

# =====================================================================
# 77. COGNITIVE SCIENCE
# =====================================================================
H_COGNITIVE = {
    "belief_updating": {
        "expr": xi**2/2 + alpha*(x - beta)**2/2 - gamma*sp.log(sp.cosh(x)),
        "dim": 1,
        "category": "cognitive",
        "description": "Bayesian belief updating — Gaussian prior with logistic evidence.",
    },
    "attention_potential": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.exp(-alpha*(x**2 + y**2)),
        "dim": 2,
        "category": "cognitive",
        "description": "Spatial attention field — Gaussian focus of perceptual resources.",
    },
    "predictive_coding": {
        "expr": xi**2/2 + alpha*(x - y)**2/2,
        "dim": 2,
        "category": "cognitive",
        "description": "Predictive coding error — minimization of prediction vs sensation.",
    },
    "working_memory": {
        "expr": xi**2/2 + alpha*x**4 - beta*x**2,
        "dim": 1,
        "category": "cognitive",
        "description": "Bistable working memory — winner-take-all attractor dynamics.",
    },
}

# =====================================================================
# 78. LEGAL SYSTEMS
# =====================================================================
H_LEGAL = {
    "norm_diffusion": {
        "expr": xi**2/(2*m) - alpha*sp.log(1 + sp.Abs(x)),
        "dim": 1,
        "category": "legal",
        "description": "Diffusion of legal norms — logarithmic penalty for deviation.",
    },
    "precedent_flow": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.tanh(x)*sp.tanh(y),
        "dim": 2,
        "category": "legal",
        "description": "Precedent alignment — mutual reinforcement of case outcomes.",
    },
    "legal_entropy": {
        "expr": -x*sp.log(x + eps) - (1 - x)*sp.log(1 - x + eps) + alpha*x*xi,
        "dim": 1,
        "category": "legal",
        "description": "Legal uncertainty (entropy) as potential — binary legal states.",
    },
    "jurisprudential_tension": {
        "expr": xi**2/2 + alpha*sp.cos(beta*x),
        "dim": 1,
        "category": "legal",
        "description": "Cyclic legal interpretation — oscillation between doctrines.",
    },
}

# =====================================================================
# 79. ART & MUSIC
# =====================================================================
H_ART_MUSIC = {
    "harmonic_tension": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/12)),
        "dim": 1,
        "category": "art_music",
        "description": "Harmonic tension in 12-tone equal temperament — circular pitch space.",
    },
    "consonance_potential": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "art_music",
        "description": "Consonance model — energy minimized at simple frequency ratios.",
    },
    "generative_composition": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.sin(alpha*x)*sp.sin(beta*y),
        "dim": 2,
        "category": "art_music",
        "description": "Lissajous-inspired generative music — beat and phasing dynamics.",
    },
    "color_harmony": {
        "expr": (xi**2 + eta**2)/2 + V0*(1 - sp.cos(sp.atan2(y, x))),
        "dim": 2,
        "category": "art_music",
        "description": "Color harmony on hue circle — angular similarity in HSV space.",
    },
}

# =====================================================================
# 80. EDUCATION & LEARNING DYNAMICS
# =====================================================================
H_EDUCATION = {
    "learning_curve": {
        "expr": xi**2/2 + V0*(1 - sp.exp(-alpha*x)),
        "dim": 1,
        "category": "education",
        "description": "Learning curve — diminishing returns in skill acquisition.",
    },
    "knowledge_diffusion": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "education",
        "description": "Diffusion of knowledge — logarithmic attraction in idea space.",
    },
    "forgetting_potential": {
        "expr": xi**2/2 + alpha*x**2*sp.exp(-beta*x),
        "dim": 1,
        "category": "education",
        "description": "Ebbinghaus forgetting curve — memory decay with rehearsal.",
    },
    "curriculum_design": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x - y)**2 + beta*x**2,
        "dim": 2,
        "category": "education",
        "description": "Curriculum scaffolding — alignment of prior and new knowledge.",
    },
}

# =====================================================================
# 81. RELIGION & DOCTRINAL DYNAMICS
# =====================================================================
H_RELIGION = {
    "doctrinal_evolution": {
        "expr": xi**2/2 + V0*sp.cos(alpha*x),
        "dim": 1,
        "category": "religion",
        "description": "Doctrinal oscillation — cyclic reinterpretation of sacred texts.",
    },
    "ritual_periodicity": {
        "expr": xi**2/2 + alpha*(1 - sp.cos(2*sp.pi*x/T)),
        "dim": 1,
        "category": "religion",
        "description": "Ritual timing — harmonic potential for liturgical cycles.",
    },
    "sectarian_splitting": {
        "expr": (xi**2 + eta**2)/2 - alpha*x**2 - beta*y**2 + gamma*(x - y)**4,
        "dim": 2,
        "category": "religion",
        "description": "Sect formation — symmetry breaking in belief space.",
    },
    "religious_entropy": {
        "expr": -x*sp.log(x + eps) - (1 - x)*sp.log(1 - x + eps) + alpha*x*xi,
        "dim": 1,
        "category": "religion",
        "description": "Uncertainty in belief commitment — binary doctrinal states.",
    },
}

# =====================================================================
# 82. SPORTS & TACTICAL DYNAMICS
# =====================================================================
H_SPORTS = {
    "tactical_flow": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.exp(-alpha*(x**2 + y**2)),
        "dim": 2,
        "category": "sports",
        "description": "Team tactical focus — Gaussian attractor in field space.",
    },
    "player_interaction": {
        "expr": (xi**2 + eta**2)/2 + alpha/(sp.sqrt((x - y)**2 + eps)) - beta*(x + y)**2,
        "dim": 2,
        "category": "sports",
        "description": "Player coupling — attraction/repulsion in cooperative play.",
    },
    "game_momentum": {
        "expr": xi**2/2 + alpha*sp.tanh(beta*x)*x,
        "dim": 1,
        "category": "sports",
        "description": "Psychological momentum — nonlinear reinforcement of success.",
    },
    "zone_defense": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.Abs(x) + beta*sp.Abs(y),
        "dim": 2,
        "category": "sports",
        "description": "Zone defense potential — piecewise linear spatial control.",
    },
}

# =====================================================================
# 83. AGRICULTURE & ECOLOGICAL MANAGEMENT
# =====================================================================
H_AGRICULTURE = {
    "crop_rotation": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.cos(2*sp.pi*x/3) + beta*sp.cos(2*sp.pi*y/3),
        "dim": 2,
        "category": "agriculture",
        "description": "Three-field crop rotation — periodic soil nutrient cycling.",
    },
    "pest_predator": {
        "expr": (xi**2 + eta**2)/2 + alpha*x - beta*x*y + gamma*y,
        "dim": 2,
        "category": "agriculture",
        "description": "Pest dynamics with biological control — Lotka–Volterra analog.",
    },
    "soil_nutrient_diffusion": {
        "expr": xi**2/(2*m) - alpha*sp.log(x + eps) + beta*x,
        "dim": 1,
        "category": "agriculture",
        "description": "Soil fertility gradient — logarithmic depletion + replenishment.",
    },
    "drought_response": {
        "expr": xi**2/2 + V0*sp.exp(-alpha*x**2) + beta*x**4,
        "dim": 1,
        "category": "agriculture",
        "description": "Plant stress response — resilience under water scarcity.",
    },
}

# =====================================================================
# 84. PUBLIC HEALTH
# =====================================================================
H_PUBLIC_HEALTH = {
    "vaccination_campaign": {
        "expr": xi**2/2 + V0*x*(1 - x) - alpha*x,
        "dim": 1,
        "category": "public_health",
        "description": "Vaccination uptake dynamics — logistic coverage with cost penalty.",
    },
    "epidemic_preparedness": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.log(1 + x**2 + y**2),
        "dim": 2,
        "category": "public_health",
        "description": "Preparedness as attraction to central response hub — logarithmic potential.",
    },
    "herd_immunity_threshold": {
        "expr": xi**2/2 - alpha*(x - beta)**2/2,
        "dim": 1,
        "category": "public_health",
        "description": "Herd immunity as stable equilibrium — Gaussian well at critical coverage.",
    },
    "contact_tracing_flow": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.exp(-beta*(x**2 + y**2)),
        "dim": 2,
        "category": "public_health",
        "description": "Contact tracing as localized information potential — Gaussian kernel.",
    },
}

# =====================================================================
# 85. ARCHITECTURE
# =====================================================================
H_ARCHITECTURE = {
    "structural_flow": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2)**2,
        "dim": 2,
        "category": "architecture",
        "description": "Structural load distribution — quartic stiffness in planar frame.",
    },
    "spatial_perception": {
        "expr": (xi**2 + eta**2)/2 - V0*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "architecture",
        "description": "Perceptual attraction to center — logarithmic spatial focus.",
    },
    "circulation_potential": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.Abs(x) + beta*sp.Abs(y),
        "dim": 2,
        "category": "architecture",
        "description": "Pedestrian circulation — piecewise linear corridor constraints.",
    },
    "proportion_harmony": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/golden_ratio)),
        "dim": 1,
        "category": "architecture",
        "description": "Aesthetic proportion — harmonic potential at golden ratio intervals.",
    },
}

# =====================================================================
# 86. CUISINE
# =====================================================================
H_CUISINE = {
    "flavor_pairing": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.exp(-beta*(x - y)**2),
        "dim": 2,
        "category": "cuisine",
        "description": "Flavor compatibility — Gaussian attraction between similar tastes.",
    },
    "umami_potential": {
        "expr": xi**2/2 + V0/(1 + sp.exp(-alpha*x)) - beta*x**2,
        "dim": 1,
        "category": "cuisine",
        "description": "Umami taste response — sigmoid activation with saturation.",
    },
    "recipe_dynamics": {
        "expr": (xi**2 + eta**2)/2 + alpha*x*y*(1 - x - y),
        "dim": 2,
        "category": "cuisine",
        "description": "Recipe balance — ternary constraint (x + y ≤ 1) for ingredient ratios.",
    },
    "cooking_time_opt": {
        "expr": xi**2/(2*m) + alpha*(x - beta)**2 + gamma*sp.exp(-delta*x),
        "dim": 1,
        "category": "cuisine",
        "description": "Optimal cooking time — trade-off between doneness and degradation.",
    },
}

# =====================================================================
# 87. FASHION
# =====================================================================
H_FASHION = {
    "trend_diffusion": {
        "expr": xi**2/(2*m) - alpha*sp.log(1 + sp.Abs(x)),
        "dim": 1,
        "category": "fashion",
        "description": "Trend adoption — logarithmic resistance to deviation from norm.",
    },
    "style_cycles": {
        "expr": xi**2/2 + V0*sp.cos(alpha*x),
        "dim": 1,
        "category": "fashion",
        "description": "Cyclic revival of styles — periodic potential over decades.",
    },
    "aesthetic_tension": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "fashion",
        "description": "Outfit coherence — consonance between garment elements.",
    },
    "fast_fashion_dissipation": {
        "expr": xi**2/2 + beta*x**2 - gamma*x**3,
        "dim": 1,
        "category": "fashion",
        "description": "Fast fashion decay — rapid trend obsolescence (cubic instability).",
    },
}

# =====================================================================
# 88. RELAXATION & WELLNESS
# =====================================================================
H_WELLNESS = {
    "stress_recovery": {
        "expr": xi**2/2 + alpha*x**2*sp.exp(-beta*x),
        "dim": 1,
        "category": "wellness",
        "description": "Stress decay with nonlinear recovery — psychological resilience model.",
    },
    "circadian_rhythm": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/24)),
        "dim": 1,
        "category": "wellness",
        "description": "Circadian cycle — 24-hour periodic biological oscillator.",
    },
    "heart_rate_variability": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "wellness",
        "description": "HRV coherence — coupling between respiration and heart rate.",
    },
    "meditation_potential": {
        "expr": xi**2/2 - alpha*sp.log(sp.cosh(x)) + beta*x**2,
        "dim": 1,
        "category": "wellness",
        "description": "Meditative state as energy well — entropy reduction in mental noise.",
    },
}

# =====================================================================
# 89. DIGITAL CULTURE
# =====================================================================
H_DIGITAL_CULTURE = {
    "meme_spread": {
        "expr": xi**2/(2*m) - alpha*sp.log(1 + sp.Abs(x)) + beta*x**2,
        "dim": 1,
        "category": "digital_culture",
        "description": "Meme virality — logarithmic resistance to novelty saturation.",
    },
    "algorithmic_bias": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.tanh(x)*sp.tanh(y),
        "dim": 2,
        "category": "digital_culture",
        "description": "Feedback loop in recommendation systems — polarization attractor.",
    },
    "attention_economy": {
        "expr": xi**2/2 - V0*sp.exp(-alpha*x**2),
        "dim": 1,
        "category": "digital_culture",
        "description": "Attention as scarce resource — Gaussian capture by content.",
    },
    "digital_echo_chamber": {
        "expr": (xi**2 + eta**2)/2 - alpha*x*y + beta*(x**2 + y**2),
        "dim": 2,
        "category": "digital_culture",
        "description": "Echo chamber formation — alignment reinforced by platform design.",
    },
}

# =====================================================================
# 90. URBAN MYTH & FOLKLORE
# =====================================================================
H_FOLKLORE = {
    "narrative_diffusion": {
        "expr": xi**2/(2*m) - alpha*sp.log(sp.sqrt(x**2 + eps)),
        "dim": 1,
        "category": "folklore",
        "description": "Myth propagation — inverse-square attenuation with distance.",
    },
    "rumor_dynamics": {
        "expr": xi**2/2 + V0*x*(1 - x)*(x - beta),
        "dim": 1,
        "category": "folklore",
        "description": "Rumor spread — bistable potential between belief and skepticism.",
    },
    "archetype_potential": {
        "expr": xi**2/2 + alpha*(1 - sp.cos(2*sp.pi*x)),
        "dim": 1,
        "category": "folklore",
        "description": "Jungian archetype cycle — periodic recurrence in cultural narratives.",
    },
    "legend_persistence": {
        "expr": (xi**2 + eta**2)/2 + V0*sp.exp(-alpha*(x**2 + y**2)),
        "dim": 2,
        "category": "folklore",
        "description": "Urban legend as localized attractor — spatial-temporal memory kernel.",
    },
}

# =====================================================================
# 91. PERFUMERY
# =====================================================================
H_PERFUMERY = {
    "scent_harmony": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "perfumery",
        "description": "Olfactory consonance — energy minimized at balanced note ratios.",
    },
    "volatility_gradient": {
        "expr": xi**2/2 + alpha*sp.exp(-beta*x),
        "dim": 1,
        "category": "perfumery",
        "description": "Top-middle-base note volatility — exponential decay of scent intensity.",
    },
    "fragrance_accord": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.exp(-alpha*x**2) + sp.exp(-beta*y**2) + gamma*sp.exp(-delta*(x-y)**2)),
        "dim": 2,
        "category": "perfumery",
        "description": "Perfume accord — blend of Gaussian scent profiles with interaction term.",
    },
    "olfactory_adaptation": {
        "expr": xi**2/2 - alpha*sp.log(1 + x) + beta*x**2,
        "dim": 1,
        "category": "perfumery",
        "description": "Nose fatigue — logarithmic desensitization to persistent odorants.",
    },
}

# =====================================================================
# 92. DREAM DYNAMICS
# =====================================================================
H_DREAM = {
    "rem_cycle": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/90)),
        "dim": 1,
        "category": "dream",
        "description": "REM sleep cycle — 90-minute ultradian rhythm as periodic potential.",
    },
    "latent_narrative": {
        "expr": (xi**2 + eta**2)/2 + alpha*sp.exp(-beta*(x**2 + y**2)) * sp.cos(gamma*x),
        "dim": 2,
        "category": "dream",
        "description": "Latent narrative flow — associative memory landscape with modulation.",
    },
    "dream_instability": {
        "expr": xi**2/2 + alpha*x**2 - beta*x**4,
        "dim": 1,
        "category": "dream",
        "description": "Bistable dream state — abrupt transitions between narrative modes.",
    },
    "hypnagogic_potential": {
        "expr": xi**2/2 - alpha*sp.log(sp.cosh(x)) + beta*sp.sin(gamma*x),
        "dim": 1,
        "category": "dream",
        "description": "Hypnagogic state — noise-driven symbolic emergence at sleep onset.",
    },
}

# =====================================================================
# 93. GARDENING
# =====================================================================
H_GARDENING = {
    "growth_rhythm": {
        "expr": xi**2/2 + alpha*x**2*sp.exp(-beta*x),
        "dim": 1,
        "category": "gardening",
        "description": "Plant growth rhythm — sigmoidal biomass accumulation with senescence.",
    },
    "companion_planting": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.exp(-beta*(x - y)**2) + gamma*(x**2 + y**2),
        "dim": 2,
        "category": "gardening",
        "description": "Companion planting — mutualistic attraction between crop species.",
    },
    "phototropism_potential": {
        "expr": (xi**2 + eta**2)/2 - V0*sp.exp(-alpha*(x - beta)**2),
        "dim": 2,
        "category": "gardening",
        "description": "Phototropism — directional growth toward light source at x=β.",
    },
    "seasonal_cycle": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/365)),
        "dim": 1,
        "category": "gardening",
        "description": "Annual seasonal cycle — planting/harvesting rhythm over 365 days.",
    },
}

# =====================================================================
# 94. TYPOGRAPHY
# =====================================================================
H_TYPOGRAPHY = {
    "visual_tension": {
        "expr": (xi**2 + eta**2)/2 + alpha*(1 - sp.cos(x - y)),
        "dim": 2,
        "category": "typography",
        "description": "Visual tension between glyph positions — harmonic alignment principle.",
    },
    "glyph_rhythm": {
        "expr": xi**2/2 + V0*sp.sin(alpha*x)*sp.sin(beta*x),
        "dim": 1,
        "category": "typography",
        "description": "Rhythm of glyph spacing — beat frequency in text layout.",
    },
    "kerning_potential": {
        "expr": xi**2/2 + alpha/(sp.Abs(x) + eps) - beta*sp.exp(-gamma*sp.Abs(x)),
        "dim": 1,
        "category": "typography",
        "description": "Kerning dynamics — repulsion at close spacing, attraction at medium range.",
    },
    "typographic_balance": {
        "expr": (xi**2 + eta**2)/2 + alpha*x**2 + beta*y**2 - gamma*x*y,
        "dim": 2,
        "category": "typography",
        "description": "Page layout balance — weighted composition of type elements.",
    },
}

# =====================================================================
# 95. CEREMONY
# =====================================================================
H_CEREMONY = {
    "ritual_timing": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/T)),
        "dim": 1,
        "category": "ceremony",
        "description": "Ritual timing — periodic structure of ceremonial acts (T = cycle length).",
    },
    "symbolic_energy": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.log(sp.sqrt(x**2 + y**2) + eps),
        "dim": 2,
        "category": "ceremony",
        "description": "Symbolic energy — focus toward ritual center (e.g., altar, fire).",
    },
    "liminal_transition": {
        "expr": xi**2/2 + alpha*sp.tanh(beta*x),
        "dim": 1,
        "category": "ceremony",
        "description": "Liminal phase — smooth transition between social states (van Gennep).",
    },
    "communal_synchrony": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.cos(x - y),
        "dim": 2,
        "category": "ceremony",
        "description": "Communal synchrony — phase alignment in group ritual (e.g., chanting, dance).",
    },
}

# =====================================================================
# 59. MYTHOPOETICS
# =====================================================================
H_MYTHOPOETICS = {
    "hero_journey": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/3)),
        "dim": 1,
        "category": "mythopoetics",
        "description": "Hero's journey — three-act structure as periodic potential.",
    },
    "threshold_crossing": {
        "expr": xi**2/2 - alpha*sp.log(sp.Abs(x) + eps) + beta*x**2,
        "dim": 1,
        "category": "mythopoetics",
        "description": "Liminal threshold — logarithmic barrier between worlds.",
    },
    "archetypal_duality": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.cos(x - y),
        "dim": 2,
        "category": "mythopoetics",
        "description": "Shadow–self duality — phase alignment of opposing archetypes.",
    },
    "mythic_recurrence": {
        "expr": xi**2/2 + alpha*sp.sin(beta*x)*sp.sin(gamma*x),
        "dim": 1,
        "category": "mythopoetics",
        "description": "Eternal return — interference of nested mythic cycles.",
    },
}

# =====================================================================
# 60. CULINARY ARTS
# =====================================================================
H_CULINARY = {
    "plating_geometry": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2) + beta*(x*y)**2,
        "dim": 2,
        "category": "culinary",
        "description": "Plating symmetry — visual balance on the plate.",
    },
    "taste_sequencing": {
        "expr": xi**2/2 + V0*sp.exp(-alpha*x)*sp.sin(beta*x),
        "dim": 1,
        "category": "culinary",
        "description": "Taste sequence — transient flavor arc over time.",
    },
    "umami_resonance": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.exp(-beta*(x - y)**2),
        "dim": 2,
        "category": "culinary",
        "description": "Umami pairing — attraction between complementary tastes.",
    },
    "bitter_sweet_tension": {
        "expr": xi**2/2 + alpha*x**2 - beta*x**3,
        "dim": 1,
        "category": "culinary",
        "description": "Bitter-sweet contrast — cubic instability in flavor profile.",
    },
}

# =====================================================================
# 61. DANCE
# =====================================================================
H_DANCE = {
    "kinetic_flow": {
        "expr": (xi**2 + eta**2)/2 + alpha*(x**2 + y**2),
        "dim": 2,
        "category": "dance",
        "description": "Kinetic energy envelope — bounding ellipse of movement.",
    },
    "choreographic_potential": {
        "expr": (xi**2 + eta**2)/2 + V0*(sp.cos(x) + sp.cos(y) + sp.cos(x + y)),
        "dim": 2,
        "category": "dance",
        "description": "Choreographic lattice — spatial motifs on triangular grid.",
    },
    "rhythmic_tension": {
        "expr": xi**2/2 + alpha*(1 - sp.cos(2*sp.pi*x/T)),
        "dim": 1,
        "category": "dance",
        "description": "Metric pulse — beat-driven potential (T = bar length).",
    },
    "partner_synchrony": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.cos(x - y),
        "dim": 2,
        "category": "dance",
        "description": "Duet synchrony — phase locking in partner dance.",
    },
}

# =====================================================================
# 62. POETICS
# =====================================================================
H_POETICS = {
    "metrical_potential": {
        "expr": xi**2/2 + V0*(1 - sp.cos(2*sp.pi*x/5)),
        "dim": 1,
        "category": "poetics",
        "description": "Iambic pentameter — 5-beat periodic structure.",
    },
    "rhyme_attraction": {
        "expr": (xi**2 + eta**2)/2 - alpha*sp.exp(-beta*(x - y)**2),
        "dim": 2,
        "category": "poetics",
        "description": "Rhyme coupling — Gaussian attraction between phonetic endpoints.",
    },
    "caesura_tension": {
        "expr": xi**2/2 + alpha*sp.Abs(x - 0.5) - beta*sp.exp(-gamma*(x - 0.5)**2),
        "dim": 1,
        "category": "poetics",
        "description": "Caesura break — linear tension with localized relaxation.",
    },
    "enjambment_flow": {
        "expr": xi**2/2 - alpha*sp.log(sp.Abs(x - 1) + eps),
        "dim": 1,
        "category": "poetics",
        "description": "Enjambment — logarithmic pull across line boundary at x=1.",
    },
}

# =====================================================================
# 100. METAPHYSICAL & SPECULATIVE DYNAMICS
# =====================================================================
H_METAPHYSICAL = {
    "observer_effect": {
        "expr": xi**2/(2*m) + V0*x**2/2 + alpha*sp.Abs(xi)*sp.Abs(x),
        "dim": 1,
        "category": "metaphysical",
        "description": "Observer effect — measurement coupling between position and momentum.",
    },
    "arrow_of_time": {
        "expr": xi**2/(2*m) + k*x**2/2 + beta*sp.exp(-gamma*sp.Abs(xi)),
        "dim": 1,
        "category": "metaphysical",
        "description": "Thermodynamic arrow — irreversible friction in phase space.",
    },
    "platonian_form": {
        "expr": (xi**2 + eta**2)/2 + V0*(x**2 + y**2 - 1)**2 + alpha*(x**2 - y**2)**2,
        "dim": 2,
        "category": "metaphysical",
        "description": "Platonic ideal — perfect symmetry (square + circle) as attractor.",
    },
    "consciousness_potential": {
        "expr": xi**2/2 - alpha*sp.log(sp.cosh(x)) + beta*sp.sin(gamma*x),
        "dim": 1,
        "category": "metaphysical",
        "description": "Integrated information analog — bistable awareness landscape.",
    },
    "void_dynamics": {
        "expr": sp.Abs(xi)**alpha + eps*sp.log(sp.Abs(x) + eps),
        "dim": 1,
        "category": "metaphysical",
        "description": "Dynamics of nothingness — minimal structure emerging from noise.",
    },
}

# =====================================================================
# Merge all families
# =====================================================================
CATALOG = {}
for d in [
    H_INTEGRABLE, H_CHAOTIC, H_MAGNETIC, H_OPTICAL, H_RELATIVISTIC,
    H_POTENTIALS, H_GEOMETRIC, H_QUANTUM, H_ASTROPHYSICS, H_LATTICE,
    H_DISSIPATIVE, H_BIOPHYSICS, H_PLASMA, H_ACCELERATOR, H_EXOTIC,
    H_CLASSICAL_EXTENDED, H_TOPOLOGICAL, H_NONLINEAR_OPTICS, H_SPIN_SYSTEMS,
    H_REACTION_DIFFUSION, H_ELASTICITY, H_STATISTICAL, H_NEUROSCIENCE,
    H_ECONOPHYSICS, H_QFT, H_MATHEMATICAL, H_COSMOLOGY, H_TURBULENCE,
    H_GRANULAR, H_ACTIVE_MATTER, H_METAMATERIALS, H_QUANTUM_INFO, H_GEOPHYSICS,
    H_CLIMATE, H_CAVITY_QED, H_DEFECTS, H_ULTRACOLD, H_STOCHASTIC,
    H_STRING_THEORY, H_PARTICLE_PHYSICS, H_QUANTUM_GRAVITY,
    H_INTEGRABLE_ADVANCED, H_NON_EQUILIBRIUM, H_TWISTOR, H_SUPERSYMMETRY,
    H_DARK_SECTOR, H_NEUTRINO, H_EXOTIC_MATTER, H_QUANTUM_INFO_ADVANCED,
    H_PURE_MATH, H_BSM, H_BLACK_HOLES, H_FIELD_THEORY_PROPER,
    H_CONTROL_THEORY, H_ACOUSTICS, H_NETWORK_DYNAMICS,
    H_SPIN_GLASS, H_MESOSCOPIC, H_POLYMERS, 
    H_TFT, H_GEOMETRIC_ADVANCED, H_SYMMETRY_REDUCED,
    H_QUANTUM_TOPOLOGICAL_EXTENDED, H_CONTINUUM_SOLITONS,
    H_STOCHASTIC_ADVANCED, H_MULTI_SCALE_CHAOS, H_MODERN_EXTENSIONS,
    H_GAME_DYNAMICS, H_OPTIMIZATION, H_QUANT_FINANCE,
    H_SYMBOLIC_COMPUTATION, H_GENERATIVE_DESIGN,
    H_EPIDEMIOLOGY, H_LINGUISTICS, H_ECOLOGY, H_INFERENCE,
    H_URBAN, H_COGNITIVE, H_LEGAL, H_ART_MUSIC,
    H_EDUCATION, H_RELIGION, H_SPORTS, H_AGRICULTURE,
    H_PUBLIC_HEALTH, H_ARCHITECTURE, H_CUISINE, H_FASHION,
    H_WELLNESS, H_DIGITAL_CULTURE, H_FOLKLORE, H_PERFUMERY,
    H_DREAM, H_GARDENING, H_TYPOGRAPHY, H_CEREMONY,
    H_MYTHOPOETICS, H_CULINARY, H_DANCE, H_POETICS, 
    H_METAPHYSICAL
]:
    CATALOG.update(d)

# =====================================================================
# Utility functions
# =====================================================================
def get_hamiltonian(name: str):
    """
    Return Hamiltonian expression, variables, and metadata.

    Parameters
    ----------
    name : str
        Key identifier for the Hamiltonian.

    Returns
    -------
    H : sympy.Expr
        The Hamiltonian expression.
    vars : tuple
        Variables (x, xi) for 1D or (x, y, xi, eta) for 2D.
    info : dict
        Metadata including dimension, category, and description.

    Example
    -------
    >>> H, vars, meta = get_hamiltonian("henon_heiles")
    >>> print(meta["description"])
    Hénon–Heiles: benchmark for mixed regular/chaotic motion.
    """
    if name not in CATALOG:
        available = list(CATALOG.keys())[:10]
        raise KeyError(
            f"Unknown Hamiltonian '{name}'.\n"
            f"Available (first 10): {available}\n"
            f"Use list_hamiltonians() to see all {len(CATALOG)} entries."
        )
    info = CATALOG[name]
    H = info["expr"]
    dim = info["dim"]
    vars = (x, xi) if dim == 1 else (x, y, xi, eta)
    return H, vars, info

def list_categories():
    """
    List all categories and their counts.

    Returns
    -------
    dict
        Dictionary mapping category names to counts.
    """
    c = Counter([v["category"] for v in CATALOG.values()])
    return dict(c)

def list_hamiltonians(category=None, dim=None):
    """
    List Hamiltonian names, optionally filtered by category or dimension.

    Parameters
    ----------
    category : str, optional
        Filter by category (e.g., 'chaotic', 'integrable').
    dim : int, optional
        Filter by dimension (1 or 2).

    Returns
    -------
    list
        List of Hamiltonian names matching the criteria.

    Example
    -------
    >>> list_hamiltonians(category='chaotic')
    ['henon_heiles', 'quartic_coupled', ...]
    >>> list_hamiltonians(dim=1)
    ['free_particle', 'harmonic_oscillator', ...]
    """
    result = []
    for name, info in CATALOG.items():
        if category and info["category"] != category:
            continue
        if dim and info["dim"] != dim:
            continue
        result.append(name)
    return sorted(result)

def search_hamiltonians(keyword: str):
    """
    Search for Hamiltonians by keyword in name or description.

    Parameters
    ----------
    keyword : str
        Search term (case-insensitive).

    Returns
    -------
    list
        List of matching Hamiltonian names.

    Example
    -------
    >>> search_hamiltonians('pendulum')
    ['double_pendulum_reduced', 'driven_pendulum', 'spherical_pendulum', ...]
    """
    keyword = keyword.lower()
    result = []
    for name, info in CATALOG.items():
        if keyword in name.lower() or keyword in info["description"].lower():
            result.append(name)
    return sorted(result)

def print_hamiltonian_info(name: str):
    """
    Print detailed information about a specific Hamiltonian.

    Parameters
    ----------
    name : str
        Hamiltonian identifier.
    """
    H, vars, info = get_hamiltonian(name)
    print(f"\n{'='*70}")
    print(f"Hamiltonian: {name}")
    print(f"{'='*70}")
    print(f"Category:    {info['category']}")
    print(f"Dimension:   {info['dim']}D")
    print(f"Variables:   {vars}")
    print(f"\nDescription:\n  {info['description']}")
    print(f"\nExpression:\n  H = {H}")
    print(f"{'='*70}\n")

def get_catalog_summary():
    """
    Return a formatted summary of the entire catalog.

    Returns
    -------
    str
        Multi-line summary with statistics.

    Note: use `print(get_catalog_summary())`
    """
    total = len(CATALOG)
    categories = list_categories()
    dim_1 = len([v for v in CATALOG.values() if v["dim"] == 1])
    dim_2 = len([v for v in CATALOG.values() if v["dim"] == 2])

    summary = [
        "=" * 70,
        "HAMILTONIAN CATALOG SUMMARY",
        "=" * 70,
        f"Total Hamiltonians: {total}",
        f"  - 1D systems: {dim_1}",
        f"  - 2D systems: {dim_2}",
        "",
        "Categories:",
    ]

    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        summary.append(f"  {cat:20s} : {count:3d}")

    summary.append("=" * 70)
    return "\n".join(summary)

def get_hamiltonians_by_keywords(*keywords):
    """
    Multi-keyword search with AND operator.
    Example
    -------
    >>> get_hamiltonians_by_keywords('quantum', 'oscillator')
    """
    results = []
    for name, info in CATALOG.items():
        text = (name + ' ' + info['description']).lower()
        if all(kw.lower() in text for kw in keywords):
            results.append(name)
    return sorted(results)

def get_tree():
    """
    Returns a hierarchical tree of categories reflecting the full scope
    of the extended Hamiltonian catalog (including physical, biological,
    social, cognitive, and speculative systems).

    Returns
    -------
    dict
        Tree structured by super-categories mapping to subcategories
        that actually appear in the catalog, with counts of Hamiltonians per subcategory.
    """
    # First, collect all actual categories used and count Hamiltonians per category
    from collections import Counter
    category_counts = Counter(v["category"] for v in CATALOG.values())
    all_categories = set(category_counts.keys())

    tree = {
        # ────────────────────────────────
        # PHYSICAL SCIENCES
        # ────────────────────────────────
        "Classical & Celestial Mechanics": [
            "integrable", "chaotic", "nonlinear", "classical", "integrable_advanced",
            "astrophysics", "geophysics", "climate"
        ],
        "Quantum & Atomic Physics": [
            "quantum", "atomic", "molecular", "nuclear", "ultracold", "mesoscopic",
            "quantum_topological_extended", "semiclassical"
        ],
        "Field Theory & High-Energy Physics": [
            "qft", "particle_physics", "string_theory", "quantum_gravity",
            "supersymmetry", "bsm", "dark_sector", "neutrino", "exotic_matter",
            "field_theory"
        ],
        "Condensed Matter & Materials": [
            "lattice", "spin_systems", "spin_glass", "defects", "metamaterials",
            "topological", "quantum_info", "quantum_info_advanced", "polymers"
        ],
        "Electromagnetism & Optics": [
            "magnetic", "optical", "plasma", "cavity_qed", "nonlinear_optics",
            "acoustics", "rotating"
        ],
        "Relativity & Gravitation": [
            "relativistic", "black_holes", "cosmology", "geometric", "geometric_advanced"
        ],
        "Statistical & Non-Equilibrium Physics": [
            "statistical", "stochastic", "stochastic_advanced", "dissipative",
            "non_equilibrium", "reaction_diffusion", "turbulence"
        ],
        "Fluids, Soft Matter & Active Systems": [
            "fluid", "granular", "active_matter", "elasticity"
        ],
        "Solitons & Nonlinear Waves": [
            "continuum_solitons", "multi_scale_chaos"
        ],

        # ────────────────────────────────
        # APPLIED & INTERDISCIPLINARY
        # ────────────────────────────────
        "Biophysics & Life Sciences": [
            "biophysics", "neuroscience", "epidemiology", "public_health", "ecology"
        ],
        "Engineering & Technology": [
            "accelerator", "control_theory", "optics", "acoustics"
        ],
        "Earth & Environmental Systems": [
            "geophysics", "climate", "agriculture", "urban"
        ],

        # ────────────────────────────────
        # INFORMATION, COGNITION & SOCIETY
        # ────────────────────────────────
        "Information & Computation": [
            "quantum_info", "quantum_info_advanced", "symbolic", "optimization",
            "inference", "network_dynamics", "modern_extensions"
        ],
        "Cognitive & Psychological Dynamics": [
            "cognitive", "wellness", "dream", "neuroscience", "education"
        ],
        "Social & Cultural Systems": [
            "econophysics", "game_dynamics", "linguistics", "religion", "folklore",
            "digital_culture", "legal", "sports", "urban", "quant_finance"
        ],

        # ────────────────────────────────
        # CREATIVE & AESTHETIC DOMAINS
        # ────────────────────────────────
        "Aesthetic & Design Domains": [
            "art_music", "typography", "architecture", "perfumery", "fashion", 
            "cuisine", "culinary", "generative", "dance", "poetics"
        ],
        "Cultural & Symbolic Practices": [
            "ceremony", "mythopoetics", "folklore"
        ],

        # ────────────────────────────────
        # WELLNESS & LIFESTYLE
        # ────────────────────────────────
        "Health & Wellness": [
            "wellness", "dream", "gardening"
        ],

        # ────────────────────────────────
        # PURE & ADVANCED MATHEMATICS
        # ────────────────────────────────
        "Mathematical Structures": [
            "pure_math", "mathematical", "twistor", "tft", "symmetry_reduced",
            "exotic"
        ],

        # ────────────────────────────────
        # SPECULATIVE & FRONTIER DOMAINS
        # ────────────────────────────────
        "Metaphysical & Speculative": [
            "metaphysical", "exotic"
        ],
    }

    # Replace each subcategory with its actual Hamiltonian count, if present
    result = {}
    for super_cat, subcats in tree.items():
        filtered = {
            cat: category_counts[cat]
            for cat in subcats
            if cat in all_categories and category_counts[cat] > 0
        }
        if filtered:
            result[super_cat] = filtered

    return result

def export_latex_table(category=None, filename='hamiltonians.tex'):
    """
    Exports a LaTeX table of Hamiltonians.
    Parameters
    ----------
    category : str, optional
        Category to export (all if None).
    filename : str
        Output file name.
    """
    import sympy as sp

    # List of Hamiltonians to export
    hamiltonians = list_hamiltonians(category=category) if category else list(CATALOG.keys())

    # LaTeX table header
    lines = [
        r"\begin{longtable}{|l|c|p{8cm}|}",
        r"\hline",
        r"\textbf{Name} & \textbf{Dim} & \textbf{Hamiltonian} \\",
        r"\hline",
        r"\endfirsthead",
        r"\hline",
        r"\textbf{Name} & \textbf{Dim} & \textbf{Hamiltonian} \\",
        r"\hline",
        r"\endhead",
        r"\hline",
        r"\endfoot",
    ]

    # Add rows for each Hamiltonian
    for name in hamiltonians:
        info = CATALOG[name]
        H_latex = sp.latex(info['expr'])
        dim = info['dim']
        name_latex = name.replace('_', r'\_')
        lines.append(f"{name_latex} & {dim}D & ${H_latex}$" + r" \\" + "\n")
        lines.append(r"\hline")

    # End of the table
    lines.append(r"\end{longtable}")

    # Write to file
    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))
        print(f"Exported {len(hamiltonians)} Hamiltonians to {filename}")
    except IOError as e:
        print(f"Error writing to file {filename}: {e}")

def get_dimensional_analysis(name: str):
    """
    Basic dimensional analysis of a Hamiltonian.
    Parameters
    ----------
    name : str
        Name of the Hamiltonian.
    Returns
    -------
    dict
        Information about structural properties.
    """
    H, vars, info = get_hamiltonian(name)
    terms = H.as_ordered_terms()
    analysis = {
        'name': name,
        'dimension': info['dim'],
        'num_terms': len(terms),
        'polynomial_degree': 0,
        'has_trigonometric': False,
        'has_exponential': False,
        'has_logarithm': False,
        'has_sqrt': False,
        'has_abs': False,
        'has_rational': False,
        'complexity_score': 0
    }
    H_str = str(H)
    analysis['has_trigonometric'] = any(f in H_str for f in ['sin', 'cos', 'tan', 'cot', 'sec', 'csc'])
    analysis['has_exponential'] = 'exp' in H_str or '**' in H_str  # crude but effective
    analysis['has_logarithm'] = 'log' in H_str
    analysis['has_sqrt'] = 'sqrt' in H_str
    analysis['has_abs'] = 'Abs' in H_str or 'abs(' in H_str.lower()
    analysis['has_rational'] = any(op in H_str for op in ['/ ', '/(', '/x', '/y'])
    
    # Safely estimate polynomial degree
    for var in vars:
        for term in terms:
            if term.has(var):
                try:
                    deg = sp.degree(term, var)
                    if deg is not None and deg >= 0:
                        analysis['polynomial_degree'] = max(analysis['polynomial_degree'], int(deg))
                except (sp.PolynomialError, ValueError, TypeError, AttributeError):
                    # Non-polynomial term (e.g., log, sqrt, exp) — skip degree calculation
                    continue

    complexity = len(H_str)
    complexity += 10 * analysis['num_terms']
    complexity += 20 * int(analysis['has_trigonometric'])
    complexity += 20 * int(analysis['has_exponential'])
    complexity += 15 * int(analysis['has_logarithm'])
    complexity += 10 * int(analysis['has_sqrt'])
    complexity += 10 * int(analysis['has_abs'])
    analysis['complexity_score'] = complexity
    return analysis

def find_similar_hamiltonians(name: str, top_n=5):
    """
    Finds similar Hamiltonians by structural analysis.
    Parameters
    ----------
    name : str
        Name of the reference Hamiltonian.
    top_n : int
        Number of results to return.
    Returns
    -------
    list
        List of tuples (name, similarity score).
    """
    ref_analysis = get_dimensional_analysis(name)
    ref_info = CATALOG[name]
    similarities = []
    for other_name in CATALOG:
        if other_name == name:
            continue
        other_analysis = get_dimensional_analysis(other_name)
        other_info = CATALOG[other_name]
        score = 0
        if ref_info['dim'] == other_info['dim']:
            score += 30
        if ref_info['category'] == other_info['category']:
            score += 40
        if ref_analysis['has_trigonometric'] == other_analysis['has_trigonometric']:
            score += 10
        if ref_analysis['has_exponential'] == other_analysis['has_exponential']:
            score += 10
        if ref_analysis['has_logarithm'] == other_analysis['has_logarithm']:
            score += 10
        term_diff = abs(ref_analysis['num_terms'] - other_analysis['num_terms'])
        score += max(0, 10 - term_diff)
        similarities.append((other_name, score))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]

def validate_hamiltonians():
    """
    Validates all Hamiltonians to detect common errors.
    Returns
    -------
    dict
        Validation report with warnings and errors.
    """
    report = {
        'valid': [],
        'warnings': [],
        'errors': [],
        'suspicious': []
    }
    for name, info in CATALOG.items():
        H = info['expr']
        H_str = str(H)
        issues = []
        if 'Derivative' in H_str and 'Derivative(x, x)' in H_str:
            issues.append("Contains Derivative(x,x) which equals 1")
        if not any(var in H_str for var in ['x', 'y', 'xi', 'eta']):
            issues.append("No dynamical variables found")
        if ('1/x' in H_str or '1/y' in H_str) and 'eps' not in H_str:
            issues.append("Division by coordinate without regularization")
        if 'sqrt' in H_str and '-' in H_str:
            issues.append("Potential sqrt of negative quantity")
        if 'log' in H_str and not 'Abs' in H_str and 'eps' not in H_str:
            issues.append("Logarithm without absolute value or regularization")
        if len(H_str) > 500:
            issues.append(f"Very complex expression (length: {len(H_str)})")
        if issues:
            report['warnings'].append({
                'name': name,
                'issues': issues,
                'expression': H_str[:100] + ' ...' if len(H_str) > 100 else H_str
            })
        else:
            report['valid'].append(name)
    return report

def batch_export_hamiltonians(output_dir='hamiltonians_export', formats=['json', 'yaml', 'csv']):
    """
    Exports the entire catalog in multiple formats.
    Parameters
    ----------
    output_dir : str
        Output directory.
    formats : list
        Desired formats: 'json', 'yaml', 'csv', 'markdown'.
    """
    os.makedirs(output_dir, exist_ok=True)
    if 'json' in formats:
        catalog_json = {}
        for name, info in CATALOG.items():
            catalog_json[name] = {
                'expression': str(info['expr']),
                'dimension': info['dim'],
                'category': info['category'],
                'description': info['description']
            }
        with open(f'{output_dir}/catalog.json', 'w') as f:
            json.dump(catalog_json, f, indent=2)
        print(f"✓ Exported to {output_dir}/catalog.json")
    if 'yaml' in formats:
        try:
            import yaml
            with open(f'{output_dir}/catalog.yaml', 'w') as f:
                yaml.dump(catalog_json, f, default_flow_style=False)
            print(f"✓ Exported to {output_dir}/catalog.yaml")
        except ImportError:
            print("✗ YAML export requires PyYAML package")
    if 'csv' in formats:
        with open(f'{output_dir}/catalog.csv', 'w') as f:
            f.write("Name,Dimension,Category,Description\n")
            for name, info in CATALOG.items():
                desc = info['description'].replace(',', ';')
                f.write(f'{name},{info["dim"]},{info["category"]},"{desc}"\n')
        print(f"✓ Exported to {output_dir}/catalog.csv")
    if 'markdown' in formats:
        with open(f'{output_dir}/catalog.md', 'w') as f:
            f.write("# Hamiltonian Catalog\n\n")
            f.write(f"**Total Systems**: {len(CATALOG)}\n\n")
            for category in sorted(set(info['category'] for info in CATALOG.values())):
                f.write(f"\n## {category.replace('_', ' ').title()}\n\n")
                hamiltonians = [name for name, info in CATALOG.items() if info['category'] == category]
                for name in sorted(hamiltonians):
                    info = CATALOG[name]
                    f.write(f"### {name}\n")
                    f.write(f"- **Dimension**: {info['dim']}D\n")
                    f.write(f"- **Description**: {info['description']}\n")
                    f.write(f"- **Expression**: `{info['expr']}`\n\n")
        print(f"✓ Exported to {output_dir}/catalog.md")
