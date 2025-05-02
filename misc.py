# Miscellaneous functions
import numpy as np
def gaussian_function_1D(x, center, sigma):
    A = 1 / np.sqrt(2 * np.pi * sigma**2)  # Amplitude so that the integral is equal to 1
    return A * np.exp(-((x - center)**2) / (2 * sigma**2))

def gaussian_function_2D(x, y, center, sigma):
    A = 1 / (2 * np.pi * sigma**2)  # Amplitude so that the integral is equal to 1
    center_x, center_y = center
    return A * np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2))

def ramp_function(x, y, point1, point2, direction='increasing'):
    """
    Creates a ramp (generalized Heaviside) function between two points.
    Args:
        x, y: meshgrid arrays
        point1: (x1, y1), first point on the ramp axis
        point2: (x2, y2), second point on the ramp axis
        direction: 'increasing' (from point1 to point2) or 'decreasing'
    Returns:
        A 2D array with values in [0, 1]
    """
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    norm2 = dx**2 + dy**2 + 1e-12  # Avoid division by zero

    # Projection (scalar parameter along the axis)
    s = ((x - x1) * dx + (y - y1) * dy) / norm2

    # Orientation
    if direction == 'increasing':
        ramp = np.clip(s, 0, 1)
    elif direction == 'decreasing':
        ramp = 1 - np.clip(s, 0, 1)
    else:
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    
    return ramp


def sigmoid_ramp(x, y, point1, point2, width=1.0, direction='increasing'):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    norm2 = dx**2 + dy**2 + 1e-12
    s = ((x - x1) * dx + (y - y1) * dy) / np.sqrt(norm2)
    if direction == 'decreasing':
        s = -s
    return 1 / (1 + np.exp(-s / width))

def tanh_ramp(x, y, point1, point2, width=1.0, direction='increasing'):
    x1, y1 = point1
    x2, y2 = point2
    dx = x2 - x1
    dy = y2 - y1
    norm2 = dx**2 + dy**2 + 1e-12
    s = ((x - x1) * dx + (y - y1) * dy) / np.sqrt(norm2)
    if direction == 'decreasing':
        s = -s
    return 0.5 * (1 + np.tanh(s / width))

def top_hat_band(x, y, x_min, x_max):
    return ((x >= x_min) & (x <= x_max)).astype(float)

def radial_gradient(x, y, center, radius, direction='increasing'):
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    s = r / radius
    if direction == 'increasing':
        return np.clip(s, 0, 1)
    else:
        return 1 - np.clip(s, 0, 1)

# Circle
circle_function = lambda x, y: (x - center_circle[0])**2 + (y - center_circle[1])**2 - radius_circle**2

# Ellipse
ellipse_function = lambda x, y: ((x - center_ellipse[0])**2 / semi_major_axis**2) + ((y - center_ellipse[1])**2 / semi_minor_axis**2) - 1

# Rectangle
rectangle_function = lambda x, y: (x - corner1_rectangle[0]) * (x - corner2_rectangle[0]) * (y - corner1_rectangle[1]) * (y - corner2_rectangle[1])

# Cross
cross_function = lambda x, y: min(abs(x - center_cross[0]) - width_cross, abs(y - center_cross[1]) - height_cross) + 2

operator_symbols = {
    "identity": {
        "physical": "u(x)",
        "fourier": "1",
        "equation": "Identity operator (leaves u unchanged)",
    },
    "first_derivative": {
        "physical": "∂u/∂x",
        "fourier": "I * kx",
        "equation": "First spatial derivative",
    },
    "second_derivative": {
        "physical": "∂²u/∂x²",
        "fourier": "-kx**2",
        "equation": "Second spatial derivative",
    },
    "third_derivative": {
        "physical": "∂³u/∂x³",
        "fourier": "-I * kx**3",
        "equation": "Third spatial derivative",
    },
    "fourth_derivative": {
        "physical": "∂⁴u/∂x⁴",
        "fourier": "kx**4",
        "equation": "Fourth spatial derivative",
    },
    "laplacian": {
        "physical": "∂²u/∂x²  (1D) or ∇²u = ∂²u/∂x² + ∂²u/∂y² (2D)",
        "fourier": "-kx**2  (1D)  or  -(kx**2 + ky**2) (2D)",
        "equation": "Laplacian operator",
    },
    "bilaplacian": {
        "physical": "∂⁴u/∂x⁴ (1D) or ∇⁴u (2D)",
        "fourier": "kx**4  (1D)  or  (kx**2 + ky**2)**2 (2D)",
        "equation": "Bilaplacian operator",
    },
    "mixed_derivative": {
        "physical": "∂²u/∂x∂y",
        "fourier": "I * kx * ky",
        "equation": "Mixed partial derivative",
    },
    "fractional_laplacian": {
        "physical": "(-Δ)^(α/2) u",
        "fourier": "abs(kx)**alpha (1D) or (kx**2 + ky**2)**(alpha/2) (2D)",
        "equation": "Fractional Laplacian operator",
    },
    "inverse_derivative": {
        "physical": "∫u dx",
        "fourier": "1 / (I * kx)",
        "equation": "Inverse derivative (antiderivative)",
    },
    "gaussian_filter": {
        "physical": "convolution with Gaussian kernel",
        "fourier": "exp(-sigma**2 * kx**2) (1D) or exp(-sigma**2 * (kx**2 + ky**2)) (2D)",
        "equation": "Gaussian smoothing filter",
    },
    "viscous_dissipation": {
        "physical": "ν ∇²u",
        "fourier": "-nu * kx**2  (1D)  or  -nu * (kx**2 + ky**2) (2D)",
        "equation": "Viscous dissipation term",
    },
    "linear_dispersion": {
        "physical": "α ∂³u/∂x³",
        "fourier": "alpha * kx**3",
        "equation": "Linear dispersion term",
    },
    "helmholtz_inverse": {
        "physical": "(1 - α∇²)⁻¹ u",
        "fourier": "1 / (1 + alpha * kx**2) (1D) or 1 / (1 + alpha * (kx**2 + ky**2)) (2D)",
        "equation": "Inverse Helmholtz operator",
    },
    "fractional_diffusion": {
        "physical": "-μ(-Δ)^(α/2) u",
        "fourier": "-mu * abs(kx)**alpha (1D) or -mu * (kx**2 + ky**2)**(alpha/2) (2D)",
        "equation": "Fractional diffusion operator",
    },
    "ginzburg_landau": {
        "physical": "-(1 + iβ) ∇²u",
        "fourier": "-(1 + I*beta) * kx**2  (1D)  or  -(1 + I*beta) * (kx**2 + ky**2) (2D)",
        "equation": "Ginzburg-Landau operator",
    },
    "schrodinger_dispersion": {
        "physical": "i ∂u/∂t = -∇²u",
        "fourier": "-kx**2  (1D)  or  -(kx**2 + ky**2) (2D)",
        "equation": "Schrödinger equation dispersion",
    },
    "helmholtz_operator": {
        "physical": "(λ² - ∇²)u",
        "fourier": "-kx**2 + lambda_**2 (1D)  or  -(kx**2 + ky**2) + lambda_**2 (2D)",
        "equation": "Helmholtz operator",
    },
    "green_operator": {
        "physical": "-∇⁻²u",
        "fourier": "-1 / kx**2 (1D)  or  -1 / (kx**2 + ky**2) (2D)",
        "equation": "Green's function operator",
    },
    "bessel_filter": {
        "physical": "(1 - ∇²)^(-α) u",
        "fourier": "(1 + kx**2)**(-alpha)  (1D)  or  (1 + kx**2 + ky**2)**(-alpha) (2D)",
        "equation": "Bessel regularization filter",
    },
    "poisson_kernel": {
        "physical": "Poisson kernel (boundary solution)",
        "fourier": "exp(-abs(kx) * y)  (1D)  or  exp(-sqrt(kx**2 + ky**2) * y) (2D)",
        "equation": "Poisson kernel in Fourier",
    },
    "hilbert_transform": {
        "physical": "Hilbert transform H[u]",
        "fourier": "-I * sign(kx)",
        "equation": "Hilbert transform operator",
    },
    "riesz_derivative": {
        "physical": "Riesz fractional derivative",
        "fourier": "-abs(kx)**alpha (1D)  or  -(kx**2 + ky**2)**(alpha/2) (2D)",
        "equation": "Riesz fractional derivative operator",
    },
    "convolution_box": {
        "physical": "convolution with box function (width h)",
        "fourier": "sinc(kx * h)",
        "equation": "Box convolution filter",
    },
    "anisotropic_diffusion": {
        "physical": "κₓ∂²u/∂x² + κᵧ∂²u/∂y²",
        "fourier": "-kx**2 * kappa_x  (1D)  or  -(kx**2 * kappa_x + ky**2 * kappa_y) (2D)",
        "equation": "Anisotropic diffusion operator",
    },
    "directional_derivative": {
        "physical": "aₓ∂u/∂x + aᵧ∂u/∂y",
        "fourier": "I * (a_x * kx + a_y * ky)",
        "equation": "Directional derivative",
    },
    "convection": {
        "physical": "c ∂u/∂x  (1D)  or  cₓ∂u/∂x + cᵧ∂u/∂y  (2D)",
        "fourier": "I * c * kx  (1D)  or  I * (c_x * kx + c_y * ky) (2D)",
        "equation": "Convection operator",
    },
    "telegraph_operator": {
        "physical": "∂²u/∂t² + a∂u/∂t + bu",
        "fourier": "-a*I*kx + b  (1D)  or  -a*I*(kx + ky) + b (2D)",
        "equation": "Telegraph operator",
    },
    "regularized_inverse_derivative": {
        "physical": "Regularized integral ∫u dx (avoiding singularity at kx=0)",
        "fourier": "1 / (I * (kx + eps))",
        "equation": "Regularized inverse derivative (integral operator with small epsilon shift)",
    },
    "hilbert_shifted": {
        "physical": "Hilbert transform with exponential regularization",
        "fourier": "1 / (I * (kx + I * eps))",
        "equation": "Hilbert transform regularized by shift (avoids singularity at kx=0)",
    },
    "convolution_general": {
        "physical": "convolution with arbitrary kernel f_kernel(x)",
        "fourier": "F[f_kernel](kx / (2*pi))  (1D)  or  F[f_kernel](kx / (2*pi)) * F[f_kernel](ky / (2*pi)) (2D)",
        "equation": "General convolution: u * f_kernel(x)",
    },
}

convolution_kernels = {
    "gaussian": {
        "physical": "1 / (sqrt(2 * pi) * sigma) * exp(-x**2 / (2 * sigma**2))",
        "fourier": "exp(-sigma**2 * kx**2)",  # In 2D: exp(-sigma**2 * (kx**2 + ky**2))
        "equation": "Gaussian smoothing kernel",
    },
    "box": {
        "physical": "1 / h  if abs(x) <= h/2  else 0",
        "fourier": "sinc(kx * h / 2)",  # In 2D: sinc(kx * h / 2) * sinc(ky * h / 2)
        "equation": "Box (rectangle) convolution filter",
    },
    "triangle": {
        "physical": "1 / h * (1 - abs(x) / h)  if abs(x) <= h  else 0",
        "fourier": "sinc(kx * h / 2)**2",
        "equation": "Triangle kernel (convolution of two box functions)",
    },
    "exponential": {
        "physical": "1 / (2 * a) * exp(-abs(x) / a)",
        "fourier": "1 / (1 + a**2 * kx**2)",
        "equation": "Exponential decay kernel (Poisson filter)",
    },
    "lorentzian": {
        "physical": "a / (pi * (x**2 + a**2))",
        "fourier": "exp(-a * abs(kx))",
        "equation": "Lorentzian (Cauchy) convolution kernel",
    },
}


    

