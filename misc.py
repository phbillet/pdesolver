import numpy as np
from sympy import sympify
import matplotlib.pyplot as plt
from PIL import Image
import librosa, librosa.display
import soundfile as sf
import matplotlib.animation as animation
import subprocess
import os
import soundfile as sf
from sympy.core.function import AppliedUndef
from sympy import Function

# Miscellaneous functions and classes
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

def make_symbol(g=None, b=None, V=None):
    """
    Assemble a 2D psiOp symbol from:
    - g: a symmetric metric tensor g = [[g_xx, g_xy], [g_yx, g_yy]] (functions or strings)
    - b: a vector b = [b_x, b_y] (functions or strings)
    - V: a scalar potential V(x, y) (function or string)

    Returns a SymPy expression representing the symbol sigma(x, y, xi, eta).
    """
    terms = []

    # Metric term (quadratic)
    if g is not None:
        g_xx, g_xy = g[0]
        g_yx, g_yy = g[1]

        # Symmetrize manually
        terms.append(f"({g_xx})*xi**2")
        terms.append(f"({g_yy})*eta**2")
        sym_xy = f"0.5*(({g_xy}) + ({g_yx}))"
        terms.append(f"2*({sym_xy})*xi*eta")

    # Vector (torsion-like) term (linear)
    if b is not None:
        b_x, b_y = b
        terms.append(f"({b_x})*xi")
        terms.append(f"({b_y})*eta")

    # Scalar potential term
    if V is not None:
        terms.append(f"({V})")

    symbol_str = " + ".join(terms)
    return sympify(symbol_str)

# Sonification
def sonify_solution(u, Nt, Nx, Lt, Lx, method="pan", samplerate=44100, outfile="sonification.wav"):
    """
    Enhanced stereo sonification of a PDE solution with rich, percussive, and original sounds.
    
    Parameters
    ----------
    u : ndarray (Nt, Nx)
        Solution of the PDE.
    Nt, Nx : int
        Number of points in time and space.
    Lt, Lx : float
        Total length in time and in space.
    method : str
        "pan" (dynamic barycenter), "fft" (spatial modes), "events" (percussions)
    samplerate : int
        Audio sampling rate.
    outfile : str
        Name of the output WAV file.
    """
    # Grids
    t = np.linspace(0, Lt, Nt)
    x = np.linspace(-Lx/2, Lx/2, Nx)
    n_samples = int(Lt * samplerate)
    time_audio = np.linspace(0, Lt, n_samples)

    # Base signal: energy + gradient
    energy = np.mean(np.abs(u)**2, axis=1)
    grad = np.mean(np.abs(np.gradient(u, axis=1)), axis=1)
    signal_base = energy + 0.5*grad
    signal_base /= np.max(signal_base) + 1e-12
    base_signal = np.interp(np.linspace(0, Nt-1, n_samples), np.arange(Nt), signal_base)

    left, right = np.zeros_like(base_signal), np.zeros_like(base_signal)

    # --- PAN: dynamic barycenter with harmonics + vibrato ---
    if method == "pan":
        bary = (u**2 @ x) / (np.sum(u**2, axis=1) + 1e-12)
        bary /= np.max(np.abs(bary)) + 1e-12
        bary_interp = np.interp(np.linspace(0, Nt-1, n_samples), np.arange(Nt), bary)
        lfo = 0.2 * np.sin(2*np.pi*0.3*time_audio)  # vibrato
        pan = bary_interp + lfo
        for i, val in enumerate(base_signal):
            freq = 220 + 220 * (val)  # frequency mod by amplitude
            wave = val * (np.sin(2*np.pi*freq*time_audio[i]) + 0.5*np.sin(2*np.pi*1.5*freq*time_audio[i]))
            L = np.cos(np.pi/4*(pan[i]+1))
            R = np.sin(np.pi/4*(pan[i]+1))
            left[i], right[i] = L*wave, R*wave

    # --- FFT: spatial modes with sin + square waves ---
    elif method == "fft":
        Y = np.fft.fftshift(np.fft.fft(u, axis=1), axes=1)
        freqs_x = np.fft.fftshift(np.fft.fftfreq(Nx, d=(x[1]-x[0])))
        idx = np.argsort(np.mean(np.abs(Y), axis=0))[-5:]  # 5 dominant modes
        freqs_audio = np.linspace(220, 880, len(idx))
        for j, f_audio in zip(idx, freqs_audio):
            amp = np.abs(Y[:, j])
            amp /= np.max(amp) + 1e-12
            amp_interp = np.interp(np.linspace(0, Nt-1, n_samples), np.arange(Nt), amp)
            pan = np.sign(freqs_x[j])
            Lgain, Rgain = (0.6,0.4) if pan < 0 else (0.4,0.6)
            # wave = sin + square, modulated by amplitude
            wave = amp_interp * (np.sin(2*np.pi*f_audio*time_audio) +
                                 0.3*np.sign(np.sin(2*np.pi*f_audio*time_audio)))
            left += Lgain * wave
            right += Rgain * wave

    # --- EVENTS: percussive beeps with pitch + double hit ---
    elif method == "events":
        maxpos = np.argmax(u, axis=1)
        maxvals = np.max(u, axis=1)
        threshold = 0.5*np.max(maxvals)
        for i in range(Nt):
            if maxvals[i] > threshold:
                pos = (maxpos[i]-Nx/2)/(Nx/2)
                Lgain, Rgain = (1-pos)/2, (1+pos)/2
                center = int(i/Nt*n_samples)
                dur = int(0.07*samplerate)
                env = np.linspace(0,1,dur//2, endpoint=False)
                env = np.concatenate([env, env[::-1]])
                if len(env) < dur:
                    env = np.pad(env, (0, dur-len(env)), mode='edge')
                freq = 440 + 200*pos  # pitch varies with position
                beep = 0.5*np.sign(np.sin(2*np.pi*freq*np.arange(dur)/samplerate)) * env
                for dt in [-1,0,1]:  # double/triple hit
                    idx = center + dt*int(0.02*samplerate)
                    if 0 <= idx < n_samples-dur:
                        left[idx:idx+dur] += Lgain*beep
                        right[idx:idx+dur] += Rgain*beep

    # --- Auto-gain normalization by RMS ---
    stereo = np.vstack([left, right]).T
    rms = np.sqrt(np.mean(stereo**2))
    target_rms = 0.1
    if rms > 1e-12:
        stereo *= (target_rms / rms)

    # --- Peak normalization ---
    stereo /= np.max(np.abs(stereo)) + 1e-12

    # Write WAV
    sf.write(outfile, stereo, samplerate)
    print(f"Sonification '{method}' exported to {outfile}")


def make_video_with_sound(u, Lt, Lx, outfile="solution_with_sound.mp4", samplerate=44100):
    """
    Create a video of the solution u(t,x) with synchronized sound tracks.

    Parameters
    ----------
    u : ndarray (Nt, Nx)
        Solution of the PDE.
    Nt, Nx : int
        Number of points in time and space.
    Lt, Lx : float
        Total length in time and space.
    outfile : str
        Name of the output MP4 file.
    samplerate : int
        Audio sampling rate.

    Returns
    -------
    str
        Path to the generated video file.
    """
    # 1. Generate matplotlib animation
    Nt = u.shape[0]
    Nx = u.shape[1]
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=2)
    x = np.linspace(-Lx/2, Lx/2, Nx)
    ax.set_xlim(-Lx/2, Lx/2)
    ax.set_ylim(np.min(u), np.max(u))
    def init():
        line.set_data([], [])
        return line,
    def update(frame):
        line.set_data(x, u[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=Nt, init_func=init, blit=True)
    video_file = "solution.mp4"
    ani.save(video_file, fps=Nt/Lt, dpi=150)
    plt.close(fig)

    # 2. Generate sound tracks
    sonify_solution(u, Nt, Nx, Lt, Lx, method="pan", outfile="u_pan.wav", samplerate=samplerate)
    sonify_solution(u, Nt, Nx, Lt, Lx, method="fft", outfile="u_fft.wav", samplerate=samplerate)
    sonify_solution(u, Nt, Nx, Lt, Lx, method="events", outfile="u_events.wav", samplerate=samplerate)

    # 3. Call ffmpeg to mix and merge
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_file,
        "-i", "u_pan.wav",
        "-i", "u_fft.wav",
        "-i", "u_events.wav",
        "-filter_complex", "amix=inputs=3:normalize=0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        outfile
    ]
    subprocess.run(cmd, check=True)

    # 4. Cleanup (optional)
    for f in ["u_pan.wav", "u_fft.wav", "u_events.wav", "solution.mp4"]:
        if os.path.exists(f):
            os.remove(f)

    return outfile


def image_to_sound(
    image_path,
    output_wav="son_from_image.wav",
    sr=22050,
    hop_length=256,
    n_iter=64,
    scale_log_freq=True,
    use_hsv=True,
    gain=5.0,
    show_plot=True
):
    """
    Convert an 2D image into stereo sound by interpreting it as a spectrogram.

    The input image is treated as a time–frequency representation:
    - The vertical axis corresponds to frequency bins.
    - The horizontal axis corresponds to time frames.
    - The image is split into two halves along the horizontal axis:
      the left half is converted into the LEFT audio channel,
      the right half into the RIGHT channel.

    Parameters
    ----------
    image_path : str
        Path to the input image file. Can be grayscale or RGB.
        If RGB and `use_hsv=True`, the HSV color space is used for mapping.
    output_wav : str, optional
        Filename of the output WAV file (default: "son_from_image.wav").
    sr : int, optional
        Target audio sampling rate in Hz (default: 22050).
        Lower values produce lower-pitched sounds.
    hop_length : int, optional
        Number of samples between successive STFT frames (default: 256).
        Smaller values produce a longer audio signal.
    n_iter : int, optional
        Number of Griffin–Lim iterations for phase reconstruction (default: 64).
    scale_log_freq : bool, optional
        If True, the frequency axis of the image is remapped to a logarithmic scale
        before reconstruction (default: True).
    use_hsv : bool, optional
        If True and the image is RGB, the HSV color model is used:
        - Hue → frequency mapping,
        - Saturation → timbre,
        - Value → amplitude.
        Otherwise, the grayscale intensity is used (default: True).
    gain : float, optional
        Global amplification factor for the spectrogram intensity (default: 5.0).
    show_plot : bool, optional
        If True, displays:
        - The original image (as interpreted spectrogram),
        - The reconstructed spectrograms of LEFT and RIGHT channels (default: True).

    Returns
    -------
    y_stereo : np.ndarray, shape (n_samples, 2)
        Stereo audio signal: left and right channels as columns.
    sr : int
        Sampling rate of the generated audio.

    Notes
    -----
    - The function always outputs a stereo WAV file, even for grayscale images.
    - The stereo split is based on the horizontal axis of the image:
      left side → left ear, right side → right ear.
    - Griffin–Lim is an iterative algorithm, so reconstruction is approximate.
    """
    from PIL import Image

    # ---------------------------
    # 1. Load the image
    # ---------------------------
    img = Image.open(image_path)
    if use_hsv and img.mode == "RGB":
        img_hsv = img.convert("HSV")
        H, S, V = [np.array(ch, dtype=np.float32) for ch in img_hsv.split()]
        H /= 255.0
        S /= 255.0
        V /= 255.0
        Z = V * (0.5 + 0.5 * S)  # intensity
    else:
        img_gray = img.convert("L")
        Z = np.array(img_gray, dtype=np.float32)
        Z /= Z.max()

    n_freqs, n_frames = Z.shape
    print(f"Image interpreted as: {n_freqs} frequencies × {n_frames} frames")

    if show_plot:
        plt.figure(figsize=(8, 4))
        plt.imshow(Z, aspect='auto', origin='lower', cmap='inferno')
        plt.title("Original image (interpreted as spectrogram)")
        plt.xlabel("Frames (x)")
        plt.ylabel("Frequency bins (y)")
        plt.colorbar()
        plt.show()

    # ---------------------------
    # 2. Amplification
    # ---------------------------
    S = Z * gain

    # ---------------------------
    # 3. Option log frequency
    # ---------------------------
    if scale_log_freq:
        n_bins = n_freqs
        log_y = np.geomspace(1, n_freqs, n_bins).astype(int) - 1
        S = S[log_y, :]

    # ---------------------------
    # 4. Split into left / right halves
    # ---------------------------
    mid = n_frames // 2
    S_left, S_right = S[:, :mid], S[:, mid:]

    def reconstruct(S_part):
        n_fft = 2 * (S_part.shape[0] - 1)
        n_freqs_target = 1 + n_fft // 2
        if S_part.shape[0] != n_freqs_target:
            S_resized = np.zeros((n_freqs_target, S_part.shape[1]), dtype=np.float32)
            for t in range(S_part.shape[1]):
                S_resized[:, t] = np.interp(
                    np.linspace(0, S_part.shape[0] - 1, n_freqs_target),
                    np.arange(S_part.shape[0]),
                    S_part[:, t]
                )
        else:
            S_resized = S_part
        y = librosa.griffinlim(S_resized, hop_length=hop_length, n_fft=n_fft,
                               win_length=n_fft, n_iter=n_iter)
        return y, n_fft

    y_left, n_fft = reconstruct(S_left)
    y_right, _ = reconstruct(S_right)

    # Align lengths
    L = min(len(y_left), len(y_right))
    y_left, y_right = y_left[:L], y_right[:L]

    # Stereo assembly
    y_stereo = np.stack([y_left, y_right], axis=-1)

    # ---------------------------
    # 5. Visualization: BOTH channels
    # ---------------------------
    if show_plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

        img_left = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_left, n_fft=n_fft, hop_length=hop_length)), ref=np.max
        )
        img_right = librosa.amplitude_to_db(
            np.abs(librosa.stft(y_right, n_fft=n_fft, hop_length=hop_length)), ref=np.max
        )

        librosa.display.specshow(
            img_left, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=axs[0]
        )
        axs[0].set_title("Spectrogram LEFT channel")
        fig.colorbar(axs[0].collections[0], ax=axs[0], format="%+2.0f dB")

        librosa.display.specshow(
            img_right, sr=sr, hop_length=hop_length, x_axis="time", y_axis="log", ax=axs[1]
        )
        axs[1].set_title("Spectrogram RIGHT channel")
        fig.colorbar(axs[1].collections[0], ax=axs[1], format="%+2.0f dB")

        plt.tight_layout()
        plt.show()

    # ---------------------------
    # 6. Save audio
    # ---------------------------
    sf.write(output_wav, y_stereo, sr)
    print(f"✅ Stereo audio saved: {output_wav} ({L/sr:.2f} seconds)")
    return y_stereo, sr



# Small symbol dictionnary

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


    

