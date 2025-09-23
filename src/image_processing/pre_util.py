import numpy as np
from scipy import signal
import torch
import torch.nn.functional as F
from numba import njit, cuda, float32, int32
import math
from scipy.interpolate import griddata

def compute_speckle_maps(A_pad: np.ndarray, sigma_c0: int, sigma_x: int, speckle_size_factor: float, half_pulse: int, device: torch.device = None):
    """
    Compute per-pixel speckle size (sigma_c) and radius maps using FFT-based autocorrelation.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Convert A_pad to torch tensor to a tensor on GPU
    if not torch.is_tensor(A_pad):
        A_pad = torch.tensor(A_pad, dtype=torch.float32, device=device)
    else:
        A_pad = A_pad.to(device, dtype=torch.float32)

    Nz = A_pad.shape[0] - 2 * sigma_c0
    Nx = A_pad.shape[1] - 2 * sigma_x
    H = 2 * sigma_c0 + 1
    W = 2 * sigma_x + 1

    # Unfold into patches: shape (1, 1, Nz*Nx, H*W)
    patches = A_pad.unsqueeze(0).unsqueeze(0)  # (1, 1, H_full, W_full)
    P = F.unfold(patches, kernel_size=(H, W)).squeeze(0).T  # (Nz*Nx, H*W)
    Npix = Nz * Nx

    # Normalize: zero mean and unit Frobenius norm
    P = P - P.mean(dim=1, keepdim=True)
    norms = torch.norm(P, dim=1, keepdim=True) + 1e-12
    P = P / norms
    P = P.view(Npix, H, W)

    # FFT, autocorrelation
    F_fft = torch.fft.fft2(P)
    M = torch.fft.ifft2(F_fft * torch.conj(F_fft)).real
    M = torch.fft.fftshift(M, dim=(-2, -1))  # (Npix, H, W)
    max_vals = M.view(Npix, -1).amax(dim=1, keepdim=True)
    masks = M > (0.5 * max_vals.view(-1, 1, 1))

    # Count nonzero rows and cols
    row_counts = masks.any(dim=2).sum(dim=1)  # (Npix,)
    col_counts = masks.any(dim=1).sum(dim=1)  # (Npix,)

    sigma_c = torch.minimum(torch.maximum(row_counts, col_counts), torch.tensor(sigma_c0, device=device))
    sigma_c = (sigma_c.float() * speckle_size_factor).round().int()
    sigma_c = torch.where(sigma_c < 1, torch.tensor(half_pulse, device=device), sigma_c)

    radius = torch.ceil(sigma_c.float() / 2).int()
    # reshape back to (Nz, Nx) but stay on gpu
    sigma_c = sigma_c.view(Nz, Nx).cpu().numpy()
    radius  = radius.view(Nz, Nx).cpu().numpy()
    return sigma_c, radius

def pol2cart(theta, rho):
    return rho * np.sin(theta), rho * np.cos(theta)

@cuda.jit(fastmath = True)
def _srbf_cuda_kernel(A_pad, xx_pad, zz_pad, sigma_c_map, radius_map, B_out, sigma_c0, sigma_x, lat_support):
    """
    Each thread computes one output pixel.
    """
    # compute our pixel coords
    iz, ix = cuda.grid(2)
    Nz, Nx = sigma_c_map.shape

    if iz < Nz and ix < Nx:
        # remap to padded coords
        zc = iz + sigma_c0
        xc = ix + sigma_x

        sigma_c = sigma_c_map[iz, ix]
        r       = radius_map[iz, ix]
        Lz      = 2 * r + 1
        Lx      = 2 * lat_support + 1
        L       = Lz * Lx

        # center coordinates
        x0 = xx_pad[zc, xc]
        z0 = zz_pad[zc, xc]

        # bounding box of the patch
        z1_min = zc - r
        z1_max = zc + r + 1
        x1_min = xc - lat_support
        x1_max = xc + lat_support + 1

        weighted_sum = float32(0.0)
        norm_sum     = float32(0.0)

        # loop over the local window
        inv_two_sig2 = 1.0 / (2.0 * sigma_c * sigma_c + 1e-12)
        for zz in range(z1_min, z1_max):
            dz2 = zz_pad[zz, xc] - z0
            for xx in range(x1_min, x1_max):
                val = A_pad[zz, xx]

                # spatial weight
                dx2 = xx_pad[zz, xx] - x0
                w_sp = math.exp(- (dx2*dx2 + dz2*dz2) * inv_two_sig2)

                # speckle weight
                alpha2 = (val * val) / (2.0 * L + 1e-12)
                w_int = val / (alpha2 + 1e-12) * math.exp(-val*val/(2.0*alpha2 + 1e-12))

                w = w_sp * w_int

                weighted_sum += w * val
                norm_sum     += w

        out = weighted_sum / (norm_sum + 1e-12)
        if out < 0.0:
            out = 0.0

        B_out[iz, ix] = out

def _srbf(A_pad, xx_pad, zz_pad,
                   sigma_c_map, radius_map,
                   delta_angle,
                   sigma_c0, sigma_x,
                   lat_support,
                   threadsperblock=(16,16)):
    
    Nz, Nx = sigma_c_map.shape
    B_gpu = cuda.device_array((Nz, Nx), dtype=np.float32)

    # copy inputs to device
    A_d       = cuda.to_device(A_pad.astype(np.float32))
    xx_d      = cuda.to_device(xx_pad.astype(np.float32))
    zz_d      = cuda.to_device(zz_pad.astype(np.float32))
    sc_d      = cuda.to_device(sigma_c_map)
    r_d       = cuda.to_device(radius_map)

    # grid dimensions
    blockspergrid_x = (Nx  + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid_y = (Nz  + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid   = (blockspergrid_y, blockspergrid_x)

    # launch kernel
    _srbf_cuda_kernel[blockspergrid, threadsperblock](
        A_d, xx_d, zz_d,
        sc_d, r_d,
        B_gpu,
        sigma_c0, sigma_x,
        lat_support
    )

    # copy result back to host
    return B_gpu.copy_to_host()

def _interpolate_nans(arr):
    """
    Interpolate NaN values in a 2D numpy array using linear interpolation.
    """
    x, y = np.indices(arr.shape)
    valid_mask = ~np.isnan(arr)
    
    # Coordinates and values of known (non-NaN) points
    known_coords = np.column_stack((x[valid_mask], y[valid_mask]))
    known_values = arr[valid_mask]

    # Coordinates of NaNs
    nan_coords = np.column_stack((x[~valid_mask], y[~valid_mask]))
    interpolated_values = griddata(known_coords, known_values, nan_coords, method='linear')

    # Copy the array and insert interpolated values
    filled = arr.copy()
    filled[~valid_mask] = interpolated_values

    return np.nan_to_num(filled, nan=0.0)

def SRBF_OCT(A):
    A = signal.medfilt2d(A, kernel_size=5)
    Nz, Nx = A.shape

    # 2) Build polar‐to‐Cartesian coordinate grids
    angles = np.linspace(-195, 195, Nx) * np.pi/180
    delta_angle = angles[1] - angles[0]
    angle_grid  = np.tile(angles, (Nz,1))
    rho_grid    = np.tile((np.arange(Nz)+1)[:,None]*7.5e-6 + 1e-3, (1, Nx))
    zz, xx      = pol2cart(angle_grid, rho_grid)

    # 3) Pad arrays (replicate border)
    sigma_c0 = 3
    sigma_x  = 3
    pad = ((sigma_c0, sigma_c0), (sigma_x, sigma_x))
    A_pad  = np.pad(A,  pad, mode='edge')
    zz_pad = np.pad(zz, pad, mode='edge')
    xx_pad = np.pad(xx, pad, mode='edge')

    # 4) Algorithm parameters
    half_pulse = 2
    speckle_size_factor = 1
    lat_support = 3

    # 5) Pre‐compute per‐pixel speckle size via FFT
    sigma_c_map, radius_map = compute_speckle_maps(A_pad, sigma_c0, sigma_x, speckle_size_factor, half_pulse)

    # 6) Run the bilateral‐filter weighting on the GPU‐style via Numba
    B = _srbf(A_pad, xx_pad, zz_pad, sigma_c_map, radius_map, delta_angle, sigma_c0, sigma_x, lat_support)
    return _interpolate_nans(B)