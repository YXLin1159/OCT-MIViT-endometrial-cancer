import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import cv2
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d, CubicSpline
from scipy import signal
from skimage.filters import meijering
import torch.nn.functional as F
from typing import List

def gaussian_kernel1d(sigma: float, radius: int, device):
    """
    Returns a 1-D Gaussian kernel of length (2*radius+1) on the given device.
    """
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma)**2)
    return g / g.sum()

def gaussian_blur_gpu(img: torch.Tensor, sigma: float):
    """
    img: (1,1,H,W) tensor on GPU
    sigma: float
    returns: (1,1,H,W) tensor on GPU
    2D Gaussian blur implemented via separable 1D convolutions.
    """
    device = img.device
    radius = int(3 * sigma)
    g1d = gaussian_kernel1d(sigma, radius, device)
    
    kx = g1d.view(1, 1, 1, -1)   # shape: (1,1,1,K)
    ky = g1d.view(1, 1, -1, 1)   # shape: (1,1,K,1)

    blurred = F.conv2d(img, kx, padding=(0, radius))
    blurred = F.conv2d(blurred, ky, padding=(radius, 0))
    return blurred

def multi_scale_dog(img_np: torch.Tensor, sigmas: List[float]):
    """
    img_np: H×W numpy array
    sigmas: list of floats
    returns: H×W numpy DoG max projection
    """
    # move to GPU and normalize
    t = torch.from_numpy(img_np.astype('float32')).unsqueeze(0).unsqueeze(0).cuda()
    t = (t - t.min()) / (t.max() - t.min() + 1e-12)

    dogs = []
    for s1, s2 in zip(sigmas, sigmas[1:]):
        g1 = gaussian_blur_gpu(t, s1)
        g2 = gaussian_blur_gpu(t, s2)
        dogs.append(g2 - g1)

    # stack along new dim 0 → (S,1,H,W), then max over S
    stack = torch.cat(dogs, dim=0)          # shape: (S,1,H,W)
    dog_ms = stack.max(dim=0).values        # shape: (1,H,W)
    dog_ms = (dog_ms - dog_ms.min()) / (dog_ms.max() - dog_ms.min() + 1e-12)

    return dog_ms.squeeze().cpu().numpy()

def multi_scale_dog_bright(img_np: torch.Tensor, sigmas):
    """
    img_np: H×W numpy array
    sigmas: list of floats
    returns: H×W numpy DoG max projection
    """
    # move to GPU and normalize
    t = torch.from_numpy(img_np.astype('float32')).unsqueeze(0).unsqueeze(0).cuda()
    t = (t - t.min()) / (t.max() - t.min() + 1e-12)

    dogs = []
    for s1, s2 in zip(sigmas, sigmas[1:]):
        g1 = gaussian_blur_gpu(t, s1)
        g2 = gaussian_blur_gpu(t, s2)
        dogs.append(g1 - g2)

    # stack along new dim 0 → (S,1,H,W), then max over S
    stack = torch.cat(dogs, dim=0)          # shape: (S,1,H,W)
    dog_ms = stack.max(dim=0).values        # shape: (1,H,W)
    dog_ms = (dog_ms - dog_ms.min()) / (dog_ms.max() - dog_ms.min() + 1e-12)

    return dog_ms.squeeze().cpu().numpy()

def fast_postprocess_dark_targets(enhanced: np.ndarray,
                     img_log: np.ndarray,
                     wt_enhance: float,
                     min_size: int = 100):
    """
    Post-process the enhanced image to remove small artifacts and enhance contrast.
    """
    _, bw = cv2.threshold(enhanced.astype(np.float32), 0.2, 255, cv2.THRESH_BINARY)
    bw = bw.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_size:
            bw[labels == lbl] = 0
    bw = bw.astype(bool)

    masked = enhanced * bw
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mm = (masked - masked.min())/(masked.max()-masked.min()+1e-12)
    mm8 = np.uint8(mm * 255)
    img2 = clahe.apply(mm8).astype(np.float32) / 255.0
    img_log = img_log.astype(np.float32)
    img_enh = np.max(img_log)*0.8 - img_log + wt_enhance * img2
    np.maximum(img_enh, 0, out=img_enh)  # in-place clamp to ≥0
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img_enh)
        bf = cv2.cuda.createBilateralFilter(ddepth=-1, diameter=3, sigmaColor=2, sigmaSpace=2)
        img_enh = bf.apply(gpu).download()
    else:
        img_enh = cv2.bilateralFilter(img_enh, 3, 2, 2)
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)
    img_enh = cv2.filter2D(img_enh, ddepth=-1, kernel=kernel)
    return img_enh

def fast_postprocess_bright_targets(enhanced: np.ndarray,
                     img_log: np.ndarray,
                     wt_enhance: float,
                     min_size: int = 100):
    """
    Post-process the enhanced image to remove small artifacts and enhance contrast.
    """
    _, bw = cv2.threshold(enhanced.astype(np.float32), 0.2, 255, cv2.THRESH_BINARY)
    bw = bw.astype(np.uint8)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] < min_size:
            bw[labels == lbl] = 0
    bw = bw.astype(bool)
    masked = enhanced * bw
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    mm = (masked - masked.min())/(masked.max()-masked.min()+1e-12)
    mm8 = np.uint8(mm * 255)
    img2 = clahe.apply(mm8).astype(np.float32) / 255.0

    img_enh = wt_enhance * img_log.astype(np.float32) + img2
    np.maximum(img_enh, 0, out=img_enh)  # in-place clamp to ≥0
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        gpu = cv2.cuda_GpuMat()
        gpu.upload(img_enh)
        bf = cv2.cuda.createBilateralFilter(ddepth=-1, diameter=3, sigmaColor=2, sigmaSpace=2)
        img_enh = bf.apply(gpu).download()
    else:
        img_enh = cv2.bilateralFilter(img_enh, 3, 2, 2)
    kernel = np.array([[0, -1,  0],
                       [-1, 5, -1],
                       [0, -1,  0]], dtype=np.float32)
    img_enh = cv2.filter2D(img_enh, ddepth=-1, kernel=kernel)
    return img_enh

def enhance_glands(img_log, wt_enhance):
    sigmas = [4, 8, 12, 16, 24, 32]
    enhanced = multi_scale_dog(img_log, sigmas)
    img_enhanced = fast_postprocess_dark_targets(enhanced, img_log, wt_enhance)
    return img_enhanced

def enhance_fibers(img_log, wt_enhance):
    sigmas = [2, 4, 8, 12, 16]
    enhanced = multi_scale_dog_bright(img_log, sigmas)
    img_enhanced = fast_postprocess_bright_targets(enhanced, img_log, wt_enhance)
    return img_enhanced

def find_surface_bscan_automatic(img_log_ori: np.ndarray, tube_depth: int):
    img_log = img_log_ori.copy()
    Nz , Nx = img_log.shape
    img_log[:tube_depth,:]=0
    
    ub= 0.8
    np.clip(img_log, None, ub, out=img_log)
    img_log /= ub
    
    shrink_factor = 5
    img_log_small = cv2.resize(img_log.astype(np.float32), (Nx//shrink_factor , Nz//shrink_factor))
    img_log_blur = cv2.bilateralFilter(img_log_small,7,2,2)
    
    ret, _ = cv2.threshold(cv2.normalize(img_log_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    otsu_mask = img_log_blur > ((ret - 5) / 255.0)

    top_surface = np.argmax(otsu_mask, axis=0)
    all_zero = ~otsu_mask.any(axis=0)
    top_surface[all_zero] = -1
    
    top_surface = top_surface[np.newaxis, :].astype(np.float32)
    top_surface = cv2.medianBlur(top_surface, 5).flatten()
    
    gaps = top_surface == -1
    indices = np.arange(len(top_surface))
    spline = CubicSpline(indices[~gaps], top_surface[~gaps])
    top_surface = spline(indices)*shrink_factor    

    top_surface = np.clip(top_surface, tube_depth + 5, Nz)
    interpolator = interp1d(np.arange(0,len(top_surface)),top_surface,kind='cubic')
    top_surface = interpolator(np.linspace(0,len(top_surface)-1,Nx))
    top_surface = signal.savgol_filter(top_surface, window_length=25, polyorder=3)
    return top_surface.astype(np.int32)

def extract_patches(cscan_seg: np.ndarray, overlap_perc: float=0.5):
    """
    cscan_seg: (M,Nz,Nx) numpy array
    overlap_perc: float in [0,1)
    returns: patches (num_patches,Nz,Nz), positions (num_patches,)
    Extract overlapping patches of size (Nz,Nz) along the last dimension of cscan_seg:
    the patches have the same depth as the input and are generated by sliding a window along the radial direction.
    """
    _, Nz, Nx = cscan_seg.shape
    step = int(Nz * (1-overlap_perc))
    n_windows = np.ceil((Nx - Nz) / step) + 1
    total_covered = (n_windows - 1) * step + Nz
    pad_right = int(max(0, total_covered - Nx))
    cscan_pad = np.pad(cscan_seg, pad_width=((0, 0), (0,0), (0, pad_right)),mode='wrap')
    windows = sliding_window_view(cscan_pad, window_shape=Nz, axis=2)
    patches = windows.transpose(2, 0, 1, 3)[::step]
    positions = np.arange(0, cscan_pad.shape[2]-Nz+1, step)
    return patches, positions

def svdFilterPatch(cscan_seg_patch: np.ndarray , N_sv_cutoff: int)-> np.ndarray:
    """
    Apply SVD-based denoising to a 3D patch of B-scans (remove the largest N_sv_cutoff singular vectors).
    """
    [Nbscan,Nz,Nx] = cscan_seg_patch.shape
    corsati = cscan_seg_patch.reshape(Nbscan, -1)
    U,S,VT = np.linalg.svd(corsati , full_matrices=False)
    S[:N_sv_cutoff] = 0
    SIGMA = np.diag(S)
    patch_recon = U @ SIGMA @ VT
    return np.std(patch_recon.reshape(Nbscan,Nz,Nx) , axis=0)

def reconstruct_from_patches(patches: np.ndarray, positions: np.ndarray, original_shape: tuple[int,int])-> np.ndarray:
    """
    Reconstruct the full B-scan from overlapping patches using weighted averaging with a Hanning window.
    """
    M, N = original_shape
    _, _, L = patches.shape

    canvas = np.zeros((M, N), dtype=patches.dtype)
    weight = np.zeros((M, N), dtype=np.float32)
    win = np.hanning(L)[None, :]
    win2d = np.repeat(win, M, axis=0)
    for patch, col in zip(patches, positions):
        canvas[:, col:col+L] += patch*win2d
        weight[:, col:col+L] += win2d

    weight[weight == 0] = 1
    return canvas / weight

def svdFilterBScan(cscan_seg: np.ndarray , tissue_surface: np.array , N_sv_cutoff: int) -> np.ndarray:
    """
    Apply SVD-based filtering to the entire C-scan by processing overlapping patches and reconstructing.
    """
    [M,N,L] = cscan_seg.shape
    overlap_perc = 0.5
    patches, positions = extract_patches(cscan_seg , overlap_perc)
    patch_filtered = np.empty((len(positions),N,N))
    for idx_patch in range(patches.shape[0]):
        patch_tmp = patches[idx_patch]
        patch_filtered[idx_patch] = svdFilterPatch(patch_tmp , N_sv_cutoff)
        
    bscan_filtered = reconstruct_from_patches(patch_filtered,positions,[N,positions[-1]+N])        
    bscan_filtered = np.imag(signal.hilbert2(bscan_filtered[:,:L]))
    bscan_filtered += 0.25*cscan_seg[M//2]
    bscan_filtered[bscan_filtered<0]=0
    bscan_filtered = meijering(bscan_filtered, sigmas=range(2,10,2), black_ridges=False) 

    row_indices = np.arange(N).reshape(-1, 1)
    mask = row_indices >= tissue_surface.reshape(1, -1)
    bscan_filtered[~mask]=0
    return bscan_filtered