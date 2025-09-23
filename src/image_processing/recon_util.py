from dataclasses import dataclass
from pathlib import Path
import numpy as np
import numba as nb
from scipy import signal

N_SAMPLES = 6144
HAMM = signal.windows.hamming(N_SAMPLES, False)  # periodic
KAISER = signal.windows.kaiser(N_SAMPLES, 16, False)

def load_fringe_bin_seg_YL(filepath: str, n_alines: int, n_bscan_to_load: int, i_start = int)->np.ndarray:
    """
    Load a segment of raw fringe data from a binary file.
    """
    elements_per_bscan = n_alines * N_SAMPLES
    bytes_per_element = 2  # uint16
    total_elements = n_bscan_to_load * elements_per_bscan
    offset = i_start * elements_per_bscan * bytes_per_element
    try:
        with open(filepath, 'rb') as f:
            f.seek(offset, 0)
            raw = np.fromfile(f, dtype=np.uint16, count=total_elements)
        actual_n_bscan_loaded = raw.size // elements_per_bscan
        if actual_n_bscan_loaded < n_bscan_to_load:
            print(f"Warning: only {actual_n_bscan_loaded} scans available (requested {n_bscan_to_load})")
        raw = raw[: actual_n_bscan_loaded * elements_per_bscan]
        fringe = raw.reshape((actual_n_bscan_loaded, n_alines, N_SAMPLES)).astype(np.float32)
    except Exception as e:
        print(f"Error loading {filepath}:\n{e}")
        raise
    return fringe

def load_background_bin(file: Path | str):
    """
    Similar to load_fringe_bin (loads the same kind of bin file), but don't care about
    frames and B-scans (reshape to 2D).
    """
    #fringe = np.fromfile(file, dtype=np.uint16)
    fringe = np.loadtxt(file)
    n_alines = fringe.size // N_SAMPLES
    fringe = fringe.reshape((n_alines, N_SAMPLES), order="C")
    return fringe.mean(axis=0)

@dataclass
class Calib:
    """
    OCT Calibration data
    """

    ss_idx: np.ndarray
    ss_l_coeff: np.ndarray
    ss_r_coeff: np.ndarray
    l_coeff: np.ndarray
    r_coeff: np.ndarray
    background: np.ndarray | None

    n_alines: int = 2500  # num A-lines
    theory_alines: int = 2000
    imagedepth: int = 600

    @staticmethod
    def load_calib_files(calib_file: Path | str):
        _calib = np.loadtxt(calib_file, dtype=np.double, max_rows=N_SAMPLES)
        ss_idx = _calib[:, 0].astype(int) - 1  # index
        ss_l_coeff = _calib[:, 1]
        ss_r_coeff = _calib[:, 2]
        return ss_idx, ss_l_coeff, ss_r_coeff

    @classmethod
    def invivo(
        cls,
        background_bin: Path | str | None = None,
        calib_file: Path | str | None = None,
    ):
        if calib_file is None:
            calib_file = Path("oct_proc/SSOCTCalibration180MHZ.txt")
        assert calib_file.exists()
        ss_idx, ss_l_coeff, ss_r_coeff = cls.load_calib_files(calib_file)

        if background_bin is not None:
            background_bin = Path(background_bin)
            assert background_bin.exists()
            background = load_background_bin(background_bin)
        else:
            background = None

        return cls(
            background=background,
            ss_idx=ss_idx,
            ss_l_coeff=ss_l_coeff,
            ss_r_coeff=ss_r_coeff,
            l_coeff=ss_l_coeff[ss_idx],
            r_coeff=ss_r_coeff[ss_idx],
            n_alines=2200,
            theory_alines=2000,
        )

def _mean_axis0(a):
    res = np.zeros(a.shape[1], dtype=np.double)
    for i in nb.prange(a.shape[0]):
        res += a[i]
    return res / a.shape[0]

def recon_bscan_YL(fringe_bscan: np.ndarray, calib: Calib, imagedepth: int = 0)-> np.ndarray:
    """
    Reconstruct a single B-scan (2D numpy array) from raw fringe data (2D numpy array).
    """
    n_alines, n_samples = fringe_bscan.shape
    if imagedepth == 0:
        imagedepth = n_samples // 2 + 1  # rfft size

    I = np.zeros((imagedepth, n_alines))
    win = KAISER

    # Estimate background with mean of the fringes
    background = _mean_axis0(fringe_bscan)
    for j in nb.prange(n_alines):
        # 1. subtract background
        fringe_sub = fringe_bscan[j, :] - background
        # 2. interpolate phase calib data to be linear in k-space
        linear_k_fringe = (fringe_sub[calib.ss_idx] * calib.l_coeff + fringe_sub[calib.ss_idx + 1] * calib.r_coeff)
        # 3. fft on A-line
        fft_fringe = np.fft.ifft(linear_k_fringe * win, norm = "backward")
        fft_fringe = np.abs(np.real(fft_fringe[:imagedepth])) + np.abs(np.real(np.flip(fft_fringe[n_samples-imagedepth : n_samples])))
        I[:, j] = fft_fringe[:imagedepth]
    return np.array(I)

def log_compress_auto(img , depth_last):
    """
    Log-compress the OCT image with automatic dynamic range adjustment and TGC.
    """
    img = img[:depth_last,:]
    img = np.nan_to_num(img, nan=0.0)
    signal_max = np.percentile(img, 99.95)
    noise_floor = img[depth_last-25:depth_last,:].mean()
    db_oct = 20 * np.log10(signal_max / noise_floor) - 1
    
    img_norm = img / signal_max
    img_log = (20/db_oct) * np.log10(img_norm+1e-12)+1
    img_log = np.clip(img_log , 0.0 , 1.0)
    
    mut_correction = 5 # [cm^-1]
    Nz , Nx = img_log.shape
    tgc = 1 + 20/db_oct*mut_correction*np.arange(1,Nz+1)*7.7*1e-5
    tgc = tgc/tgc[120]
    tgc[:120]=1
    tgc = tgc.reshape(-1,1)
    img_log = img_log * tgc
    return img_log