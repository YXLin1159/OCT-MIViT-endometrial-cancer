import numpy as np
import matplotlib.pyplot as plt

def overlay_two_maps_v2(bg, ov1, ov2, cmap1="Reds", cmap2="Greens"):
    """
    bg, ov1, ov2: H×W numpy arrays
    returns: fig, ax with overlayed images (pseudo-RGB)
    """
    bg_min, bg_max = bg.min(), bg.max()
    bg_norm = (bg - bg_min) / (bg_max - bg_min + 1e-12)

    def norm01(x):
        mn, mx = x.min(), x.max()
        return (x - mn) / (mx - mn + 1e-12)

    ov1_norm = norm01(ov1)
    ov2_norm = norm01(ov2)
    gamma = 0.5
    fig, ax = plt.subplots(dpi=1200)
    ax.imshow(bg_norm, cmap='Blues', alpha=0.9*np.power(bg_norm,gamma), interpolation='none')
    ax.imshow(ov1_norm, cmap=cmap1, alpha=0.9*np.power(ov1_norm,gamma), interpolation='none')
    ax.imshow(ov2_norm, cmap=cmap2, alpha=0.9*np.power(ov2_norm,gamma), interpolation='none')
    ax.axis('off')
    plt.tight_layout()
    return fig, ax
