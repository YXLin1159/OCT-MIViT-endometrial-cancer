from .dataset import get_embedding_dataloader
import os
import re
from typing import List, Optional, Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from .model_bank import OCT_MIViT
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

def load_bag_embedding_one_sample(idx_class: int, 
                                  idx_sample_in_class: int, 
                                  metadata_path: str, 
                                  path_dataset: str):
    if idx_class == 0:
        benign_indices = [idx_sample_in_class]
        cancer_indices = []
    elif idx_class == 1:
        cancer_indices = [idx_sample_in_class]
        benign_indices = []
    data_loader = get_embedding_dataloader(path_dataset, 0, 1, benign_indices, cancer_indices, metadata_path, 150)
    return data_loader

def load_volume_by_aspect(folder_path: str, aspect: str,
                          allowed_exts: Optional[Tuple[str,...]] = None,
                          require_same_shape: bool = True,
                          verbose: bool = False
                          ) -> Tuple[np.ndarray, List[str], List[int]]:
    """
    Load images from subfolders in `folder_path` that match the given `aspect`
    ('anterior' or 'posterior'), sort them by scanID, and stack into a volume.

    Returns:
      volume: numpy array with shape (Z, H, W) or (Z, H, W, C)
      paths:  list of file paths in the sorted order
      scanIDs: list of integers (scanIDs) in the same order

    Notes:
      - Subfolder detection: any immediate subfolder whose name contains `_{aspect}_`
        (case-insensitive) will be inspected.
      - Filename patterns supported (case-insensitive):
          YYYYMMDD_patientID_{aspect}_scanID.ext
          patientID_{aspect}_scanID.ext
      - Raises ValueError if no matching images found, or if images have inconsistent sizes
        and require_same_shape is True.
    """
    if allowed_exts is None:
        allowed_exts = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')

    aspect = aspect.lower()
    if aspect not in ('anterior', 'posterior'):
        # still proceed, but warn (user might have other tokens)
        if verbose:
            print(f"Warning: aspect='{aspect}' is not 'anterior'/'posterior'. Proceeding anyway.")

    # list immediate subdirectories that mention the aspect token
    all_entries = os.listdir(folder_path)
    subdirs = [d for d in all_entries
               if os.path.isdir(os.path.join(folder_path, d)) and f'_{aspect}_' in d.lower()]

    if verbose:
        print(f"Found {len(subdirs)} subdirectories matching aspect '{aspect}'")

    # regex patterns for filenames (without extension)
    esc_aspect = re.escape(aspect)
    pat_full = re.compile(rf'^(?P<date>\d{{8}})_(?P<pid>\d+)_{esc_aspect}_(?P<scan>\d+)$', re.IGNORECASE)
    pat_short = re.compile(rf'^(?P<pid>\d+)_{esc_aspect}_(?P<scan>\d+)$', re.IGNORECASE)

    matches = []  # list of tuples (scanID:int, fullpath:str)
    for d in tqdm(subdirs, desc="Loading OCT B scans"):
        subpath = os.path.join(folder_path, d)
        for fname in os.listdir(subpath):
            full = os.path.join(subpath, fname)
            if not os.path.isfile(full):
                continue
            name, ext = os.path.splitext(fname)
            if ext.lower() not in allowed_exts:
                continue

            m = pat_full.match(name)
            if not m:
                m = pat_short.match(name)
            if not m:
                # skip files that do not match expected naming
                if verbose:
                    print(f"Skipping file (no match): {full}")
                continue

            scan_str = m.group('scan')
            try:
                scan_id = int(scan_str)
            except ValueError:
                if verbose:
                    print(f"Skipping file (scanID not int): {full}")
                continue

            matches.append((scan_id, full))

    if not matches:
        raise ValueError(f"No images found for aspect='{aspect}' under '{folder_path}'.")

    # sort by scanID numeric (ascending)
    matches.sort(key=lambda x: x[0])
    scanIDs = [m[0] for m in matches]
    paths = [m[1] for m in matches]

    if verbose:
        print(f"Loading {len(paths)} images, scanID range {scanIDs[0]}..{scanIDs[-1]}")

    # load images
    images = []
    shapes = []
    for p in paths:
        with Image.open(p) as im:
            arr = np.asarray(im)  # keeps channels if present
        images.append(arr)
        shapes.append(arr.shape)

    # check shape consistency
    first_shape = shapes[0]
    if require_same_shape:
        inconsistent = [s for s in shapes if s != first_shape]
        if inconsistent:
            # give a helpful error
            raise ValueError(f"Not all images have the same shape. Example shapes found: {set(shapes)}. "
                             "Set require_same_shape=False to allow stacking with manual handling.")
    else:
        # attempt to broadcast or pad to the maximum HxW if needed (simple approach)
        if any(s != first_shape for s in shapes):
            # find max height/width and number of channels
            max_h = max(s[0] for s in shapes)
            max_w = max(s[1] for s in shapes)
            # determine channels if available
            max_c = None
            if len(first_shape) == 3:
                max_c = first_shape[2]
            # pad each image to (max_h, max_w, C) or (max_h, max_w)
            padded = []
            for arr in images:
                if arr.ndim == 2:
                    h, w = arr.shape
                    target = np.zeros((max_h, max_w), dtype=arr.dtype)
                    target[:h, :w] = arr
                else:
                    h, w, c = arr.shape
                    target = np.zeros((max_h, max_w, c), dtype=arr.dtype)
                    target[:h, :w, :c] = arr
                padded.append(target)
            images = padded
            first_shape = images[0].shape

    # stack into volume: axis=0 is Z / scan order
    volume = np.stack(images, axis=0)
    return volume, paths, scanIDs

def forward_get_attention(model_wt_path:str , pca_init_path:str, data_loader, device = torch.device("cuda:0")):
    """
    Given a trained model weight path and a dataloader, extract the attention scores and instance-level predictions.
    Returns a dictionary with keys:
        'cancer_probability': (N, num_classes) - softmax probabilities for each bag
        'attention_strip': (N, L_instances) - attention scores for each instance in the bag
        'prototype_similarity_scores': (N, L_instances) - similarity scores between instance projections and bag prototype
        'instance_projections': (N, L_instances, P) - projected instance features (P is the projection dimension)
    Note: L_instances is the maximum number of instances per bag (padding/truncation applied in dataloader)
    """
    pca_init = np.load(pca_init_path, allow_pickle=True)
    pca_components = pca_init['PCA components']
    pca_means      = pca_init['PCA mean']
    _, _, mask_tmp, _ = next(iter(data_loader))
    L_instances = mask_tmp.sum(dim=1).tolist() # actual number of input B scans from a sample does not equal max number of instances
    L_instances = max(L_instances)
    
    model = OCT_MIViT(input_dim = 4096, proj_dim = 512, pca_components = pca_components, pca_mean = pca_means)
    model.load_state_dict(torch.load(model_wt_path))
    model.eval().to(device)
    all_logits = []
    all_labels = []
    all_attn_probs = []
    all_inst_proj = []
    all_inst_probs = []
    with torch.no_grad():
        for bags, labels, masks, metas in tqdm(data_loader, desc="Testing", leave=False):
            bags, labels, masks, metas = bags.to(device), labels.to(device).long(), masks.to(device), metas.to(device)
            logits, attn_probs, _, inst_proj, _, _ = model(bags, mask = masks , meta = metas)
            pos_proto = (inst_proj * attn_probs.unsqueeze(-1)).sum(dim=1) / (attn_probs.sum(dim=1, keepdim=True)+1e-8)  # (B,P)
            sim = F.cosine_similarity(inst_proj, pos_proto.unsqueeze(1), dim=-1)  # (B,N)
            inst_probs = (sim + 1.0) / 2.0

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_attn_probs.append(attn_probs.detach().cpu())
            all_inst_proj.append(inst_proj.detach().cpu())
            all_inst_probs.append(inst_probs.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0) # (20,2)
    all_labels = torch.cat(all_labels, dim=0) # (20,)
    
    all_attn_probs = torch.cat(all_attn_probs, dim=0) #(20,150)
    all_attn_probs = all_attn_probs[:,0:L_instances]
    all_attn_probs = all_attn_probs.cpu().numpy().flatten('F')
    
    all_inst_probs = torch.cat(all_inst_probs,dim=0)
    all_inst_probs = all_inst_probs[:,0:L_instances]
    all_inst_probs = all_inst_probs.cpu().numpy().flatten('F')
    
    all_inst_proj  = torch.cat(all_inst_proj,  dim=0) #(20,150,128)
    probs_cancer = torch.softmax(all_logits, dim=1)
    all_inst_proj = all_inst_proj.cpu().numpy()[0]

    xai_results = {
        'cancer_probability': probs_cancer,
        'attention_strip': all_attn_probs,
        'prototype_similarity_scores': all_inst_probs,
        'instance_projections': all_inst_proj,
    }

    return xai_results

def plot_attention_strips(attn, inst_probs, bag_prob=None, smooth_sigma=2.5, top_k=10):
    """
    Given attention scores and instance-level prediction scores for a bag, plot the two curves
    and their combined suspiciousness score. Annotate the top-k B scans by the combined score.
    Args:
        attn: (N,) numpy array of attention scores for each instance in the bag
        inst_probs: (N,) numpy array of instance-level prediction scores (e.g. cancer probability)
        bag_prob: optional float, bag-level probability (e.g. cancer probability) to display in title
        smooth_sigma: float, standard deviation for Gaussian smoothing of the curves
        top_k: int, number of top instances to annotate by vertical dashed lines
    Returns:
        res: dictionary with keys:
            'attention_scores': (N,) numpy array of normalized attention scores
            'prototype_similarity_scores': (N,) numpy array of normalized instance prediction scores
            'diagnostic_significance': (N,) numpy array of combined suspiciousness scores
            'top_idx': (top_k,) numpy array of indices of top-k instances by combined score
    """
    attn_s = gaussian_filter1d(attn, sigma=smooth_sigma)
    inst_s = gaussian_filter1d(inst_probs, sigma=smooth_sigma)

    attn_norm = 1.0-(attn_s - attn_s.min()) / (attn_s.max()-attn_s.min()+1e-8)
    inst_norm = 1.0 - (inst_s - inst_s.min()) / (inst_s.max()-inst_s.min()+1e-8)

    # combined suspiciousness
    combined = attn_norm * inst_norm
    combined_norm = (combined - combined.min()) / (combined.max()-combined.min()+1e-8)

    N = len(attn)
    _, axes = plt.subplots(3, 1, figsize=(12, 4), sharex=True, dpi = 400)
    axes[0].plot(np.linspace(1,N,N), attn_norm ,color='black')
    axes[0].set_ylabel('Attention')
    axes[1].plot(np.linspace(1,N,N), inst_norm ,color='black')
    axes[1].set_ylabel('Prototype sim')
    axes[2].plot(np.linspace(1,N,N), combined_norm ,color='maroon')
    axes[2].set_ylabel('Suspiciousness')
    axes[2].set_xlabel('B scan index')
    axes[2].set_xlim([0,N])

    # annotate top-k by combined score
    top_idx = np.argsort(-combined_norm)[:top_k]
    for ax in axes:
        for idx in top_idx:
            ax.axvline(idx, color='white', linestyle='--', linewidth=0.8, alpha=0.8)
    plt.show()
    res = {'attention_scores': attn_norm, 
           'prototype_similarity_scores': inst_norm, 
           'diagnostic_significance': combined_norm, 
           'top_idx': top_idx}
    return res
