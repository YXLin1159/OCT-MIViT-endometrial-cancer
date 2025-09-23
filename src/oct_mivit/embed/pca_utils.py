from typing import Optional
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA, IncrementalPCA
import torch

def _extract_valid_instances_from_batch(bags: torch.Tensor, mask: torch.Tensor, max_per_bag: Optional[int] = None):
    '''
    Given one batch, return a 2D numpy array (M, D) of valid instance embeddings.
    '''
    device = bags.device
    B, N, D = bags.shape
    mask_bool = mask.bool()
    rows = []
    for b in range(B):
        valid_idx = torch.where(mask_bool[b])[0]
        if valid_idx.numel() == 0:
            continue
        inst = bags[b, valid_idx]  # (n_valid, D)
        if max_per_bag is not None and inst.shape[0] > max_per_bag:
            # random sample without replacement
            perm = torch.randperm(inst.shape[0], device=device)[:max_per_bag]
            inst = inst[perm]
        rows.append(inst.cpu().numpy())
    if not rows:
        return np.empty((0, D), dtype=np.float32)
    return np.concatenate(rows, axis=0)  # (M, D)

def compute_pca_from_dataloader(
    dataloader,
    n_components: int = 512, # default value is 512 for OCT-MIViT
    use_incremental: bool = False, # memory sufficient for my dataset, but for larger datasets, use incremental
    ipca_batch_size: int = 10000,
    max_instances_per_bag: Optional[int] = None,
    max_samples: Optional[int] = None,
    sample_every_n_batches: int = 1,
    save_dir: Optional[str] = None,
    save_prefix: str = "pca",
    random_seed: int = 0,
    verbose: bool = True,
):
    '''
    Fit PCA (or IncrementalPCA) using embeddings pulled from training dataloader.
    Returns:
        components: np.float32 array shape (n_components, L_embed)
        mean:       np.float32 array shape (L_embed,)
    '''
    rng = np.random.RandomState(random_seed)

    # 1st pass: collect instance vectors
    processed = 0
    D = None
    if use_incremental:
        ipca = IncrementalPCA(n_components=n_components)
        buffer_rows = []
        buffer_count = 0
        for b_idx, batch in enumerate(dataloader):
            if (sample_every_n_batches > 1) and (b_idx % sample_every_n_batches != 0):
                continue
            bags, _, masks, _ = batch
            if not isinstance(bags, torch.Tensor):
                bags = torch.tensor(bags)
            if not isinstance(masks, torch.Tensor):
                masks = torch.tensor(masks).bool()

            rows = _extract_valid_instances_from_batch(bags, masks, max_per_bag=max_instances_per_bag)
            if rows.size == 0:
                continue
            if max_samples is not None:
                remaining = max_samples - processed
                if remaining <= 0:
                    break
                if rows.shape[0] > remaining:
                    idx = rng.choice(rows.shape[0], remaining, replace=False)
                    rows = rows[idx]

            buffer_rows.append(rows)
            buffer_count += rows.shape[0]
            processed += rows.shape[0]
            if D is None:
                D = rows.shape[1]
            if verbose and (processed % 5000 < rows.shape[0]):
                print(f"[compute_pca] collected {processed} samples (target components={n_components})")

            # flush buffer if large enough
            if buffer_count >= ipca_batch_size:
                X_chunk = np.vstack(buffer_rows).astype(np.float32)
                ipca.partial_fit(X_chunk)
                buffer_rows = []
                buffer_count = 0

            if max_samples is not None and processed >= max_samples:
                break

        if buffer_count > 0:
            X_chunk = np.vstack(buffer_rows).astype(np.float32)
            ipca.partial_fit(X_chunk)
            buffer_rows = []
            buffer_count = 0

        components = ipca.components_.astype(np.float32)    # (n_components, L_embed)
        mean = ipca.mean_.astype(np.float32)                # (L_embed,)
    else:
        all_rows = []
        for b_idx, batch in enumerate(dataloader):
            if (sample_every_n_batches > 1) and (b_idx % sample_every_n_batches != 0):
                continue
            bags, labels, masks, metas = batch
            rows = _extract_valid_instances_from_batch(bags, masks, max_per_bag=max_instances_per_bag)
            if rows.size == 0:
                continue
            all_rows.append(rows)
            processed += rows.shape[0]
            if D is None:
                D = rows.shape[1]
            if max_samples is not None and processed >= max_samples:
                break
        if not all_rows:
            raise RuntimeError("No instance embeddings found in dataloader.")
        X = np.vstack(all_rows).astype(np.float32)
        if max_samples is not None and X.shape[0] > max_samples:
            idx = rng.choice(X.shape[0], max_samples, replace=False)
            X = X[idx]
        if verbose:
            print(f"[compute_pca] Fitting PCA on {X.shape[0]} samples with dim {X.shape[1]}")
        pca = PCA(n_components=n_components, random_state=random_seed)
        pca.fit(X)
        components = pca.components_.astype(np.float32)
        mean = pca.mean_.astype(np.float32)

    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        comp_path = save_dir / f"{save_prefix}_components.npy"
        mean_path = save_dir / f"{save_prefix}_mean.npy"
        np.save(str(comp_path), components)
        np.save(str(mean_path), mean)
        if verbose:
            print(f"[compute_pca] Saved components to {comp_path} and mean to {mean_path}")

    return components, mean
