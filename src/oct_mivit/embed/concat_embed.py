'''
Usage:
from src.oct_mil.embed.concat_embed import concat_embeddings_in_folder
# Suppose embeddings are stored under data/DS_threeChannels/Embedding (per-image .npz files)
concat_embeddings_in_folder(
    embeddings_root="data/DS_threeChannels/Embedding",
    model_keys=["dinov2", "opt-optimus", "phikon", "nomic"],
    out_root="data/DS_threeChannels/Embedding_concatenated",
    fill_missing="zeros",
    overwrite=False
)
'''
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Dict, Tuple, Optional, List
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _load_embedding_file(path: Path) -> Dict[str, np.ndarray]:
    '''
    Load an embedding file (.npz or .npy) and return a mapping of keys to 1D float32 arrays.
    Supports:
    - .npz files with multiple arrays (keys preserved)
    - .npy files with single array (key 'emb_legacy')
    - .npy files with object arrays (keys 'emb_obj_i' for each element)
    If the file does not exist or cannot be read, returns an empty dict.
    All arrays are flattened to 1D float32.
    Returns:
        dict mapping str -> np.ndarray (1D float32)
    '''
    path = Path(path)
    out: Dict[str, np.ndarray] = {}
    if not path.exists():
        return out

    suffix = path.suffix.lower()
    try:
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as z:
                for k in z.files:
                    arr = np.asarray(z[k])
                    # flatten to 1D if possible
                    if arr.ndim > 1:
                        arr = arr.reshape(-1)
                    out[k] = arr.astype(np.float32)
        elif suffix == ".npy":
            # try non-pickle load first
            try:
                arr = np.load(path, allow_pickle=False)
            except Exception:
                arr = np.load(path, allow_pickle=True)
            arr = np.asarray(arr)
            if arr.dtype == object:
                # object-array: try to flatten into named keys
                elems = list(arr.flatten())
                for i, e in enumerate(elems):
                    try:
                        earr = np.asarray(e).reshape(-1).astype(np.float32)
                        out[f"emb_obj_{i}"] = earr
                    except Exception:
                        # skip elements we can't interpret
                        continue
            elif arr.ndim == 1:
                out["emb_legacy"] = arr.astype(np.float32)
            else:
                out["emb_legacy"] = arr.reshape(-1).astype(np.float32)
        else:
            try:
                data = np.load(path, allow_pickle=True)
                if isinstance(data, dict) or hasattr(data, "files"):
                    for k in getattr(data, "files", []):
                        out[k] = np.asarray(data[k]).reshape(-1).astype(np.float32)
                else:
                    arr = np.asarray(data)
                    out["emb_legacy"] = arr.reshape(-1).astype(np.float32)
            except Exception:
                logger.warning(f"Could not load embedding file {path}")
    except Exception as e:
        logger.exception(f"Error loading {path}: {e}")
    return out

def _discover_model_dims(files: Iterable[Path], model_keys: List[str]) -> Dict[str, int]:
    dims: Dict[str, int] = {}
    for p in files:
        if len(dims) == len(model_keys):
            break
        mapping = _load_embedding_file(p)
        for mk in model_keys:
            key = f"emb_{mk}"
            if key in mapping and mapping[key].ndim == 1:
                dims[mk] = mapping[key].shape[0]
    return dims

def concat_embeddings_in_folder(
    embeddings_root: str | Path,
    model_keys: List[str],
    out_root: Optional[str | Path] = None,
    recursive: bool = True,
    allowed_exts: Tuple[str, ...] = (".npz", ".npy"),
    fill_missing: str = "zeros",   # "zeros" or "skip" or "error"
    overwrite: bool = False,
) -> None:
    """
    Concatenate multiple model embeddings (per-image) into a single vector per image.

    Args:
      embeddings_root: root folder containing per-image embedding files (preserves relative layout).
      model_keys: ordered list of model keys to concatenate. Must match keys used when saving: 'emb_<model_key>'.
      out_root: where to write concatenated results. If None, writes into embeddings_root / "concatenated".
      recursive: whether to walk subdirectories recursively.
      allowed_exts: file extensions to consider.
      fill_missing: 'zeros' (default) -> substitute zeros for missing model embeddings;
                    'skip' -> skip files missing any model; 'error' -> raise on missing.
      overwrite: overwrite existing concatenated files if present.

    Output:
      Writes .npy (float32) files named like the input but under out_root with same relative paths.
      Each .npy contains the concatenated 1D float32 vector.
    """
    embeddings_root = Path(embeddings_root)
    if out_root is None:
        out_root = embeddings_root / "concatenated"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # gather input files
    if recursive:
        files = [p for p in embeddings_root.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts]
    else:
        files = [p for p in embeddings_root.glob("*") if p.is_file() and p.suffix.lower() in allowed_exts]

    if not files:
        logger.warning(f"No embedding files found under {embeddings_root}")
        return

    dims = _discover_model_dims(files, model_keys)
    missing_models = [mk for mk in model_keys if mk not in dims]
    if missing_models:
        if fill_missing == "zeros":
            # allow missing models but set their dim=0 (will effectively skip them)
            logger.warning(f"Could not discover dims for models {missing_models}. Missing models will be replaced with zeros.")
            for mk in missing_models:
                dims[mk] = 0
        else:
            raise RuntimeError(f"Could not find embeddings for models {missing_models} in dataset files. Try 'fill_missing=\"zeros\"' or ensure embeddings exist.")

    for src in files:
        rel = src.relative_to(embeddings_root)
        dst = out_root / rel.with_suffix(".npy")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and not overwrite:
            continue

        mapping = _load_embedding_file(src)
        parts: List[np.ndarray] = []
        skip_file = False
        for mk in model_keys:
            key = f"emb_{mk}"
            if key in mapping:
                arr = np.asarray(mapping[key], dtype=np.float32).reshape(-1)
                parts.append(arr)
            else:
                if dims[mk] == 0:
                    continue
                if fill_missing == "zeros":
                    parts.append(np.zeros((dims[mk],), dtype=np.float32))
                elif fill_missing == "skip":
                    skip_file = True
                    break
                elif fill_missing == "error":
                    raise RuntimeError(f"Missing embedding for model '{mk}' in file {src}")
        if skip_file:
            logger.debug(f"Skipping {src} because some model embeddings are missing")
            continue
        if not parts:
            logger.debug(f"No parts to concatenate for {src}. Skipping.")
            continue

        concat = np.concatenate(parts, axis=0).astype(np.float32)
        np.save(dst, concat)
    logger.info(f"Concatenation complete. Concatenated files written to {out_root}")
