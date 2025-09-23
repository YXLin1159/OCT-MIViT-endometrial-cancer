from __future__ import annotations
import os
import argparse
from pathlib import Path
import logging
from typing import Callable, Dict, Tuple, Optional
import numpy as np
from tqdm import tqdm

import torch
from PIL import Image
from torchvision import transforms
import timm
from transformers import AutoModel, AutoImageProcessor, CLIPImageProcessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("compute-embeddings")

def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def save_embedding(save_path: Path, model_key: str, emb: np.ndarray, overwrite: bool = False):
    if save_path.exists() and not overwrite:
        try:
            with np.load(save_path, allow_pickle=False) as data:
                existing = dict(data)
        except Exception:
            existing = {}
        existing[f"emb_{model_key}"] = emb.astype(np.float32)
        np.savez_compressed(save_path, **existing)
    else:
        np.savez_compressed(save_path, **{f"emb_{model_key}": emb.astype(np.float32)})

def load_timm_model(model_identifier: str, device: torch.device, pretrained: bool = True, **timm_kwargs):
    '''
    Load a timm model and its corresponding preprocessing transform.
    model_identifier: timm model id or 'hf-hub:{repo_id}' style (timm supports hf-hub URIs).
    Returns (model, preprocess_transform)
    '''
    model = timm.create_model(model_name=model_identifier, pretrained=pretrained, **timm_kwargs)
    model.eval().to(device)
    cfg = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**cfg, is_training=False)
    return model, preprocess

def load_transformers_model(model_repo: str, device: torch.device, use_processor: bool = True):
    '''
    Load a HuggingFace transformers model and optionally its image processor.
    model_repo: HF model repo id (e.g. 'facebook/dinov2-small')
    Returns (model, processor or None)
    '''
    processor = AutoImageProcessor.from_pretrained(model_repo) if use_processor else None
    model = AutoModel.from_pretrained(model_repo).eval().to(device)
    return model, processor

def _register_timm(model_name: str):
    def factory(device: torch.device):
        '''
        Load a timm model and return (model, preprocess_transform, forward_fn)
        forward_fn(model, input, device) -> numpy array (embedding)
        '''
        model, transform = load_timm_model(model_name, device)
        def forward_fn(model, img, device):
            # img: PIL.Image
            t = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
            with torch.no_grad():
                out = model(t)
            # many timm models return features or logits; if (1, C) assume features
            emb = out.cpu().numpy().squeeze()
            return emb
        return model, transform, forward_fn
    return factory

def _register_RadioDino(model_name: str):
    def factory(device: torch.device):
        model = timm.create_model(model_name, pretrained=True).eval().to(device)
        transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])   
        def forward_fn(model, img, device):
            # img: PIL.Image
            t = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
            with torch.no_grad():
                out = model(t)
            # many timm models return features or logits; if (1, C) assume features
            emb = out.cpu().numpy().squeeze()
            return emb
        return model, transform, forward_fn
    return factory

def _register_UNI(model_name: str):
    def factory(device: torch.device):
        timm_kwargs = {
            'img_size': 224, 
            'patch_size': 14, 
            'depth': 24,
            'num_heads': 24,
            'init_values': 1e-5, 
            'embed_dim': 1536,
            'mlp_ratio': 2.66667*2,
            'num_classes': 0, 
            'no_embed_class': True,
            'mlp_layer': timm.layers.SwiGLUPacked, 
            'act_layer': torch.nn.SiLU, 
            'reg_tokens': 8, 
            'dynamic_img_size': True
        }
        model = timm.create_model(model_name, pretrained=True, **timm_kwargs).eval().to(device)
        transform = timm.data.transforms_factory.create_transform(**timm.data.resolve_data_config(model.pretrained_cfg, model=model))
        def forward_fn(model, img, device):
            # img: PIL.Image
            t = transform(img).unsqueeze(0).to(device)  # (1,C,H,W)
            with torch.no_grad():
                out = model(t)
            # many timm models return features or logits; if (1, C) assume features
            emb = out.cpu().numpy().squeeze()
            return emb
        return model, transform, forward_fn
    return factory

def _register_transformers(repo: str):
    def factory(device: torch.device):
        '''
        Load a HF transformers model and return (model, processor, forward_fn)
        forward_fn(model, input, device) -> numpy array (embedding)
        '''
        model, processor = load_transformers_model(repo, device)
        def forward_fn(model, img, device):
            # use processor to produce pixel_values
            inputs = processor(img, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            # default: CLS token at last_hidden_state[:,0,:]
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()
            return emb
        return model, processor, forward_fn
    return factory

def _register_RADIO(repo: str = "nvidia/RADIO-B"):
    def factory(device: torch.device):
        model = AutoModel.from_pretrained(repo , trust_remote_code=True).eval().to(device)
        processor = CLIPImageProcessor.from_pretrained(repo)
        def forward_fn(model, img, device):
            # use processor to produce pixel_values
            inputs = processor(img, return_tensors="pt",do_resize=True).pixel_values.to(device)
            with torch.no_grad():
                outputs , _ = model(**inputs)
            # default: CLS token at last_hidden_state[:,0,:]
            emb = outputs.cpu().numpy().squeeze()
            return emb
        return model, processor, forward_fn
    return factory

MODEL_REGISTRY: Dict[str, Callable[[torch.device], Tuple[object, object, Callable]]] = {
    # timm models (hf-hub supported names)
    "kaiko": _register_timm("hf-hub:1aurent/vit_small_patch14_224.dinobloom"),
    "optimus": _register_timm("hf-hub:bioptimus/H-optimus-1"),
    "gigapath": _register_timm("hg_hub:prov-gigapath/prov-gigapath"),
    "uni": _register_UNI("hf-hub:MahmoodLab/UNI2-h"),
    "radiodino": _register_RadioDino("hf_hub:Snarcy/RadioDino-s16"),
    # transformer-style HF models
    "radio": _register_RADIO("nvidia/RADIO-B"),
    "phikon": _register_transformers("owkin/phikon-v2"),
    "dinov2": _register_transformers("facebook/dinov2-small"),
    "nomic": _register_transformers("nomic-ai/nomic-embed-vision-v1.5"),
    # torchvision example
    "resnet34": lambda device: (torchvision_resnet_feature_extractor(device), None, torchvision_forward),
    # for MedImageInsights, please download the model weights and load locally
}

# minimal resnet helper to show how to include torchvision models (example)
def torchvision_resnet_feature_extractor(device: torch.device):
    import torchvision.models as models
    from torchvision import transforms as T
    resnet = models.resnet34(pretrained=True)
    feat = torch.nn.Sequential(*list(resnet.children())[:-1]).to(device).eval()
    return feat

def torchvision_forward(model, img: Image.Image, device: torch.device):
    from torchvision import transforms as T
    t = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                   T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    x = t(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x).squeeze()
    return out.cpu().numpy()

# main processing function
def process_folder_images(
    model_key: str,
    model_factory: Callable[[torch.device], Tuple[object, object, Callable]],
    src_dir: Path,
    out_dir: Path,
    device: Optional[torch.device] = None,
    overwrite: bool = False,
    start_case: int = 0,
):
    '''
    Walks the dataset tree under src_dir, finds subfolders (cases), then partitions inside them;
    for each .png file found, computes embedding and saves a .npz to out_dir with same relative path.
    '''
    device = device or get_device()
    model, preprocess, forward_fn = model_factory(device)
    logger.info(f"Model key={model_key}, device={device}")

    # find case folders
    case_folders = [p for p in sorted(Path(src_dir).iterdir()) if p.is_dir()]
    logger.info(f"Found {len(case_folders)} case folders in {src_dir}")
    for idx_case, casefolder in enumerate(tqdm(case_folders[start_case:], desc="cases")):
        part_folders = [p for p in sorted(casefolder.iterdir()) if p.is_dir()]
        for part in part_folders:
            # for each image in a partition
            for img_path in sorted(part.glob("*.png")):
                rel = img_path.relative_to(Path(src_dir)).with_suffix(".npz")
                save_path = Path(out_dir) / rel
                save_dir = save_path.parent
                save_dir.mkdir(parents=True, exist_ok=True)
                try:
                    img = Image.open(img_path).convert("RGB")
                    emb = forward_fn(model, img, device)
                    if emb is None:
                        logger.warning(f"No embedding from model for {img_path}; skipping")
                        continue
                    save_embedding(save_path, model_key, np.asarray(emb, dtype=np.float32), overwrite=overwrite)
                except Exception as exc:
                    logger.exception(f"Failed processing {img_path}: {exc}")
                    # continue on error
    logger.info("Done processing images.")

# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Compute image embeddings for many foundation models.")
    p.add_argument("--src_dir", type=str, required=True, help="Root folder containing per-case subfolders.")
    p.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()), help="Model key to compute embeddings with.")
    p.add_argument("--out_dir", type=str, default=None, help="Where to save embeddings (defaults to <src_dir>/Embedding).")
    p.add_argument("--workers", type=int, default=4, help="Not used yet; placeholder for future parallelization.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing .npz files.")
    p.add_argument("--start_case", type=int, default=0, help="Skip cases up to this index (useful to resume).")
    return p.parse_args()

def main():
    args = parse_args()
    src_dir = Path(args.src_dir).resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else src_dir / "Embedding"
    ensure_outdir(out_dir)

    # Optional: set HF token from env if present
    hf_token = os.getenv("HF_TOKEN", None)
    if hf_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
        logger.info("Using HF_TOKEN from environment for model downloads (OK).")

    model_key = args.model
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model {model_key}. Available: {list(MODEL_REGISTRY.keys())}")
    factory = MODEL_REGISTRY[model_key]

    device = get_device()
    process_folder_images(model_key=model_key, model_factory=factory, src_dir=src_dir, out_dir=out_dir,
                          device=device, overwrite=args.overwrite, start_case=args.start_case)


if __name__ == "__main__":
    main()
