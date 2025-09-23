import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class focal_loss(nn.Module):
    """
    Focal Loss with optional label smoothing and class weights. alpha is a tensor of shape (C,) for class weights.
    Reference: https://arxiv.org/abs/1708.02002
    Forward signature:
        loss = focal_loss(logits, targets)
    Where:
        logits: (B, C) - raw logits
        targets: (B,) - ground truth class indices
    Returns:
        loss: scalar tensor
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing: float = 0.05):
        super().__init__()
        if alpha is not None:
            self.register_buffer("alpha", alpha.clone().detach().requires_grad_(True))
        else:
            self.alpha = None
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,) long
        ce = F.cross_entropy(logits, targets, reduction="none", label_smoothing=self.label_smoothing)  # (B,)
        pt = torch.exp(-ce)  # probability of the true class
        if self.alpha is not None:
            alpha_factor = self.alpha[targets]  # (B,)
        else:
            alpha_factor = 1.0
        loss = alpha_factor * ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

def clustering_contrastive_loss(
    inst_proj: torch.Tensor,
    attn_probs: torch.Tensor,
    mask: torch.BoolTensor,
    labels: torch.Tensor,
    top_k: int = 10,
    bottom_k: int = 10,
    margin: float = 0.2,
    cancer_clustering: bool = True,
    cancer_clustering_weight: float = 0.2,
):
    """
    Contrastive loss to separate top-k and bottom-k instance features within each bag (contrastive).
    Optionally, also pulls together positive prototypes of bags with the same class label (clustering).
    Forward signature:
        loss = clustering_contrastive_loss(inst_proj, attn_probs, mask, labels, top_k=10, bottom_k=10, margin=0.2)
    Where:
        inst_proj: (B, N, P) - projected instance features  (P is the projection dimension)
        attn_probs: (B, N) - attention probabilities for each instance (before masking)
        mask: (B, N) - boolean mask indicating valid instances (True = valid)
        labels: (B,) - bag-level class labels
    Returns:
        loss: scalar tensor
    """
    B, N, P = inst_proj.shape # P is the cluster projection dimension
    device = inst_proj.device
    losses = []

    # For each bag, we will compute pos_proto and neg_proto (P-dim)
    pos_protos = []
    neg_protos = []
    valid_counts = mask.sum(dim=1)  # (B,)

    for i in range(B):
        valid_n = int(valid_counts[i].item())
        if valid_n == 0:
            # skip entirely padded bag
            pos_protos.append(torch.zeros((P,), device=device))
            neg_protos.append(torch.zeros((P,), device=device))
            losses.append(torch.tensor(0.0, device=device))
            continue

        k_top = min(top_k, valid_n)
        k_bot = min(bottom_k, valid_n)

        # get top-k indices (attn_probs already masked to zero for invalid positions)
        ap = attn_probs[i].clone()
        ap = ap.masked_fill(~mask[i], 0.0) # attention in the padding instances is set to 0

        _, top_idx = torch.topk(ap, k_top, largest=True)
        _, bot_idx = torch.topk(ap, k_bot, largest=False)

        pos_feats = inst_proj[i, top_idx]  # (k_top, P)
        neg_feats = inst_proj[i, bot_idx]  # (k_bot, P)

        pos_proto = pos_feats.mean(dim=0)  # (P,)
        neg_proto = neg_feats.mean(dim=0)  # (P,)

        pos_protos.append(pos_proto)
        neg_protos.append(neg_proto)

        # within-bag separation using cosine similarity
        cos_sim = F.cosine_similarity(pos_proto.unsqueeze(0), neg_proto.unsqueeze(0)).squeeze(0)
        loss_contrastive = F.relu(cos_sim - margin)
        losses.append(loss_contrastive)

    losses = torch.stack(losses)  # (B,)
    loss_sep_batch = losses.mean()

    loss_total = loss_sep_batch

    if cancer_clustering:
        pos_protos_tensor = torch.stack(pos_protos, dim=0)  # (B, P)
        # For each bag i, look for other bags j in the batch with the same label (j != i)
        pull_losses = []
        for i in range(B):
            same_mask = (labels == labels[i])
            same_mask[i] = False  # exclude self
            idxs = torch.nonzero(same_mask, as_tuple=False).squeeze(-1)
            if idxs.numel() == 0:
                continue
            # take mean of pos_protos of same-class others
            other_proto = pos_protos_tensor[idxs].mean(dim=0)  # (P,)
            sim = F.cosine_similarity(pos_protos_tensor[i].unsqueeze(0), other_proto.unsqueeze(0)).squeeze(0)
            pull_loss = (1.0 - sim)
            pull_losses.append(pull_loss)
        if len(pull_losses) > 0:
            pull_loss_batch = torch.stack(pull_losses).mean()
            loss_total = loss_total + cancer_clustering_weight * pull_loss_batch
    return loss_total