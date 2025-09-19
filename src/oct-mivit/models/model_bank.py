import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class MedHistEncoder(nn.Module):
    """
    Simple MLP to encode metadata history into a feature vector.
    Input: (B, input_dim)
    Output: (B, out_dim)
    """
    def __init__(self, input_dim: int, out_dim: int = 32, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        return self.net(x)  # (B, out_dim)

class OCT_MIViT(nn.Module):
    """
    MIL model for OCT data with metadata conditioning.
    Uses a transformer encoder to process B-scan features, with an additional metadata token.
    """
    def __init__(
        self,
        input_dim: int = 8192,
        proj_dim: int = 512,
        pca_components: Optional[np.ndarray] = None,
        pca_mean: Optional[np.ndarray] = None,
        freeze_proj: bool = False,
        attn_hidden: int = 256,
        classifier_hidden: int = 128,
        cluster_proj_hidden: int = 128,
        cluster_proj_out: int = 128,
        num_classes: int = 2,
        meta_input_dim: int = 6,
        meta_token_dim: int = 32,
        meta_cluster_dim: int = 16,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.proj_dim = proj_dim
        self.attn_hidden = attn_hidden
        self.meta_input_dim = meta_input_dim
        self.meta_token_dim = meta_token_dim
        self.meta_cluster_dim = meta_cluster_dim
        self._pca_initialized = True

        # INPUT PROJECTION, 8192 IS TOO LARGE
        self.input_proj = nn.Linear(input_dim, proj_dim, bias=True)
        if pca_components is not None and pca_mean is not None:
            # Expect pca_components shape: (pca_proj_dim, embedding_dim)
            # Linear weight shape: (out_features, in_features) i.e. (proj_dim, input_dim)
            assert pca_components.shape[0] == proj_dim and pca_components.shape[1] == input_dim, (
                f"pca_components must be shape ({proj_dim}, {input_dim})"
            )
            pca_weight = torch.from_numpy(pca_components.astype(np.float32)).to(torch.get_default_dtype())
            pca_bias = - torch.from_numpy(pca_mean.astype(np.float32)).to(torch.get_default_dtype()) @ pca_weight.T
            with torch.no_grad():
                self.input_proj.weight.copy_(pca_weight)
                self.input_proj.bias.copy_(pca_bias)
            if freeze_proj:
                for p in self.input_proj.parameters():
                    p.requires_grad = False
        else:
            nn.init.xavier_uniform_(self.input_proj.weight)
            nn.init.constant_(self.input_proj.bias, 0.0)
        
        # TRANSFORMER / ATTENTION HEAD
        encoder_layer = nn.TransformerEncoderLayer(d_model=proj_dim, nhead=2, dim_feedforward=1024, batch_first=True, dropout=0.1, activation="relu")
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # ADD ATTENTION ON TOP OF TRANSFORMER TO PROVDE INTERPRETABILITY
        self.attn_V = nn.Linear(self.proj_dim, self.attn_hidden)
        self.attn_U = nn.Linear(self.proj_dim, self.attn_hidden)
        self.attn_score = nn.Linear(self.attn_hidden, 1)
        
        # SAMPLE-LEVEL CLASSIFIER:
        self.classifier = nn.Sequential(nn.Linear(self.proj_dim+self.meta_token_dim, classifier_hidden), nn.GELU(), nn.Dropout(0.2), nn.Linear(classifier_hidden, num_classes))

        # Instance projection head for contrastive/clustering loss (operates on instance features)
        self.cluster_proj = nn.Sequential(nn.Linear(self.proj_dim + self.meta_cluster_dim, cluster_proj_hidden), nn.ReLU(inplace=True), nn.Linear(cluster_proj_hidden, cluster_proj_out))
        
        # Handles metadata conditioning
        self.meta_encoder = MedHistEncoder(input_dim = self.meta_input_dim, out_dim = self.proj_dim, hidden=max(64, meta_input_dim*4)) # learn an extra instance based on metadata
        self.meta_proj_for_classifier = nn.Linear(self.proj_dim, self.meta_token_dim)
        self.meta_proj_for_cluster = nn.Linear(self.proj_dim, self.meta_cluster_dim) # I keep this small so that model wont overfit on the metadata conditioning
        self.learned_meta_token = nn.Parameter(torch.zeros(self.proj_dim), requires_grad=True)
        # Initialization helpers (optional)
        self._init_weights()


    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if self._pca_initialized and m is self.input_proj:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None, meta: Optional[torch.Tensor] = None):
        """
        x: (B, N, input_dim)
        mask: (B, N) bool (True valid)
        meta: (B, meta_input_dim) float or None
        Returns: logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out
        """
        B, N, D_in = x.shape
        assert D_in == self.input_dim, f"Expected input_dim {self.input_dim}, got {D_in}"

        if mask is None:
            mask = torch.ones((B, N), dtype=torch.bool, device=x.device)

        x_proj = self.input_proj(x)  # (B, N, proj_dim)

        # ---- Prepare metadata token ----
        if meta is None:
            # use learned token broadcast to batch
            m_token = self.learned_meta_token.unsqueeze(0).expand(B, -1)  # (B, proj_dim)
        else:
            m_token = self.meta_encoder(meta)  # (B, proj_dim)

        # Append metadata to the input bag
        x_with_meta = torch.cat([m_token.unsqueeze(1), x_proj], dim=1)  # (B, N+1, proj_dim)
        meta_mask = torch.ones((B, 1), dtype=torch.bool, device=mask.device)
        mask_with_meta = torch.cat([meta_mask, mask], dim=1)  # (B, N+1)
        inst_with_meta_out = self.transformer(x_with_meta, src_key_padding_mask=~mask_with_meta)  # (B, N+1, proj_dim)

        # separate the metadata instance from B scans
        meta_out = inst_with_meta_out[:, 0, :]  # (B, proj_dim)
        inst_feats = inst_with_meta_out[:, 1:, :]  # (B, N, proj_dim)

        # compute gated additive attention ONLY over B scan tokens
        V = torch.tanh(self.attn_V(inst_feats))      # (B, N, H)
        U = torch.sigmoid(self.attn_U(inst_feats))   # (B, N, H)
        attn_logits = self.attn_score(V * U).squeeze(-1)  # (B, N)

        # mask padded positions
        attn_logits = attn_logits.masked_fill(~mask, float("-inf"))
        attn_probs = torch.softmax(attn_logits, dim=1)  # (B, N)

        # bag embedding = weighted sum of B scan features based on attention scores
        bag_embed = torch.bmm(attn_probs.unsqueeze(1), inst_feats).squeeze(1)  # (B, proj_dim)

        # classifier: concat B-scan_bag_embed and meta_out (project meta_out to meta_token_dim) ----
        meta_for_cls = self.meta_proj_for_classifier(meta_out)  # (B, meta_token_dim)
        classifier_input = torch.cat([bag_embed, meta_for_cls], dim=1)  # (B, proj_dim + meta_token_dim)
        logits = self.classifier(classifier_input)  # (B, num_classes)
        
        meta_small = self.meta_proj_for_cluster(meta_out)  # add a new linear layer in __init__: self.meta_proj_for_cluster = nn.Linear(proj_dim, meta_cluster_dim = 16)
        meta_expand = meta_small.unsqueeze(1).expand(-1, inst_feats.size(1), -1)  # (B, N, meta_cluster_dim = 16)
        inst_feats_cond = torch.cat([inst_feats, meta_expand], dim=-1)

        # B scan projection head for contrastive/clustering
        inst_proj = self.cluster_proj(inst_feats_cond)   # (B, N, cluster_proj_out)
        inst_proj = F.normalize(inst_proj, p=2, dim=-1)

        return logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out
