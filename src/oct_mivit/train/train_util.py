import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from oct_mivit.models import OCT_MIViT
from oct_mivit.losses import focal_loss, clustering_contrastive_loss
from oct_mivit.utils.save_util import make_model_save_path
import os

def recreate_scheduler_for_remaining(current_epoch: int, N_EPOCH: int, optimizer_obj):
    """
    Recreate a cosine annealing LR scheduler for the remaining epochs.
    This is used to change the LR schedule after unfreezing layers.
    Forward signature: new_sched = recreate_scheduler_for_remaining(current_epoch, N_EPOCH, optimizer_obj)
    """
    remaining = max(1, N_EPOCH - current_epoch + 1)
    new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_obj, T_max=remaining, eta_min=1e-6)
    new_sched.last_epoch = current_epoch - 1
    return new_sched

def train_one_epoch(model, dataloader, optimizer,
    epoch:int, N_EPOCH:int, second_cycle_start_epoch: int = 36, clust_weight: float = 0.5, unfreeze_epoch: int = 5, pca_proj_lr: float = 1e-5,
    device = torch.device("cuda:0"),
    top_k: int = 5, bottom_k: int = 5, margin: float = 0.2, scheduler = None):
    """
    Train the model for one epoch on the given dataloader.
    Uses focal loss for classification and clustering contrastive loss for instance feature separation.
    """
    device = torch.device(device)
    model.train().to(device)
    class_weights = torch.FloatTensor(np.array([1.2, 1.0])).to(device)
    focal = focal_loss(alpha=class_weights, gamma=2.0, label_smoothing=0.0)

    if (unfreeze_epoch is not None) and (epoch == unfreeze_epoch):
        if hasattr(model, "input_proj"):
            for p in model.input_proj.parameters():
                p.requires_grad = True # ALLOW MODEL TO TRAIN PCA PROJECTION LAYER
                scheduler = recreate_scheduler_for_remaining(epoch, N_EPOCH, optimizer)
        else:
            print("Warning: model has no attribute input_proj to unfreeze.")
    
    if epoch == second_cycle_start_epoch:
            scheduler = recreate_scheduler_for_remaining(epoch, N_EPOCH, optimizer)
            
    total_loss = 0.0
    
    ema_cel = 0.5
    ema_ccl = 0.5
    momentum = 0.95
    one_loss_warmup_epochs = 5
    warmup_increase_epochs = 5
    for bags, labels, masks, metas in tqdm(dataloader, desc=f"Training Epoch #{epoch}", leave=False):
        bags, labels, masks, metas = bags.to(device), labels.to(device).long(), masks.to(device), metas.to(device)

        optimizer.zero_grad()
        logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out = model(bags, mask = masks , meta = metas)
        loss_cls = focal(logits, labels)
        loss_clust = clustering_contrastive_loss(inst_proj, attn_probs, masks, labels,
                                                 top_k=top_k, bottom_k=bottom_k, margin=margin)

        ema_cel = momentum * ema_cel + (1-momentum) * loss_cls.item()
        ema_ccl = momentum * ema_ccl + (1-momentum) * loss_clust.item()
        
        if epoch < one_loss_warmup_epochs:
            loss = loss_cls/(ema_cel+1e-9)
        else:
            clust_weight_rampup_frac = min(1.0, (epoch - one_loss_warmup_epochs) / float(warmup_increase_epochs))
            clust_weight_epoch = clust_weight*clust_weight_rampup_frac
            cls_weight_epoch   = 1.0 - clust_weight_epoch
            loss = cls_weight_epoch * (loss_cls/(ema_cel+1e-9)) + clust_weight_epoch * (loss_clust/(ema_ccl+1e-9))
        
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item() * bags.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss

def test_on_dataloader(model, dataloader, device = torch.device("cuda:0")):
    """
    Test the model on the given dataloader and compute accuracy and AUC.
    Forward signature: accuracy, tpr_interp, auc_score = test_on_dataloader(model, dataloader, device)
    Where:
        accuracy: float
        tpr_interp: (101,) numpy array of TPR values interpolated at FPR points [0.0, 0.01, ..., 1.0]
        auc_score: float AUC value computed on the interpolated TPR
    """
    model.eval().to(device)
    fpr_interp_points = np.linspace(0,1,num=101,endpoint=True)
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for bags, labels, masks, metas in tqdm(dataloader, desc="Testing", leave=False):
            bags, labels, masks, metas = bags.to(device), labels.to(device).long(), masks.to(device), metas.to(device)
            logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out = model(bags, mask = masks , meta = metas)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs_pos = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    try:
        fpr, tpr, thresholds = roc_curve(all_labels.numpy(), probs_pos)
        tpr_interp = np.interp(fpr_interp_points,fpr,tpr)
        auc_score = auc(fpr_interp_points, tpr_interp)
    except Exception:
        auc_score = float("nan")
        
    preds = (probs_pos >= 0.5).astype(int)
    accuracy = sum(all_labels.numpy()==preds) / len(preds)
    return accuracy, tpr_interp, auc_score

def test_LOO_on_sample(model, dataloader, device = torch.device("cuda:0")):
    """
    Test the model on the given dataloader and compute average positive class probability
    and accuracy at the sample level (averaging predictions over all bags from the same sample).
    """
    model.eval().to(device)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for bags, labels, masks, metas in tqdm(dataloader, desc="Testing", leave=False):
            bags, labels, masks, metas = bags.to(device), labels.to(device).long(), masks.to(device), metas.to(device)
            logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out = model(bags, mask = masks , meta = metas)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs_pos = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    avg_probs_pos = np.mean(probs_pos)
    preds = (probs_pos >= 0.5).astype(int)
    accuracy = sum(all_labels.numpy()==preds) / len(preds)
    return avg_probs_pos , accuracy

def test_LOO_on_specimen(model, dataloader, device = torch.device("cuda:0")):
    """
    Test the model on the given dataloader and compute average positive class probability separately for each specimen aspect.
    Forward signature: avg_probs_pos_side1, avg_probs_pos_side2 = test_LOO_on_specimen(model, dataloader, device)
    Where:
        avg_probs_pos_side1: float average positive class probability for the first aspect of the specimen
        avg_probs_pos_side2: float average positive class probability for the second aspect of the specimen
    """
    model.eval().to(device)
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for bags, labels, masks, metas in tqdm(dataloader, desc="Testing", leave=False):
            bags, labels, masks, metas = bags.to(device), labels.to(device).long(), masks.to(device), metas.to(device)
            logits, attn_probs, inst_feats, inst_proj, bag_embed, meta_out = model(bags, mask = masks , meta = metas)

            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    probs_pos = torch.softmax(all_logits, dim=1)[:, 1].numpy()
    N_pred = len(probs_pos)
    avg_probs_pos_side1 = np.mean(probs_pos[0:N_pred//2])
    avg_probs_pos_side2 = np.mean(probs_pos[N_pred//2:N_pred])
    return avg_probs_pos_side1, avg_probs_pos_side2

def run(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means,
        N_EPOCH:int, second_cycle_start_epoch: int, clust_weight: float, device = torch.device("cuda:0")):
    """
    Main training and evaluation loop for the OCT_MIViT model.
    Uses PCA initialization for the input projection layer.
    Forward signature: perf_results = run(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means,
                                          N_EPOCH, second_cycle_start_epoch, clust_weight, device)
    Where:
        perf_results: dict with keys:  
            "valid_acc_log": (N_EPOCH,) numpy array of validation accuracies per epoch
            "valid_auc_log": (N_EPOCH,) numpy array of validation AUCs per epoch
            "test_acc_log":  (N_EPOCH,) numpy array of test accuracies per epoch
            "test_auc_log":  (N_EPOCH,) numpy array of test AUCs per epoch
            "final_test_accuracy": float final test accuracy after training
            "final_test_auc": float final test AUC after training
            "final_test_roc": (101,) numpy array of TPR values at FPR points [0.0, 0.01, ..., 1.0] after training
    """
    test_bag, _, _ = next(iter(test_dataloader))
    embedding_dim = test_bag.shape[-1] # dimension of the input features
    model = OCT_MIViT(input_dim = embedding_dim, proj_dim = 512, pca_components = pca_components, pca_mean = pca_means)
    
    base_params = [p for n,p in model.named_parameters() if "input_proj" not in n]
    proj_params = list(model.input_proj.parameters())

    base_lr = 1e-4
    proj_lr = 1e-5 # low LR for PCA projection layer to avoid destroying the PCA initialization too quickly
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": base_lr},
        {"params": proj_params, "lr": proj_lr}
        ], weight_decay=1e-6)
    
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCH, eta_min=1e-6)
    model = model.to(device)
    # track validation and test accuracy and AUC for each epoch
    valid_acc_log = np.zeros(N_EPOCH)
    valid_auc_log = np.zeros(N_EPOCH)
    test_acc_log  = np.zeros(N_EPOCH)
    test_auc_log  = np.zeros(N_EPOCH)
    best_acc = 0
    
    model_savepath = make_model_save_path(filename="best_model_wts.pth", subfolder="model_state_dict_logs")
    for epoch in range(1,N_EPOCH+1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch, N_EPOCH, second_cycle_start_epoch, clust_weight, scheduler = optimizer_scheduler)
        valid_accuracy, _, valid_auc = test_on_dataloader(model, valid_dataloader, device)
        test_accuracy,  _, test_auc  = test_on_dataloader(model, test_dataloader,  device)
        
        valid_acc_log[epoch-1] = valid_accuracy
        valid_auc_log[epoch-1] = valid_auc
        test_acc_log[epoch-1]  = test_accuracy
        test_auc_log[epoch-1]  = test_auc
        
        if valid_accuracy > best_acc and epoch > 15:
            torch.save(model.state_dict(), model_savepath)
            print('Updated model parameters saved.')
            best_acc = valid_accuracy
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Validation Acc = {valid_accuracy:.4f}, Test Acc = {test_accuracy:.4f}')
    
    model.load_state_dict(torch.load(model_savepath))
    test_accuracy_f,  tpr_interp_f,  test_auc_f  = test_on_dataloader(model, test_dataloader, device)
    perf_results = {"valid_acc_log": valid_acc_log, "valid_auc_log": valid_auc_log,
                    "test_acc_log": test_acc_log,   "test_auc_log": test_auc_log,
                    "final_test_accuracy": test_accuracy_f, "final_test_auc": test_auc_f, "final_test_roc": tpr_interp_f}
    return perf_results

def run_LOO_on_sample(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means, 
                      N_EPOCH:int, second_cycle_start_epoch: int, clust_weight: float, device = torch.device("cuda:0")):
    """
    Main training and evaluation loop for the OCT_MIViT model using leave-one-out cross-validation at the sample level.
    Uses PCA initialization for the input projection layer.
    Forward signature: perf_results = run_LOO_on_sample(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means,
                                                      N_EPOCH, second_cycle_start_epoch, clust_weight, device)
    Where:
        perf_results: dict with keys:
            "valid_acc_log": (N_EPOCH,) numpy array of validation accuracies per epoch
            "valid_auc_log": (N_EPOCH,) numpy array of validation AUCs per epoch
            "sample_prediction_score": float average positive class probability on the test set after training
            "sample_prediction_accuracy": float accuracy on the test set after training
    """
    test_bag, _, _, _ = next(iter(test_dataloader))
    embedding_dim = test_bag.shape[-1]
    model = OCT_MIViT(input_dim = embedding_dim, proj_dim = 512, pca_components = pca_components, pca_mean = pca_means)
    
    base_params = [p for n,p in model.named_parameters() if "input_proj" not in n]
    proj_params = list(model.input_proj.parameters())

    base_lr = 1e-4
    proj_lr = 1e-5
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": base_lr},
        {"params": proj_params, "lr": proj_lr}
        ], weight_decay=1e-6)
    
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCH, eta_min=1e-6)
    model = model.to(device)
    # track validation accuracy and AUC for each epoch
    valid_acc_log = np.zeros(N_EPOCH)
    valid_auc_log = np.zeros(N_EPOCH)
    best_acc = 0

    model_savepath = make_model_save_path(filename="best_model_wts_loo_sample.pth", subfolder="model_state_dict_logs")
    for epoch in range(1,N_EPOCH+1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch, N_EPOCH, second_cycle_start_epoch, clust_weight, scheduler = optimizer_scheduler)
        valid_accuracy, _, valid_auc = test_on_dataloader(model, valid_dataloader, device)
        valid_acc_log[epoch-1] = valid_accuracy
        valid_auc_log[epoch-1] = valid_auc
        
        if valid_accuracy > best_acc and epoch > 15:
            torch.save(model.state_dict(), model_savepath)
            print('model saved.')
            best_acc = valid_accuracy
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Validation Acc = {valid_accuracy:.4f}')
    
    model.load_state_dict(torch.load(model_savepath))    
    test_predscore_f , test_accuracy_f  = test_LOO_on_sample(model, test_dataloader, device)
    perf_results = {"valid_acc_log": valid_acc_log, "valid_auc_log": valid_auc_log,
                    "sample_prediction_score": test_predscore_f, "sample_prediction_accuracy": test_accuracy_f}
    return perf_results

def run_v3_LOO_specimen(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means,
           N_EPOCH:int, second_cycle_start_epoch: int, clust_weight: float, device = torch.device("cuda:0")):
    """
    Main training and evaluation loop for the OCT_MIViT model using leave-one-out cross-validation at the specimen level.
    Uses PCA initialization for the input projection layer.
    Forward signature: perf_results = run_v3_LOO_specimen(train_dataloader , valid_dataloader , test_dataloader, pca_components, pca_means,
                                                        N_EPOCH, second_cycle_start_epoch, clust_weight, device)
    Where:
        perf_results: dict with keys:
            "valid_acc_log": (N_EPOCH,) numpy array of validation accuracies per epoch
            "valid_auc_log": (N_EPOCH,) numpy array of validation AUCs per epoch
            "sample1_prediction_score": float average positive class probability for the first aspect of the specimen after training
            "sample2_prediction_score": float average positive class probability for the second aspect of the specimen after training
    """
    test_bag, _, _, _ = next(iter(test_dataloader))
    embedding_dim = test_bag.shape[-1]
    model = OCT_MIViT(input_dim = embedding_dim, proj_dim = 512, pca_components = pca_components, pca_mean = pca_means)
    
    base_params = [p for n,p in model.named_parameters() if "input_proj" not in n]
    proj_params = list(model.input_proj.parameters())

    base_lr = 1e-4
    proj_lr = 1e-5
    optimizer = torch.optim.AdamW([
        {"params": base_params, "lr": base_lr},
        {"params": proj_params, "lr": proj_lr}
        ], weight_decay=1e-6)
    
    optimizer_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCH, eta_min=1e-6)
    model = model.to(device)
    # track validation accuracy and AUC for each epoch
    valid_acc_log = np.zeros(N_EPOCH)
    valid_auc_log = np.zeros(N_EPOCH)
    best_acc = 0

    model_savepath = make_model_save_path(filename="best_model_wts_loo_specimen.pth", subfolder="model_state_dict_logs")
    for epoch in range(1,N_EPOCH+1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch, N_EPOCH, second_cycle_start_epoch, clust_weight, scheduler = optimizer_scheduler)
        valid_accuracy, _, valid_auc = test_on_dataloader(model, valid_dataloader, device)
        valid_acc_log[epoch-1] = valid_accuracy
        valid_auc_log[epoch-1] = valid_auc
        
        if valid_accuracy > best_acc and epoch > 15:
            torch.save(model.state_dict(), model_savepath)
            print('model saved.')
            best_acc = valid_accuracy
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}, Validation Acc = {valid_accuracy:.4f}')
    
    model.load_state_dict(torch.load(model_savepath))    
    test_predscore_f_side1, test_predscore_f_side2 = test_LOO_on_specimen(model, test_dataloader, device)
    perf_results = {"valid_acc_log": valid_acc_log, "valid_auc_log": valid_auc_log,
                    "sample1_prediction_score": test_predscore_f_side1, "sample2_prediction_score": test_predscore_f_side2}
    return perf_results