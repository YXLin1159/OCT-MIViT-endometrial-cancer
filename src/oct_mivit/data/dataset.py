import os
from pathlib import Path
from typing import List, Optional, Union
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import random

class EmbeddingDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        model_idx: int,
        benign_indices: Optional[List[int]] = None,
        cancer_indices: Optional[List[int]] = None,
        max_instances: int = 150,
        num_partitions: int = 20,
        preload_workers: int = 16,
        transform=None,
        use_partitions: bool = True,
        metadata_csv: Optional[str] = None,
        metadata_id_col: Union[int, str] = 0,                               # column index for patientID
        metadata_side_col: Union[int, str] = 1,                             # column index for side (anterior or posterior)
        metadata_cols: Optional[List[Union[int, str]]] = [2,3,4,5,6,7],     # list of columns to include as metadata; if None use side only
        debug: bool = False,
    ):
        super().__init__()
        self.root = Path(root_dir)
        self.transform = transform
        self.use_partitions = use_partitions
        self.max_instances = max_instances
        self.num_partitions = num_partitions
        self.model_idx = model_idx
        self.debug = debug

        self.bens = set(benign_indices) if (benign_indices is not None) else None
        self.cans = set(cancer_indices) if (cancer_indices is not None) else None

        # load metadata csv
        self.metadata_csv = metadata_csv
        self.metadata_map = {}   # keys -> (patient_id_str, side_key) -> meta vector (np.float32)
        self.meta_dim = len(metadata_cols)
        self._warned_missing_meta = False

        if metadata_csv is not None:
            df = pd.read_csv(metadata_csv, dtype=str, header=0)
            if self.debug:
                print(f"[EmbeddingDataset] Loaded metadata CSV: {metadata_csv}, shape={df.shape}")

            def _get_series(col):
                if isinstance(col, int):
                    return df.iloc[:, col].astype(str).str.strip()
                else:
                    return df[col].astype(str).str.strip()

            id_series = _get_series(metadata_id_col)
            side_series = _get_series(metadata_side_col)

            if metadata_cols is None:
                feature_cols = [metadata_side_col]
            else:
                feature_cols = metadata_cols

            # collect feature series order
            feature_series_list = []
            for c in feature_cols:
                feature_series_list.append(_get_series(c))

            # Build mapping
            for i in range(len(df)):
                pid = str(id_series.iat[i]).strip() # patient ID
                side_raw = str(side_series.iat[i]).strip().lower()
                side_num = int(side_raw)
                side_text = 'anterior' if side_num == 0 else 'posterior' if side_num == 1 else side_raw

                # build feature vector (as floats when possible)
                feat = []
                for s in feature_series_list:
                    val = str(s.iat[i]).strip()
                    try:
                        fv = float(val)
                    except Exception:
                        fv = float(abs(hash(val)) % 1000)
                    feat.append(fv)
                feat_arr = np.asarray(feat, dtype=np.float32)

                if side_text is not None:
                    self.metadata_map[(pid, side_text)] = feat_arr

        entries = []
        for label_folder in sorted(self.root.iterdir()):
            if not label_folder.is_dir():
                continue
            label = 0 if label_folder.name.startswith('0') else 1
            subs = [d for d in sorted(label_folder.iterdir()) if d.is_dir()]
            # filter via provided per-label indices
            if label == 0 and (self.bens is not None):
                subs = [s for i, s in enumerate(subs) if i in self.bens]
            if label == 1 and (self.cans is not None):
                subs = [s for i, s in enumerate(subs) if i in self.cans]

            for fold in subs:
                if self.use_partitions:
                    for chunk in sorted(fold.glob('partition_*')):
                        if chunk.is_dir():
                            entries.append((chunk, label, fold))  # store fold (patient folder) for metadata mapping
                else:
                    entries.append((fold, label, fold))

        def _lookup_meta_for_fold(fold_path: Path):
            """
            fold_path: patient folder Path (e.g., .../19_anterior)
            returns: numpy array shape (meta_dim,) of float32, or zeros if not found
            """
            if self.metadata_csv is None or self.meta_dim == 0:
                return None  # metadata disabled

            patient_folder_name = fold_path.name  # e.g., '19_anterior'
            if "_" in patient_folder_name:
                pid_str, side_str = patient_folder_name.rsplit("_", 1)
            else:
                pid_str = patient_folder_name
                side_str = ""

            pid_str = str(pid_str).strip()
            side_str = str(side_str).strip().lower()

            # try lookups in order: (pid, side_text), (pid, side_numeric_string), (pid,)
            candidate = None
            if (pid_str, side_str) in self.metadata_map:
                candidate = self.metadata_map[(pid_str, side_str)]
                return candidate

            if not self._warned_missing_meta:
                print(f"[EmbeddingDataset_v2] WARNING: metadata row not found for patient folder '{patient_folder_name}'. Returning zeros for meta. Path looked up in CSV: {self.metadata_csv}")
                self._warned_missing_meta = True
            return np.zeros((self.meta_dim,), dtype=np.float32)

        def load_one(entry):
            folder, lbl, fold_for_meta = entry
            insts = []
            # load .npy files inside partition (or fold)
            for p in sorted(folder.glob('*.npy')):
                arr = np.load(p) # model_idx = 2 for model with 8192-dim embeddings (fused image input types)
                if self.model_idx == 0:
                    arr = arr[0:4096] # model_idx = 0 for 4096-dim embeddings (grayscale only)
                elif self.model_idx == 1:
                    arr = arr[4096:8192] # model_idx = 1 for 4096-dim embeddings (pseudo-RGB only)
                insts.append(torch.from_numpy(arr).float())
            if not insts:
                return None
            bag = torch.stack(insts, 0)  # (N, D)
            N, D = bag.shape
            M = self.max_instances or N
            if N < M:
                pad = bag.new_zeros((M - N, D))
                bag = torch.cat([bag, pad], 0)
            else:
                bag = bag[:M]
            mask = torch.zeros(M, dtype=torch.bool)
            mask[: min(N, M)] = True

            # lookup metadata for this patient fold
            meta_vec = _lookup_meta_for_fold(fold_for_meta)
            if meta_vec is None:
                meta_tensor = None
            else:
                meta_tensor = torch.from_numpy(np.asarray(meta_vec, dtype=np.float32))

            return bag, lbl, mask, meta_tensor

        with ThreadPoolExecutor(max_workers=preload_workers) as exe:
            results = list(exe.map(load_one, entries))

        self.data = []
        self.labels = []
        self.masks = []
        self.metas = []
        for r in results:
            if r is None:
                continue
            bag, lbl, mask, meta = r
            self.data.append(bag)
            self.labels.append(lbl)
            self.masks.append(mask)
            if meta is None:
                self.metas.append(torch.empty(0))
            else:
                self.metas.append(meta)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.masks[idx], self.metas[idx]

def get_embedding_dataloader(
    root_dir: str,
    model_idx: int,
    batch_size: int,
    benign_indices: list,
    cancer_indices: list,
    metadata_path: str,
    max_instances: int = 150,
    needShuffle: bool = True,
    collate_fn = None,
    num_workers: int = 0,
) -> DataLoader:
    """
    Utility function to create a DataLoader for the EmbeddingDataset.
    Returns a DataLoader instance.
    """
    dataset = EmbeddingDataset(root_dir = root_dir,
                                  model_idx = model_idx,
                                  benign_indices = benign_indices, 
                                  cancer_indices = cancer_indices,
                                  max_instances  = max_instances,
                                  metadata_csv   = metadata_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=needShuffle, num_workers=num_workers, collate_fn=collate_fn)
    return dataloader

def splitData2FolderIdx_BINARY(main_folder_path: str):
    """
    Split the data folders in the sample-level into training, validation, and test sets for binary classification.
    The split is done by stratification per class (benign and cancer) with 60% training, 20% validation, and 20% test.
    Args:
        main_folder_path (str): Path to the main folder containing category subfolders.
    Returns:
        dict: A dictionary with keys 'benign_train', 'benign_valid', 'benign_test',
              'cancer_train', 'cancer_valid', 'cancer_test', each containing a list of folder indices.
    """
    benign_train_folder_list = []
    benign_valid_folder_list = []
    benign_test_folder_list = []
    
    cancer_train_folder_list = []
    cancer_valid_folder_list = []
    cancer_test_folder_list = []

    category_name = [f.name for f in os.scandir(main_folder_path) if f.is_dir()]
    N_class = len(category_name)

    for i in range(N_class): # ITERATE OVER EACH CATEGORY
        path_category_tmp = os.path.join(main_folder_path , category_name[i])
        N_category_tmp = sum(1 for p in Path(path_category_tmp).iterdir() if p.is_dir())

        N_validation_tmp = int(np.maximum( np.round(N_category_tmp*0.2) , 2))
        N_test_tmp       = int(np.maximum( np.round(N_category_tmp*0.2) , 1))
        N_train_tmp      = N_category_tmp - N_validation_tmp - N_test_tmp

        idx_tmp            = np.random.permutation(N_category_tmp)
        if (i==0):
            benign_train_folder_list += [idx_tmp[i] for i in idx_tmp[0:N_train_tmp]]
            benign_valid_folder_list += [idx_tmp[i] for i in idx_tmp[N_train_tmp:(N_train_tmp+N_validation_tmp)]]
            benign_test_folder_list  += [idx_tmp[i] for i in idx_tmp[(N_train_tmp+N_validation_tmp):N_category_tmp]]
            
        elif (i==1):
            cancer_train_folder_list += [idx_tmp[i] for i in idx_tmp[0:N_train_tmp]]
            cancer_valid_folder_list += [idx_tmp[i] for i in idx_tmp[N_train_tmp:(N_train_tmp+N_validation_tmp)]]
            cancer_test_folder_list  += [idx_tmp[i] for i in idx_tmp[(N_train_tmp+N_validation_tmp):N_category_tmp]]

    split_output = {
        "benign_train": benign_train_folder_list,
        "benign_valid": benign_valid_folder_list,
        "benign_test":  benign_test_folder_list,
        "cancer_train": cancer_train_folder_list,
        "cancer_valid": cancer_valid_folder_list,
        "cancer_test":  cancer_test_folder_list
    }
    return split_output

def splitData2FolderIdx_BINARY_specimen(main_folder_path: str, patientID: int, val_frac: float = 0.2):
    """
    Split the data folders in the specimen-level into training, validation, and test sets for binary classification,
    ensuring that the specified patient's anterior and posterior folders are included in the test set if they exist.
    The remaining folders are split into training and validation sets based on the specified validation fraction (default 20%).
    Args:
        main_folder_path (str): Path to the main folder containing category subfolders.
        patientID (int): The patient ID whose folders should be included in the test set if they exist.
        val_frac (float): Fraction of remaining data to use for validation (default is 0.2).
    Returns:
        dict: A dictionary with keys 'benign_train', 'benign_valid', 'benign_test',
              'cancer_train', 'cancer_valid', 'cancer_test', each containing a list of folder indices.
    """
    pid_str = f"{int(patientID):02d}"
    class_folders = {"benign": "0_Benign", "cancer": "1_Cancer"}
    # Prepare outputs
    benign_train, benign_val, benign_test = [], [], []
    cancer_train, cancer_val, cancer_test = [], [], []
    rnd = random.Random()

    for cls_name, cls_folder in class_folders.items():
        cls_path = os.path.join(main_folder_path, cls_folder)
        if not os.path.isdir(cls_path):
            raise ValueError(f"Expected class folder not found: '{cls_path}'")

        # get subfolders (only directories)
        subfolders = sorted(
            [d for d in os.listdir(cls_path) if os.path.isdir(os.path.join(cls_path, d))]
        )

        # locate patient's anterior/posterior in this class (if present) -> test set
        name_to_idx = {name: idx for idx, name in enumerate(subfolders)}
        test_indices = []
        for aspect in ("anterior", "posterior"):
            candidate_name = f"{pid_str}_{aspect}"
            if candidate_name in name_to_idx:
                test_indices.append(name_to_idx[candidate_name])

        remaining_indices = [i for i in range(len(subfolders)) if i not in test_indices]
        rnd.shuffle(remaining_indices)

        # compute validation count; ensure at least 1 val if there are >1 remaining
        n_rem = len(remaining_indices)
        n_val = int(n_rem * val_frac)
        if n_val == 0 and n_rem > 1:
            n_val = 1

        val_indices = remaining_indices[:n_val]
        train_indices = remaining_indices[n_val:]

        if cls_name == "benign":
            benign_test.extend(sorted(test_indices))
            benign_val.extend(sorted(val_indices))
            benign_train.extend(sorted(train_indices))
        else:  # cancer
            cancer_test.extend(sorted(test_indices))
            cancer_val.extend(sorted(val_indices))
            cancer_train.extend(sorted(train_indices))
    
    split_output = {
        "benign_train": benign_train,
        "benign_valid": benign_val,
        "benign_test":  benign_test,
        "cancer_train": cancer_train,
        "cancer_valid": cancer_val,
        "cancer_test":  cancer_test,
    }
    return split_output