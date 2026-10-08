#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Leakage-Controlled Evaluation of Machine Learning Landslide Susceptibility
===========================================================================
Full end-to-end pipeline reproducing every table and figure from the paper.

Inputs (must exist in ./data/):
    samples_V2.npz      -- x90, terrain, y, seq, point, group, source_id,
                           year, month, lat, lon, split, lon_block
    pointsplit_V2.npz   -- new_split (array of "train"/"val"/"test")

Optional (if samples_V2_imerg.npz exists -> feature sets E and F enabled):
    samples_V2_imerg.npz -- same keys plus `trigger` (n, 17) and `trigger_names`

Outputs (into ./output/):
    table05_main_grid.csv          -- 8 models x 4 protocols (paper Table 5)
    table06_featuresets.csv        -- feature sets x protocols
    table07_ablation.csv           -- ST-CoupleNet structured ablation
    table08_variogram.csv
    table08_kish.csv
    table_delong.csv
    table_bootstrap.csv
    fig06_roc.png                  -- ROC curves, 4 panels
    fig07_heatmap.png              -- AUC heatmap
    fig08_featureset_bars.png      -- terrain vs terrain+temporal
    fig10_variogram.png
    fig11_confusion.png            -- P3 confusion matrices, all models
    fig_training_curves.png        -- validation-AUC curves for deep models
    confusion_metrics_pointsplit.csv
"""

from __future__ import annotations
import os, sys, json, math, time, warnings, logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata, norm

from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, StratifiedGroupKFold
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="[%(asctime)s] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("lsm")


# ═══════════════════════════════════════════════════════════════════════════
# 0. CONFIG
# ═══════════════════════════════════════════════════════════════════════════

def load_config(path: str = "config.yaml") -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data(cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Load samples_V2.npz and pointsplit_V2.npz into one dict."""
    ddir = Path(cfg["data"]["dir"])
    f_samples = ddir / cfg["data"]["samples_npz"]
    f_split   = ddir / cfg["data"]["pointsplit_npz"]
    if not f_samples.exists():
        raise FileNotFoundError(f"missing {f_samples}")
    if not f_split.exists():
        raise FileNotFoundError(f"missing {f_split}")

    d = dict(np.load(f_samples, allow_pickle=True))
    sp = np.load(f_split, allow_pickle=True)
    d["new_split"] = sp["new_split"]                     # "train" / "val" / "test"
    if "keep_buffered" in sp:
        d["keep_buffered"] = sp["keep_buffered"]

    # optional: IMERG-augmented file
    f_imerg = ddir / cfg["data"].get("samples_imerg_npz", "")
    if f_imerg.name and f_imerg.exists():
        d_im = dict(np.load(f_imerg, allow_pickle=True))
        if "trigger" in d_im:
            d["trigger"] = d_im["trigger"]
            d["trigger_names"] = d_im.get("trigger_names", None)
            log.info(f"IMERG triggers loaded: shape={d['trigger'].shape}")

    log.info(f"loaded: n={len(d['y'])}, "
             f"x90={d['x90'].shape}, terrain={d['terrain'].shape}, "
             f"seq={d['seq'].shape}")
    return d


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE SETS
# ═══════════════════════════════════════════════════════════════════════════

def build_feature_set(d: Dict[str, np.ndarray], name: str,
                      cfg: Dict[str, Any]) -> Tuple[np.ndarray, Optional[int]]:
    """Return (X, lithology_index_in_X or None)."""
    x90 = d["x90"].astype(np.float32)
    ter = d["terrain"].astype(np.float32)
    LITH_T = cfg["features"]["lithology_terrain_col"]
    LITH_C = cfg["features"]["lithology_C_col"]

    if name == "A":
        return x90, None
    if name == "B":
        return ter, LITH_T
    if name == "C":
        return np.hstack([x90, ter]), LITH_C
    if name in ("E", "F"):
        if "trigger" not in d:
            raise KeyError(f"feature set {name} requires samples_V2_imerg.npz "
                           f"(with `trigger` array) — run gee/07c_merge_imerg.py first")
        trig = d["trigger"].astype(np.float32)
        if name == "E":
            return np.hstack([ter, trig]), LITH_T
        if name == "F":
            return np.hstack([x90, ter, trig]), LITH_C
    raise ValueError(f"unknown feature set: {name}")


# ═══════════════════════════════════════════════════════════════════════════
# 3. CONNECTED-COMPONENT LEAKAGE CLUSTERING (Algorithm 4)
# ═══════════════════════════════════════════════════════════════════════════

class UnionFind:
    def __init__(self, n: int): self.p = list(range(n))
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def build_clusters(d: Dict[str, np.ndarray],
                   cell_size_deg: float = 0.1) -> np.ndarray:
    """Join samples sharing point, 0.1-deg cell, or source-event id."""
    n = len(d["y"])
    lon = d["lon"]; lat = d["lat"]
    cell = (np.floor(lon / cell_size_deg).astype(np.int64) * 100_000
            + np.floor(lat / cell_size_deg).astype(np.int64))

    # vectorised edge construction via sorting
    def edges(arr: np.ndarray):
        o = np.argsort(arr, kind="stable"); s = arr[o]
        ei, ej = [], []
        st = 0
        for k in range(1, n + 1):
            if k == n or s[k] != s[st]:
                b = o[st:k]
                if len(b) > 1:
                    ei += [b[0]] * (len(b) - 1)
                    ej += list(b[1:])
                st = k
        return ei, ej

    ei_all, ej_all = [], []
    for arr in (d["point"], cell, d["source_id"]):
        ei, ej = edges(arr)
        ei_all += ei; ej_all += ej
    adj = coo_matrix((np.ones(len(ei_all)), (ei_all, ej_all)), shape=(n, n))
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


# ═══════════════════════════════════════════════════════════════════════════
# 4. PROTOCOLS (Algorithm 5)
# ═══════════════════════════════════════════════════════════════════════════

def get_folds(protocol: str, d: Dict[str, np.ndarray],
              clusters: np.ndarray, cfg: Dict[str, Any]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return a list of (train_idx, eval_idx) tuples."""
    seed = cfg["validation"]["seed"]
    n_folds = cfg["validation"]["n_folds"]
    n = len(d["y"]); y = d["y"]

    if protocol == "P1":    # random stratified
        return list(StratifiedKFold(n_folds, shuffle=True, random_state=seed)
                    .split(np.zeros(n), y))
    if protocol == "P2":    # spatial GroupKFold on 0.1-deg cell
        cell_deg = cfg["validation"]["cell_size_deg"]
        cell = (np.floor(d["lon"] / cell_deg).astype(np.int64) * 100_000
                + np.floor(d["lat"] / cell_deg).astype(np.int64))
        return list(GroupKFold(n_folds).split(np.zeros(n), y, cell))
    if protocol == "P4":    # cluster-disjoint StratifiedGroupKFold
        return list(StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
                    .split(np.zeros(n), y, clusters))
    if protocol == "P3":    # temporal point-split
        ns = d["new_split"]
        tr = np.where(ns == "train")[0]
        va = np.where(ns == "val")[0]
        te = np.where(ns == "test")[0]
        return [(tr, te)]   # (train, eval); `va` used inside deep models for early stop
    raise ValueError(f"unknown protocol {protocol}")


def p3_validation_idx(d: Dict[str, np.ndarray]) -> np.ndarray:
    ns = d["new_split"]
    return np.where(ns == "val")[0]


# ═══════════════════════════════════════════════════════════════════════════
# 5. DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════

def prep_sklearn(X: np.ndarray, tr: np.ndarray, ev: np.ndarray,
                 lith_col: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Fit imputation + scaling on train only."""
    imp = SimpleImputer(strategy="median").fit(X[tr])
    Xtr = imp.transform(X[tr]); Xev = imp.transform(X[ev])
    # scale all columns except lithology (categorical)
    mask = np.ones(Xtr.shape[1], bool)
    if lith_col is not None:
        mask[lith_col] = False
    sc = StandardScaler().fit(Xtr[:, mask])
    Xtr_s = Xtr.copy(); Xev_s = Xev.copy()
    Xtr_s[:, mask] = sc.transform(Xtr[:, mask])
    Xev_s[:, mask] = sc.transform(Xev[:, mask])
    return Xtr_s.astype(np.float32), Xev_s.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 6. SKLEARN MODELS
# ═══════════════════════════════════════════════════════════════════════════

def make_sklearn_model(name: str, cfg: Dict[str, Any], n_features: int):
    m = cfg["models"]
    if name == "LR":
        return LogisticRegression(C=m["logistic_regression"]["C"],
                                  max_iter=m["logistic_regression"]["max_iter"])
    if name == "RF":
        return RandomForestClassifier(
            n_estimators=m["random_forest"]["n_estimators"],
            max_depth=m["random_forest"]["max_depth"],
            max_features=m["random_forest"]["max_features"],
            random_state=cfg["validation"]["seed"], n_jobs=-1)
    if name == "GBT":
        return GradientBoostingClassifier(
            n_estimators=m["gradient_boosting"]["n_estimators"],
            max_depth=m["gradient_boosting"]["max_depth"],
            learning_rate=m["gradient_boosting"]["learning_rate"],
            subsample=m["gradient_boosting"]["subsample"],
            random_state=cfg["validation"]["seed"])
    if name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=tuple(m["mlp"]["hidden"]),
            alpha=m["mlp"]["alpha"], max_iter=m["mlp"]["max_iter"],
            early_stopping=m["mlp"]["early_stopping"],
            random_state=cfg["validation"]["seed"])
    raise ValueError(name)


def run_sklearn_protocol(name: str, X: np.ndarray, y: np.ndarray,
                         folds: List[Tuple[np.ndarray, np.ndarray]],
                         lith_col: Optional[int],
                         cfg: Dict[str, Any]) -> np.ndarray:
    n = len(y)
    p = np.full(n, np.nan, dtype=np.float32)
    for tr, ev in folds:
        Xtr, Xev = prep_sklearn(X, tr, ev, lith_col)
        model = make_sklearn_model(name, cfg, X.shape[1])
        model.fit(Xtr, y[tr])
        p[ev] = model.predict_proba(Xev)[:, 1]
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 7. CATBOOST
# ═══════════════════════════════════════════════════════════════════════════

def _catboost_frame(X: np.ndarray, lith_col: Optional[int]) -> pd.DataFrame:
    """CatBoost needs a DataFrame with the categorical column as int-strings."""
    df = pd.DataFrame(X)
    if lith_col is not None:
        col = df[lith_col].round().astype("Int64")
        df[lith_col] = col.astype(str).replace("<NA>", "missing")
    return df


def run_catboost_protocol(X: np.ndarray, y: np.ndarray,
                          folds: List[Tuple[np.ndarray, np.ndarray]],
                          lith_col: Optional[int],
                          cfg: Dict[str, Any]) -> np.ndarray:
    from catboost import CatBoostClassifier, Pool
    n = len(y)
    p = np.full(n, np.nan, dtype=np.float32)
    cat_features = [lith_col] if lith_col is not None else []
    cb_cfg = cfg["models"]["catboost"]
    for tr, ev in folds:
        Xtr_df = _catboost_frame(X[tr], lith_col)
        Xev_df = _catboost_frame(X[ev], lith_col)
        m = CatBoostClassifier(iterations=cb_cfg["iterations"],
                               depth=cb_cfg["depth"],
                               learning_rate=cb_cfg["learning_rate"],
                               l2_leaf_reg=cb_cfg["l2_leaf_reg"],
                               random_seed=cfg["validation"]["seed"],
                               verbose=0, thread_count=-1)
        m.fit(Pool(Xtr_df, y[tr], cat_features=cat_features))
        p[ev] = m.predict_proba(Xev_df)[:, 1]
    return p


# ═══════════════════════════════════════════════════════════════════════════
# 8. TABNET
# ═══════════════════════════════════════════════════════════════════════════

def run_tabnet_protocol(X: np.ndarray, y: np.ndarray,
                        folds: List[Tuple[np.ndarray, np.ndarray]],
                        val_idx: Optional[np.ndarray],
                        lith_col: Optional[int],
                        cfg: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[float]]]:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(y); p = np.full(n, np.nan, dtype=np.float32)
    hist = None
    tn_cfg = cfg["models"]["tabnet"]
    rng = np.random.default_rng(cfg["validation"]["seed"])

    for fold_i, (tr, ev) in enumerate(folds):
        # hold out 15% of training fold for TabNet's own early stopping
        idx = rng.permutation(len(tr)); k = int(0.85 * len(idx))
        itr, iva = tr[idx[:k]], tr[idx[k:]]
        Xtr, Xva = prep_sklearn(X, itr, iva, lith_col)
        Xev, _   = prep_sklearn(X, itr, ev, lith_col)
        m = TabNetClassifier(seed=cfg["validation"]["seed"] + fold_i,
                             verbose=0, device_name=device)
        m.fit(Xtr, y[itr],
              eval_set=[(Xva, y[iva])],
              eval_metric=["auc"],
              max_epochs=tn_cfg["max_epochs"],
              patience=tn_cfg["patience"],
              batch_size=tn_cfg["batch_size"])
        p[ev] = m.predict_proba(Xev)[:, 1]
        if fold_i == 0 and "val_0_auc" in m.history:
            hist = list(m.history["val_0_auc"])
    return p, hist


# ═══════════════════════════════════════════════════════════════════════════
# 9. FT-TRANSFORMER
# ═══════════════════════════════════════════════════════════════════════════

def run_ft_protocol(X: np.ndarray, y: np.ndarray,
                    folds: List[Tuple[np.ndarray, np.ndarray]],
                    lith_col: Optional[int],
                    cfg: Dict[str, Any]) -> Tuple[np.ndarray, Optional[List[float]]]:
    import torch
    import torch.nn as nn
    from rtdl_revisiting_models import FTTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ft = cfg["models"]["ft_transformer"]
    n = len(y); p = np.full(n, np.nan, dtype=np.float32)
    hist0 = None

    for fold_i, (tr, ev) in enumerate(folds):
        torch.manual_seed(cfg["validation"]["seed"] + fold_i)
        # hold out 15% of training fold
        rng = np.random.default_rng(cfg["validation"]["seed"] + fold_i)
        idx = rng.permutation(len(tr)); k = int(0.85 * len(idx))
        itr, iva = tr[idx[:k]], tr[idx[k:]]

        Xtr, Xva = prep_sklearn(X, itr, iva, lith_col)
        Xev, _   = prep_sklearn(X, itr, ev, lith_col)

        Xtr_t = torch.tensor(Xtr, device=device)
        Xva_t = torch.tensor(Xva, device=device)
        Xev_t = torch.tensor(Xev, device=device)
        ytr_t = torch.tensor(y[itr].astype(np.float32), device=device)

        model = FTTransformer(
            n_cont_features=Xtr.shape[1], cat_cardinalities=[], d_out=1,
            n_blocks=ft["n_blocks"], d_block=ft["d_block"],
            attention_n_heads=ft["n_heads"],
            attention_dropout=ft["attention_dropout"],
            ffn_d_hidden=None, ffn_d_hidden_multiplier=4/3,
            ffn_dropout=ft["ffn_dropout"], residual_dropout=0.0
        ).to(device)
        opt = torch.optim.AdamW(model.parameters(),
                                lr=ft["lr"], weight_decay=ft["weight_decay"])
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, wait = -1, None, 0
        hist = []
        for ep in range(ft["max_epochs"]):
            model.train()
            perm = torch.randperm(len(itr), device=device)
            for i in range(0, len(itr), ft["batch_size"]):
                b = perm[i:i + ft["batch_size"]]
                opt.zero_grad()
                out = model(Xtr_t[b], None).squeeze(-1)
                loss = loss_fn(out, ytr_t[b])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                val_p = torch.sigmoid(model(Xva_t, None).squeeze(-1)).cpu().numpy()
            v_auc = roc_auc_score(y[iva], val_p) if len(np.unique(y[iva])) > 1 else 0.5
            hist.append(float(v_auc))
            if v_auc > best_auc:
                best_auc, best_state, wait = v_auc, \
                    {k2: v.clone() for k2, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= ft["patience"]:
                    break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            p[ev] = torch.sigmoid(model(Xev_t, None).squeeze(-1)).cpu().numpy()
        if fold_i == 0:
            hist0 = hist
    return p, hist0


# ═══════════════════════════════════════════════════════════════════════════
# 10. ST-COUPLENET
# ═══════════════════════════════════════════════════════════════════════════

def _import_torch():
    import torch
    import torch.nn as nn
    return torch, nn


class _STCoupleNet:
    """Static-temporal coupling network (Eqs. 16-19 of the paper)."""

    @staticmethod
    def build(n_static: int, cfg: Dict[str, Any],
              use_static: bool = True, use_temporal: bool = True,
              use_attention: bool = True, no_interaction: bool = False,
              temporal_channels: Tuple[int, ...] = (0, 1, 2, 3, 4)):
        torch, nn = _import_torch()
        scfg = cfg["models"]["st_couplenet"]
        p_drop = scfg["dropout"]
        attn_d = scfg["attn_dim"]

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.use_static = use_static
                self.use_temporal = use_temporal
                self.temporal_channels = list(temporal_channels)
                self.no_interaction = no_interaction

                self.static = nn.Sequential(
                    nn.Linear(n_static, scfg["static_hidden"]), nn.GELU(),
                    nn.Dropout(p_drop),
                    nn.Linear(scfg["static_hidden"], scfg["static_out"]), nn.GELU(),
                )
                self.t_proj = nn.Linear(len(temporal_channels), attn_d)
                if use_attention:
                    self.Wq = nn.Linear(attn_d, attn_d)
                    self.Wk = nn.Linear(attn_d, attn_d)
                    self.Wv = nn.Linear(attn_d, attn_d)
                self.t_out = nn.Linear(attn_d, scfg["temporal_out"])
                self.t_drop = nn.Dropout(p_drop)
                self.w1 = nn.Parameter(torch.ones(scfg["temporal_out"]) * 0.5)
                self.w2 = nn.Parameter(torch.ones(scfg["temporal_out"]) * 0.5)
                if no_interaction:
                    self.register_buffer("w3", torch.zeros(scfg["temporal_out"]))
                else:
                    self.w3 = nn.Parameter(torch.ones(scfg["temporal_out"]) * 0.1)
                self.head = nn.Linear(scfg["temporal_out"], 1)

            def forward(self, xs, seq):
                if self.use_static:
                    S = self.static(xs)
                else:
                    S = torch.zeros(xs.shape[0], scfg["static_out"], device=xs.device)
                if self.use_temporal:
                    h = self.t_proj(seq[:, :, self.temporal_channels])
                    if use_attention:
                        q, k, v = self.Wq(h), self.Wk(h), self.Wv(h)
                        a = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(attn_d), -1)
                        h = a @ v
                    h = self.t_drop(h).mean(dim=1)
                    T = self.t_out(h)
                else:
                    T = torch.zeros(xs.shape[0], scfg["temporal_out"], device=xs.device)
                w3 = self.w3 if self.no_interaction else torch.relu(self.w3)
                z = self.w1 * S + self.w2 * T + w3 * (S * T)
                return self.head(z).squeeze(-1), S

        return Net()


def _standardize_train_only(train: np.ndarray, *arrs: np.ndarray,
                             skip_col: Optional[int] = None) -> List[np.ndarray]:
    """Fit mean/std on train (n, F) or (n, T, F); leave `skip_col` unscaled if given."""
    ax = tuple(range(train.ndim - 1))
    if skip_col is None:
        mu = np.nanmean(train, axis=ax, keepdims=True)
        sd = np.nanstd(train, axis=ax, keepdims=True) + 1e-6
        return [np.nan_to_num((a - mu) / sd) for a in (train,) + arrs]
    mask = np.ones(train.shape[-1], bool); mask[skip_col] = False
    mu = np.nanmean(train[..., mask], axis=ax, keepdims=True)
    sd = np.nanstd(train[..., mask], axis=ax, keepdims=True) + 1e-6
    out = []
    for a in (train,) + arrs:
        a = a.copy()
        a[..., mask] = (a[..., mask] - mu) / sd
        out.append(np.nan_to_num(a))
    return out


def _fit_stcouplenet(Xterr_tr, Xseq_tr, ytr, Xterr_va, Xseq_va, yva,
                     cfg, seed, lam, ablation_kwargs):
    torch, nn = _import_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed); np.random.seed(seed)

    scfg = cfg["models"]["st_couplenet"]
    model = _STCoupleNet.build(Xterr_tr.shape[1], cfg, **ablation_kwargs).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=scfg["lr"],
                            weight_decay=scfg["weight_decay"])
    bce = nn.BCEWithLogitsLoss()

    Str = torch.tensor(Xterr_tr, device=device)
    Qtr = torch.tensor(Xseq_tr, device=device)
    yt  = torch.tensor(ytr.astype(np.float32), device=device)
    Sva = torch.tensor(Xterr_va, device=device)
    Qva = torch.tensor(Xseq_va, device=device)

    best_auc, best_state, wait = -1, None, 0
    for ep in range(scfg["max_epochs"]):
        model.train()
        perm = torch.randperm(len(Str), device=device)
        for i in range(0, len(Str), scfg["batch_size"]):
            b = perm[i:i + scfg["batch_size"]]
            qb = Qtr[b].clone()
            if lam > 0: qb.requires_grad_(True)
            logit, S = model(Str[b], qb)
            loss = bce(logit, yt[b])
            if lam > 0:
                g = torch.autograd.grad(torch.sigmoid(logit).sum(), qb,
                                        create_graph=True, allow_unused=True)[0]
                if g is not None:
                    nrm = S.norm(dim=-1)
                    thresh = torch.quantile(nrm.detach(), 0.5)
                    hi = (nrm >= thresh).float()
                    viol = torch.relu(-g[:, :, 4]).sum(dim=1)  # rainfall is channel 4
                    loss = loss + lam * (viol * hi).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = torch.sigmoid(model(Sva, Qva)[0]).cpu().numpy()
        av = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else 0.5
        if av > best_auc:
            best_auc, best_state, wait = av, \
                {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= scfg["patience"]: break
    model.load_state_dict(best_state); model.eval()
    return model, best_auc


def run_stcouplenet_fold(d, tr, va, ev, cfg, seed,
                         ablation_kwargs: Optional[Dict] = None,
                         lam_grid: Optional[List[float]] = None,
                         record_curve: bool = False):
    """Fit + predict one fold. Returns (preds_on_eval, history_or_None)."""
    torch, _ = _import_torch()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scfg = cfg["models"]["st_couplenet"]
    lam_grid = lam_grid if lam_grid is not None else scfg["lambda_grid"]
    ablation_kwargs = ablation_kwargs or {}

    terrain = d["terrain"].astype(np.float32)
    seq = d["seq"].astype(np.float32)

    # standardize terrain (leave lithology col 12 unscaled) and seq, fit on train
    s_tr, s_va, s_ev = _standardize_train_only(
        terrain[tr], terrain[va], terrain[ev],
        skip_col=cfg["features"]["lithology_terrain_col"])
    q_tr, q_va, q_ev = _standardize_train_only(seq[tr], seq[va], seq[ev])

    best_lam, best_auc, best_preds, best_hist = None, -1, None, None
    for lam in lam_grid:
        model, av = _fit_stcouplenet(s_tr, q_tr, d["y"][tr],
                                     s_va, q_va, d["y"][va],
                                     cfg, seed, lam, ablation_kwargs)
        model.eval()
        with torch.no_grad():
            p_ev = torch.sigmoid(model(
                torch.tensor(s_ev, device=device),
                torch.tensor(q_ev, device=device))[0]).cpu().numpy()
        if av > best_auc:
            best_auc, best_lam, best_preds = av, lam, p_ev
    return best_preds, best_lam


def run_stcouplenet_protocol(d, protocol, clusters, cfg,
                             ablation_kwargs=None, seeds=None):
    """5-seed mean predictions for the given protocol."""
    seeds = seeds or cfg["models"]["st_couplenet"]["seeds"]
    n = len(d["y"])
    folds = get_folds(protocol, d, clusters, cfg)
    # P3 needs a val partition for early stopping; CV folds use an inner split
    if protocol == "P3":
        val_idx = p3_validation_idx(d)

    p_by_seed = []
    lam_by_seed = []
    for seed in seeds:
        p = np.full(n, np.nan, dtype=np.float32)
        for fi, (tr, ev) in enumerate(folds):
            if protocol == "P3":
                va = val_idx
            else:
                rng = np.random.default_rng(seed + fi)
                idx = rng.permutation(len(tr))
                k = int((1 - cfg["evaluation"]["inner_early_stop_frac"]) * len(idx))
                itr, iva = tr[idx[:k]], tr[idx[k:]]
                tr, va = itr, iva
            preds, _ = run_stcouplenet_fold(d, tr, va, ev, cfg, seed,
                                            ablation_kwargs=ablation_kwargs)
            p[ev] = preds
        p_by_seed.append(p)
        lam_by_seed.append(_)

    return p_by_seed


# ═══════════════════════════════════════════════════════════════════════════
# 11. EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

def fast_auc(y: np.ndarray, p: np.ndarray) -> float:
    r = rankdata(p); n1 = int(y.sum()); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def cm_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    yh = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return dict(
        Accuracy=accuracy_score(y, yh),
        Precision=precision_score(y, yh, zero_division=0),
        Recall=recall_score(y, yh, zero_division=0),
        Specificity=tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        F1=f1_score(y, yh, zero_division=0),
        MCC=matthews_corrcoef(y, yh),
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
    )


def _midrank(x: np.ndarray) -> np.ndarray:
    J = np.argsort(x); Z = x[J]; n = len(x); T = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j + 1); i = j
    T2 = np.empty(n); T2[J] = T; return T2


def _delong_components(y, p):
    pos, neg = p[y == 1], p[y == 0]
    m, n = len(pos), len(neg)
    tx, ty = _midrank(pos), _midrank(neg)
    tz = _midrank(np.concatenate([pos, neg]))
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return v01.mean(), v01, v10


def delong_test(y, p1, p2):
    a1, v01_1, v10_1 = _delong_components(y, p1)
    a2, v01_2, v10_2 = _delong_components(y, p2)
    m, n = len(v01_1), len(v10_1)
    S = (np.cov(np.vstack([v01_1, v01_2]), ddof=1) / m +
         np.cov(np.vstack([v10_1, v10_2]), ddof=1) / n)
    d = a1 - a2
    var_d = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = d / math.sqrt(var_d) if var_d > 0 else 0.0
    p = 2 * (1 - norm.cdf(abs(z)))
    return dict(auc1=a1, auc2=a2, diff=d, z=z, p=p)


def bootstrap_ci(y, p, groups, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a1, a2 = [], []
    n = len(y)
    for _ in range(n_boot):
        b = rng.integers(0, n, n)
        if y[b].min() == y[b].max(): continue
        a1.append(fast_auc(y[b], p[b]))
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        b = np.concatenate([idx[g] for g in pick])
        if y[b].min() == y[b].max(): continue
        a2.append(fast_auc(y[b], p[b]))
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(a1), q(a2)


# ═══════════════════════════════════════════════════════════════════════════
# 12. SPATIAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════

def terrain_variogram(lat, lon, terrain, n_pairs, bins_km):
    rng = np.random.default_rng(0)
    n = len(lat); i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    ok = i != j; i, j = i[ok], j[ok]
    # haversine (km)
    lat1, lon1, lat2, lon2 = map(np.radians, (lat[i], lon[i], lat[j], lon[j]))
    a = np.sin((lat2 - lat1) / 2) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    d = 6371 * 2 * np.arcsin(np.sqrt(a))
    terr = (terrain - np.nanmean(terrain, 0)) / (np.nanstd(terrain, 0) + 1e-9)
    sq = 0.5 * np.nanmean((terr[i] - terr[j]) ** 2, axis=1)
    rows = []
    bins_km = np.asarray(bins_km, float)
    for b in range(len(bins_km) - 1):
        m = (d >= bins_km[b]) & (d < bins_km[b + 1])
        if m.sum() >= 30:
            rows.append(dict(bin_lo=bins_km[b], bin_hi=bins_km[b + 1],
                             n=int(m.sum()), gamma=float(sq[m].mean())))
    return pd.DataFrame(rows)


def kish_effective_n(lat, lon, radius_km=2.0):
    lat_km = lat * 111.0
    lon_km = lon * 111.0 * np.cos(np.radians(lat.mean()))
    d2 = (lon_km[:, None] - lon_km[None, :]) ** 2 + (lat_km[:, None] - lat_km[None, :]) ** 2
    w = 1.0 / np.maximum((d2 <= radius_km ** 2).sum(1), 1)
    return float((w.sum() ** 2) / (w ** 2).sum())


# ═══════════════════════════════════════════════════════════════════════════
# 13. FIGURES
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    "CatBoost": "#d95f02", "GBT": "#1b9e77", "RF": "#7570b3", "LR": "#e7298a",
    "MLP": "#66a61e", "TabNet": "#e6ab02", "FT-Transformer": "#a6761d",
    "ST-CoupleNet": "#525252",
}
ORDER = ["CatBoost", "GBT", "RF", "ST-CoupleNet", "LR",
         "FT-Transformer", "TabNet", "MLP"]
PROTO_LABEL = {"P1": "P1 Random 5-fold", "P2": "P2 Spatial CV (0.1°)",
               "P3": "P3 Temporal point-split", "P4": "P4 Cluster-disjoint"}


def plot_roc(preds: Dict[str, np.ndarray], y: np.ndarray, outdir: Path):
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, (P, ttl) in zip(axes, PROTO_LABEL.items()):
        for mn in ORDER:
            key = f"{mn}|{P}"
            if key not in preds: continue
            p = preds[key]; m = ~np.isnan(p)
            if m.sum() == 0: continue
            fpr, tpr, _ = roc_curve(y[m], p[m])
            auc = fast_auc(y[m], p[m])
            ax.plot(fpr, tpr, lw=1.1, color=COLORS.get(mn, "k"),
                    label=f"{mn} {auc:.3f}")
        ax.plot([0, 1], [0, 1], ":", color="grey", lw=0.6)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(ttl, fontsize=8)
        ax.set_xlabel("False positive rate", fontsize=7)
        ax.legend(fontsize=5.5, loc="lower right", frameon=False)
    axes[0].set_ylabel("True positive rate", fontsize=7)
    plt.tight_layout()
    fig.savefig(outdir / "fig06_roc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(auc_df: pd.DataFrame, outdir: Path):
    piv = auc_df.pivot(index="model", columns="protocol", values="AUC")
    piv = piv.reindex(index=ORDER, columns=["P1", "P2", "P3", "P4"])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sns.heatmap(piv, annot=True, fmt=".3f", cmap="RdYlGn",
                vmin=0.55, vmax=1.0, cbar_kws=dict(label="AUC"), ax=ax)
    ax.set_xlabel("Validation protocol"); ax.set_ylabel("")
    plt.tight_layout()
    fig.savefig(outdir / "fig07_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_feature_set_bars(fs_df: pd.DataFrame, outdir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, P in zip(axes, ["P3", "P4"]):
        sub = fs_df[fs_df.protocol == P]
        piv = sub.pivot(index="model", columns="featureset", values="AUC")
        cols = [c for c in ["B", "C"] if c in piv.columns]
        piv = piv[cols].reindex([m for m in ORDER if m in piv.index])
        piv.plot(kind="bar", ax=ax, color=["#7fb47f", "#3b76b5"][:len(cols)])
        ax.set_ylabel("AUC"); ax.set_ylim(0.5, 0.8)
        ax.set_title(f"{PROTO_LABEL[P]}: terrain vs terrain+temporal")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(outdir / "fig08_featureset_bars.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_variogram(vg: pd.DataFrame, neff: float, n: int, outdir: Path):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x = (vg.bin_lo + vg.bin_hi) / 2
    ax.plot(x, vg.gamma, "o-", color="#c0392b")
    ax.axhline(vg.gamma.iloc[-1], ls="--", color="grey", lw=0.8,
               label=f"overall semivariance = {vg.gamma.iloc[-1]:.3f}")
    ax.set_xlabel("Separation distance (km)")
    ax.set_ylabel("Semivariance γ(h)")
    ax.set_title(f"Terrain variogram — Kish n_eff = {neff:.0f} of {n}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig10_variogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_pointsplit(preds: Dict[str, np.ndarray], y: np.ndarray,
                              outdir: Path):
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5))
    rows = []
    for ax, mn in zip(axes.flat, ORDER):
        key = f"{mn}|P3"
        if key not in preds:
            ax.axis("off"); continue
        p = preds[key]; m = ~np.isnan(p)
        if m.sum() == 0:
            ax.axis("off"); continue
        cm = confusion_matrix(y[m], (p[m] >= 0.5).astype(int), labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        mcc = matthews_corrcoef(y[m], (p[m] >= 0.5).astype(int))
        auc = fast_auc(y[m], p[m])
        ax.set_title(f"{mn}\nAUC {auc:.3f}  MCC {mcc:.3f}", fontsize=8)
        ax.set_xticklabels(["NLS", "LS"]); ax.set_yticklabels(["NLS", "LS"])
        rows.append(dict(model=mn, AUC=round(auc, 4), MCC=round(mcc, 4),
                         **{k: v for k, v in cm_metrics(y[m], p[m]).items()}))
    plt.tight_layout()
    fig.savefig(outdir / "fig11_confusion.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(rows).to_csv(outdir / "confusion_metrics_pointsplit.csv",
                              index=False)


def plot_training_curves(curves: Dict[str, List[float]], outdir: Path):
    if not curves: return
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, hist in curves.items():
        if hist is None: continue
        ax.plot(np.arange(1, len(hist) + 1), hist,
                lw=1.0, color=COLORS.get(name, "k"), label=name)
        imax = int(np.argmax(hist))
        ax.plot(imax + 1, hist[imax], "o", color=COLORS.get(name, "k"), ms=4)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation AUC")
    ax.set_title("Deep-model training curves (P3, seed 42)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(outdir / "fig_training_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# 14. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    cfg = load_config("config.yaml")
    set_seed(cfg["validation"]["seed"])
    outdir = Path(cfg["output"]["dir"]); outdir.mkdir(parents=True, exist_ok=True)
    figdir = outdir / cfg["output"]["figures_subdir"]; figdir.mkdir(parents=True, exist_ok=True)

    # ── 1. load ─────────────────────────────────────────────────
    log.info("Step 1/8  Loading data …")
    d = load_data(cfg)
    y = d["y"].astype(int); n = len(y)

    # ── 2. clusters (Algorithm 4) ───────────────────────────────
    log.info("Step 2/8  Connected-component leakage clusters …")
    clusters = build_clusters(d, cfg["validation"]["cell_size_deg"])
    log.info(f"  → {len(np.unique(clusters))} clusters, "
             f"largest = {np.bincount(clusters).max()} samples")

    # sanity: verify P4 fold disjointness
    folds_p4 = get_folds("P4", d, clusters, cfg)
    for a, b in folds_p4:
        for f in ("point", "source_id"):
            assert not set(d[f][a]) & set(d[f][b]), f"P4 leak on {f}"
    log.info("  P4 fold disjointness verified (point + source_id)")

    # ── 3. main grid: 8 models × 4 protocols, feature set C ─────
    log.info("Step 3/8  Main grid: 8 models × 4 protocols (feature set C) …")
    X_C, lith_C = build_feature_set(d, "C", cfg)
    models_sk = ["LR", "RF", "GBT", "MLP"]
    models_deep = ["CatBoost", "TabNet", "FT-Transformer", "ST-CoupleNet"]
    all_models = models_sk + models_deep

    preds: Dict[str, np.ndarray] = {}          # key = "model|P"
    rows = []
    curves: Dict[str, List[float]] = {}
    y_val = p3_validation_idx(d)

    for mn in all_models:
        for P in ["P1", "P2", "P3", "P4"]:
            log.info(f"  · {mn:15s} {P}")
            t0 = time.time()
            folds = get_folds(P, d, clusters, cfg)

            if mn in models_sk:
                p = run_sklearn_protocol(mn, X_C, y, folds, lith_C, cfg)
            elif mn == "CatBoost":
                p = run_catboost_protocol(X_C, y, folds, lith_C, cfg)
            elif mn == "TabNet":
                p, hist = run_tabnet_protocol(X_C, y, folds, y_val, lith_C, cfg)
                if P == "P3" and hist is not None: curves["TabNet"] = hist
            elif mn == "FT-Transformer":
                p, hist = run_ft_protocol(X_C, y, folds, lith_C, cfg)
                if P == "P3" and hist is not None: curves["FT-Transformer"] = hist
            elif mn == "ST-CoupleNet":
                # average over seeds
                by_seed = run_stcouplenet_protocol(d, P, clusters, cfg)
                p = np.nanmean(np.stack(by_seed, axis=0), axis=0)

            preds[f"{mn}|{P}"] = p
            m = ~np.isnan(p)
            auc = fast_auc(y[m], p[m])
            extra = cm_metrics(y[m], p[m]) if P == "P3" else {}
            rows.append(dict(model=mn, protocol=P, AUC=round(auc, 4),
                             n_eval=int(m.sum()), time_s=round(time.time() - t0, 1),
                             **{k: round(v, 4) if isinstance(v, float) else v
                                for k, v in extra.items()}))
            log.info(f"    AUC = {auc:.4f}  ({time.time() - t0:.0f}s)")

    grid = pd.DataFrame(rows)
    grid.to_csv(outdir / "table05_main_grid.csv", index=False)
    log.info(f"  → {outdir/'table05_main_grid.csv'}")

    # ── 4. feature-set comparison A/B/C (and E/F if triggers exist) ─
    log.info("Step 4/8  Feature-set comparison (A/B/C + optional E/F) …")
    available_fs = ["A", "B", "C"]
    if "trigger" in d:
        available_fs += ["E", "F"]
    fs_rows = []
    for fs in available_fs:
        X, lith = build_feature_set(d, fs, cfg)
        for P in ["P3", "P4"]:
            folds = get_folds(P, d, clusters, cfg)
            for mn in all_models:
                try:
                    if mn in models_sk:
                        p = run_sklearn_protocol(mn, X, y, folds, lith, cfg)
                    elif mn == "CatBoost":
                        p = run_catboost_protocol(X, y, folds, lith, cfg)
                    elif mn == "TabNet":
                        p, _ = run_tabnet_protocol(X, y, folds, y_val, lith, cfg)
                    elif mn == "FT-Transformer":
                        p, _ = run_ft_protocol(X, y, folds, lith, cfg)
                    elif mn == "ST-CoupleNet":
                        by_seed = run_stcouplenet_protocol(d, P, clusters, cfg)
                        p = np.nanmean(np.stack(by_seed, axis=0), axis=0)
                    else:
                        continue
                    m = ~np.isnan(p)
                    fs_rows.append(dict(featureset=fs, protocol=P, model=mn,
                                        n_cols=X.shape[1],
                                        AUC=round(fast_auc(y[m], p[m]), 4)))
                except Exception as e:
                    log.warning(f"    {fs} {P} {mn}: {e}")
    pd.DataFrame(fs_rows).to_csv(outdir / "table06_featuresets.csv", index=False)

    # ── 5. ST-CoupleNet structured ablation (P3 only) ───────────
    log.info("Step 5/8  ST-CoupleNet structured ablation (P3) …")
    ablations = {
        "full": {},
        "no_attention": {"use_attention": False},
        "no_interaction": {"no_interaction": True},
        "no_static": {"use_static": False},
        "no_temporal": {"use_temporal": False},
        "rain_only_temporal": {"temporal_channels": (4,)},
        "ndvi_removed": {"temporal_channels": (1, 2, 3, 4)},
    }
    abl_rows = []
    for ab, kwargs in ablations.items():
        by_seed = run_stcouplenet_protocol(d, "P3", clusters, cfg,
                                            ablation_kwargs=kwargs)
        aucs = []
        for p in by_seed:
            m = ~np.isnan(p)
            aucs.append(fast_auc(y[m], p[m]))
        abl_rows.append(dict(ablation=ab,
                             AUC_mean=round(float(np.mean(aucs)), 4),
                             AUC_sd=round(float(np.std(aucs, ddof=1)), 4)))
        log.info(f"  {ab:20s}  {abl_rows[-1]['AUC_mean']:.4f} ± "
                 f"{abl_rows[-1]['AUC_sd']:.4f}")
    pd.DataFrame(abl_rows).to_csv(outdir / "table07_ablation.csv", index=False)

    # ── 6. spatial diagnostics ──────────────────────────────────
    log.info("Step 6/8  Spatial diagnostics (variogram + Kish n_eff) …")
    terr_num = d["terrain"][:, :12].astype(float)
    vg = terrain_variogram(d["lat"], d["lon"], terr_num,
                           cfg["diagnostics"]["variogram_pairs"],
                           cfg["diagnostics"]["variogram_bins_km"])
    vg.to_csv(outdir / "table08_variogram.csv", index=False)
    neff = kish_effective_n(d["lat"], d["lon"],
                            cfg["diagnostics"]["decluster_radius_km"])
    pd.DataFrame([dict(quantity="Kish effective n",
                       value=round(neff, 1),
                       fraction=round(neff / n, 4),
                       n_nominal=n)]).to_csv(outdir / "table08_kish.csv", index=False)
    log.info(f"  Kish n_eff = {neff:.0f} / {n}  (fraction {neff/n:.3f})")

    # ── 7. DeLong + bootstrap CIs on P3 ─────────────────────────
    log.info("Step 7/8  DeLong + bootstrap CIs (P3) …")
    y_p3 = y.copy()
    p_gbt = preds["GBT|P3"]
    dl_rows = []
    for mn in all_models:
        if mn == "GBT": continue
        p = preds.get(f"{mn}|P3")
        if p is None: continue
        m = ~np.isnan(p) & ~np.isnan(p_gbt)
        if m.sum() < 10: continue
        res = delong_test(y[m], p_gbt[m], p[m])
        dl_rows.append(dict(model=mn, vs="GBT", **{k: round(v, 4)
                                                    for k, v in res.items()}))
    pd.DataFrame(dl_rows).to_csv(outdir / "table_delong.csv", index=False)

    boot_rows = []
    for mn in all_models:
        p = preds.get(f"{mn}|P3")
        if p is None: continue
        m = ~np.isnan(p)
        if m.sum() < 10: continue
        ci_s, ci_c = bootstrap_ci(y[m], p[m], d["group"][m],
                                   cfg["evaluation"]["bootstrap_resamples"])
        boot_rows.append(dict(model=mn, AUC=round(fast_auc(y[m], p[m]), 4),
                              CI_lo_sample=round(ci_s[0], 4),
                              CI_hi_sample=round(ci_s[1], 4),
                              CI_lo_cell=round(ci_c[0], 4),
                              CI_hi_cell=round(ci_c[1], 4)))
    pd.DataFrame(boot_rows).to_csv(outdir / "table_bootstrap.csv", index=False)

    log.info("Step 8/8  Generating figures …")
    plot_roc(preds, y, figdir)
    plot_heatmap(grid, figdir)
    plot_feature_set_bars(pd.DataFrame(fs_rows), figdir)
    plot_variogram(vg, neff, n, figdir)
    plot_confusion_pointsplit(preds, y, figdir)
    plot_training_curves(curves, figdir)
    log.info(f"  figures → {figdir.resolve()}")

    log.info("─" * 72)
    log.info(f"DONE — total time {time.time() - t_start:.0f}s")
    log.info(f"outputs: {outdir.resolve()}")
    print("\nMain grid (AUC by model × protocol):")
    piv = grid.pivot(index="model", columns="protocol", values="AUC").round(3)
    piv = piv.reindex(index=ORDER, columns=["P1", "P2", "P3", "P4"])
    print(piv.to_string())


if __name__ == "__main__":
    main()
