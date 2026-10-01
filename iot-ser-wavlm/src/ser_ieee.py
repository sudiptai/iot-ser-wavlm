"""
ser_ieee.py - Speaker-independent SER on RAVDESS with self-supervised speech backbones,
Optuna hyper-parameter optimisation, leave-speakers-out cross-validation and
IEEE-style figures/tables.

Pipeline
  1. Data      : RAVDESS speech (1440 clips, 24 actors, 8 emotions). Speaker-independent
                 splits only; augmentation on the waveform (gain, noise, speed, time-mask).
  2. Model     : SSL backbone (wav2vec2 / WavLM / HuBERT) -> learnable weighted sum of all
                 transformer layers -> attentive (or mean) pooling -> dropout -> linear.
  3. Optuna    : TPE + median pruning on an inner speaker-independent split, maximising
                 validation UAR (unweighted average recall = macro recall).
  4. Final     : k-fold leave-speakers-out CV with the best configuration; report mean +- std
                 of Accuracy, UAR, macro-F1; pooled confusion matrix and ROC.
  5. Figures   : IEEE single-column (3.5 in) / double-column (7.16 in), Times, 8 pt,
                 vector PDF + 600 dpi PNG, colour-blind-safe categorical palette.
"""
from __future__ import annotations
import glob, json, math, os, random, re, time, zipfile, urllib.request, copy
from dataclasses import dataclass, asdict, field
import numpy as np

# --------------------------------------------------------------------------- constants
RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
SR = 16000
MAX_SEC = 4.0
NAME_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)")

# colour-blind-safe categorical palette (validated: adjacent-pair CVD dE >= 9.8)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#B03A8E", "#E69F00", "#3F6FD8", "#7B3294", "#C0392B"]
SEQ_CMAP = "Blues"


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed)
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- data
def download(root: str):
    os.makedirs(root, exist_ok=True)
    z = os.path.join(root, "Audio_Speech_Actors_01-24.zip")
    if not os.path.exists(z):
        print("Downloading RAVDESS ..."); urllib.request.urlretrieve(RAVDESS_URL, z)
    if not glob.glob(os.path.join(root, "Actor_*")):
        print("Extracting ..."); zipfile.ZipFile(z).extractall(root)


def index(root: str):
    """Return list of dicts {path,label,actor,intensity,statement} - audio-only speech, de-duplicated."""
    items, seen = [], set()
    for f in sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True)):
        m = NAME_RE.match(os.path.basename(f))
        if not m: continue
        p = [int(x) for x in m.groups()]
        if p[0] != 3 or p[1] != 1 or m.group(0) in seen: continue
        seen.add(m.group(0))
        items.append({"path": f, "label": p[2] - 1, "intensity": p[3], "statement": p[4], "actor": p[6]})
    print(f"{len(items)} clips, actors {sorted({i['actor'] for i in items})}")
    return items


def load_wav(path: str) -> np.ndarray:
    import torch, torchaudio
    w, sr = torchaudio.load(path)
    w = w.mean(0)
    if sr != SR: w = torchaudio.functional.resample(w, sr, SR)
    thr = 0.01 * w.abs().max()
    nz = torch.nonzero(w.abs() > thr)
    if len(nz): w = w[nz[0, 0]: nz[-1, 0] + 1]
    w = w[: int(MAX_SEC * SR)]
    return (w / (w.abs().max() + 1e-6)).numpy().astype(np.float32)


class WavCache:
    """Load every clip once into RAM (1440 x <=4 s x 16 kHz float32 ~ 370 MB)."""
    def __init__(self, items):
        self.w = {}
        for i, it in enumerate(items):
            self.w[it["path"]] = load_wav(it["path"])
            if i % 200 == 0: print(f"  cached {i}/{len(items)}")
    def __getitem__(self, p): return self.w[p]


def augment(w: np.ndarray, cfg: "Cfg") -> np.ndarray:
    import torch, torchaudio
    if cfg.aug_speed > 0 and random.random() < 0.5:
        # quantised factors keep gcd(SR, SR*f) large -> small resampling kernel (arbitrary rates are ~100x slower)
        steps = [f for f in (0.85, 0.9, 0.95, 1.05, 1.1, 1.15) if abs(f - 1) <= cfg.aug_speed + 1e-9]
        if steps:
            f = random.choice(steps)
            w = torchaudio.functional.resample(torch.tensor(w), SR, int(round(SR * f / 100)) * 100).numpy()
    w = w * random.uniform(0.6, 1.4)
    if cfg.aug_noise > 0:
        w = w + np.random.normal(0, cfg.aug_noise * random.random(), w.shape).astype(np.float32)
    if cfg.aug_timemask > 0:
        n = len(w); L = int(n * cfg.aug_timemask * random.random())
        if L > 0:
            s = random.randint(0, n - L); w = w.copy(); w[s:s + L] = 0
    return w[: int(MAX_SEC * SR)].astype(np.float32)


class DS:
    def __init__(self, items, cache, cfg, train): self.items, self.c, self.cfg, self.train = items, cache, cfg, train
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]; w = self.c[it["path"]]
        if self.train: w = augment(w, self.cfg)
        return w, it["label"]


def collate(b):
    import torch
    L = max(len(w) for w, _ in b)
    x = np.zeros((len(b), L), np.float32); m = np.zeros((len(b), L), np.int64)
    for i, (w, _) in enumerate(b): x[i, :len(w)] = w; m[i, :len(w)] = 1
    return torch.tensor(x), torch.tensor(m), torch.tensor([y for _, y in b])


def speaker_folds(items, k=5, seed=0):
    """Leave-speakers-out folds; actors shuffled with a fixed seed, gender-balanced by parity."""
    actors = sorted({i["actor"] for i in items})
    rng = random.Random(seed)
    male, female = [a for a in actors if a % 2 == 1], [a for a in actors if a % 2 == 0]
    rng.shuffle(male); rng.shuffle(female)
    folds = [[] for _ in range(k)]
    for j, a in enumerate(male + female): folds[j % k].append(a)
    return [sorted(f) for f in folds]


# --------------------------------------------------------------------------- config
@dataclass
class Cfg:
    backbone: str = "microsoft/wavlm-base-plus"
    pooling: str = "attn"            # attn | mean
    layer_weighting: bool = True     # learnable softmax weights over hidden layers
    n_freeze: int = 0                # transformer layers frozen from the bottom (CNN always frozen)
    dropout: float = 0.2
    lr: float = 3e-5
    head_lr_mult: float = 10.0
    warmup: float = 0.1
    weight_decay: float = 0.01
    label_smoothing: float = 0.05
    mixup: float = 0.0
    batch_size: int = 8
    grad_accum: int = 2
    epochs: int = 12
    aug_noise: float = 0.005
    aug_speed: float = 0.1
    aug_timemask: float = 0.1
    seed: int = 42


# --------------------------------------------------------------------------- model
def build_model(cfg: Cfg, n_classes=8):
    import torch, torch.nn as nn
    from transformers import AutoModel

    class SER(nn.Module):
        def __init__(self):
            super().__init__()
            self.bb = AutoModel.from_pretrained(cfg.backbone, layerdrop=0.0)
            self.bb.config.output_hidden_states = True
            self.bb.config.layerdrop = 0.0            # keep the hidden-state count fixed in train mode
            if hasattr(self.bb, "freeze_feature_encoder"): self.bb.freeze_feature_encoder()
            layers = self.bb.encoder.layers
            for l in layers[:cfg.n_freeze]:
                for p in l.parameters(): p.requires_grad = False
            H = self.bb.config.hidden_size
            nL = self.bb.config.num_hidden_layers + 1
            self.lw = nn.Parameter(torch.zeros(nL)) if cfg.layer_weighting else None
            self.attn = nn.Sequential(nn.Linear(H, 128), nn.Tanh(), nn.Linear(128, 1)) if cfg.pooling == "attn" else None
            self.drop = nn.Dropout(cfg.dropout)
            self.fc = nn.Linear(H, n_classes)

        def embed(self, x, m):
            out = self.bb(x, attention_mask=m, output_hidden_states=True)
            if self.lw is not None:
                hs = torch.stack(out.hidden_states, 0)                      # L,B,T,H
                lw = self.lw[-hs.shape[0]:] if hs.shape[0] <= len(self.lw) else \
                     torch.cat([self.lw, self.lw.new_zeros(hs.shape[0] - len(self.lw))])
                h = (torch.softmax(lw, 0)[:, None, None, None] * hs).sum(0)
            else:
                h = out.last_hidden_state
            # frame-level mask
            fm = self.bb._get_feature_vector_attention_mask(h.shape[1], m).float()  # B,T
            if self.attn is not None:
                a = self.attn(h).squeeze(-1).masked_fill(fm == 0, -1e4)
                a = torch.softmax(a, -1)
                return (a[..., None] * h).sum(1)
            return (h * fm[..., None]).sum(1) / fm.sum(1, keepdim=True).clamp(min=1)

        def forward(self, x, m):
            return self.fc(self.drop(self.embed(x, m)))

    return SER()


# --------------------------------------------------------------------------- train / eval
def metrics(y, yhat):
    from sklearn.metrics import accuracy_score, f1_score, recall_score
    return {"acc": accuracy_score(y, yhat), "uar": recall_score(y, yhat, average="macro"),
            "f1": f1_score(y, yhat, average="macro")}


@dataclass
class RunResult:
    cfg: dict
    history: list = field(default_factory=list)    # per epoch dict
    best: dict = field(default_factory=dict)
    y: list = field(default_factory=list)
    yhat: list = field(default_factory=list)
    probs: list = field(default_factory=list)
    emb: list = field(default_factory=list)
    infer_ms: float = 0.0


def evaluate(model, loader, dev, want_emb=False):
    import torch
    model.eval(); ys, ps, es = [], [], []
    t0 = time.time(); n = 0
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for x, m, y in loader:
            x, m = x.to(dev), m.to(dev)
            e = model.embed(x, m); logits = model.fc(e)
            ps.append(torch.softmax(logits.float(), -1).cpu().numpy()); ys.append(y.numpy())
            if want_emb: es.append(e.float().cpu().numpy())
            n += len(y)
    if dev == "cuda": torch.cuda.synchronize()
    ms = (time.time() - t0) * 1000 / max(n, 1)
    P = np.concatenate(ps); Y = np.concatenate(ys)
    return Y, P, (np.concatenate(es) if es else None), ms


def train_run(cfg: Cfg, train_items, val_items, cache, dev="cuda", trial=None, test_items=None,
              verbose=True) -> RunResult:
    import torch, torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup
    seed_all(cfg.seed)
    model = build_model(cfg).to(dev)
    tl = DataLoader(DS(train_items, cache, cfg, True), cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=2)
    vl = DataLoader(DS(val_items, cache, cfg, False), 16, collate_fn=collate)
    head = [p for n, p in model.named_parameters() if not n.startswith("bb.") and p.requires_grad]
    body = [p for n, p in model.named_parameters() if n.startswith("bb.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": body, "lr": cfg.lr}, {"params": head, "lr": cfg.lr * cfg.head_lr_mult}],
                            weight_decay=cfg.weight_decay)
    steps = math.ceil(len(tl) / cfg.grad_accum) * cfg.epochs
    sch = get_linear_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev == "cuda")
    res = RunResult(cfg=asdict(cfg)); best_uar, best_state = -1, None
    for ep in range(cfg.epochs):
        model.train(); tot, nb = 0.0, 0
        for i, (x, m, y) in enumerate(tl):
            x, m, y = x.to(dev), m.to(dev), y.to(dev)
            with torch.autocast(dev, enabled=dev == "cuda"):
                if cfg.mixup > 0 and random.random() < 0.5:
                    lam = np.random.beta(cfg.mixup, cfg.mixup); idx = torch.randperm(len(y), device=dev)
                    x = lam * x + (1 - lam) * x[idx]; m = torch.maximum(m, m[idx])
                    logits = model(x, m)
                    loss = lam * F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing) + \
                           (1 - lam) * F.cross_entropy(logits, y[idx], label_smoothing=cfg.label_smoothing)
                else:
                    loss = F.cross_entropy(model(x, m), y, label_smoothing=cfg.label_smoothing)
                loss = loss / cfg.grad_accum
            scaler.scale(loss).backward(); tot += loss.item() * cfg.grad_accum; nb += 1
            if (i + 1) % cfg.grad_accum == 0 or i + 1 == len(tl):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step()
        Y, P, _, _ = evaluate(model, vl, dev)
        mt = metrics(Y, P.argmax(1)); mt["epoch"] = ep + 1; mt["train_loss"] = tot / nb
        mt["val_loss"] = float(-np.log(P[np.arange(len(Y)), Y] + 1e-9).mean())
        res.history.append(mt)
        if verbose: print(f"  ep {ep+1:2d}  loss {mt['train_loss']:.3f}  val acc {mt['acc']:.3f}  UAR {mt['uar']:.3f}  F1 {mt['f1']:.3f}")
        if mt["uar"] > best_uar:
            best_uar, best_state = mt["uar"], copy.deepcopy(model.state_dict())
        if trial is not None:
            trial.report(mt["uar"], ep)
            if trial.should_prune():
                import optuna; raise optuna.TrialPruned()
    model.load_state_dict(best_state)
    ev_items = test_items if test_items is not None else val_items
    el = DataLoader(DS(ev_items, cache, cfg, False), 16, collate_fn=collate)
    Y, P, E, ms = evaluate(model, el, dev, want_emb=True)
    res.best = metrics(Y, P.argmax(1)); res.best["val_uar_best"] = best_uar
    res.y, res.yhat, res.probs, res.emb, res.infer_ms = Y.tolist(), P.argmax(1).tolist(), P.tolist(), E.tolist(), ms
    res.model = model
    return res


# --------------------------------------------------------------------------- optuna
def run_optuna(items, cache, storage: str, n_trials=25, epochs=8, dev="cuda", study_name="ser_ravdess", timeout=None,
               backbones=("microsoft/wavlm-base-plus", "facebook/wav2vec2-base", "facebook/hubert-base-ls960")):
    import optuna
    folds = speaker_folds(items, 5, seed=0)
    val_actors = set(folds[0][:2]); test_actors = set(folds[0])         # keep fold-0 speakers out of HPO entirely
    inner_val = set(folds[1][:2])
    tr = [i for i in items if i["actor"] not in test_actors | inner_val]
    va = [i for i in items if i["actor"] in inner_val]
    print(f"HPO train actors: {sorted({i['actor'] for i in tr})}  val actors: {sorted(inner_val)}")

    def objective(trial):
        cfg = Cfg(
            backbone=trial.suggest_categorical("backbone", list(backbones)),
            pooling=trial.suggest_categorical("pooling", ["attn", "mean"]),
            layer_weighting=trial.suggest_categorical("layer_weighting", [True, False]),
            n_freeze=trial.suggest_int("n_freeze", 0, 8, step=2),
            dropout=trial.suggest_float("dropout", 0.05, 0.4),
            lr=trial.suggest_float("lr", 5e-6, 1e-4, log=True),
            head_lr_mult=trial.suggest_categorical("head_lr_mult", [1.0, 5.0, 10.0, 20.0]),
            warmup=trial.suggest_float("warmup", 0.0, 0.2),
            label_smoothing=trial.suggest_float("label_smoothing", 0.0, 0.15),
            mixup=trial.suggest_categorical("mixup", [0.0, 0.2, 0.4]),
            batch_size=trial.suggest_categorical("batch_size", [4, 8]),
            aug_timemask=trial.suggest_float("aug_timemask", 0.0, 0.2),
            aug_speed=trial.suggest_float("aug_speed", 0.0, 0.15),
            epochs=epochs)
        cfg.grad_accum = 16 // cfg.batch_size
        r = train_run(cfg, tr, va, cache, dev, trial=trial, verbose=False)
        trial.set_user_attr("history", r.history); trial.set_user_attr("acc", r.best["acc"])
        return r.best["val_uar_best"]

    study = optuna.create_study(direction="maximize", study_name=study_name, storage=storage, load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, multivariate=True),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3))
    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True, catch=(RuntimeError,),
                   callbacks=[lambda s, t: print(f"trial {t.number}: {t.state.name} UAR={t.value} "
                                                 f"({(t.datetime_complete - t.datetime_start).seconds//60} min)")])
    print("best:", study.best_value, study.best_params)
    return study


def cfg_from_params(p: dict, epochs=15) -> Cfg:
    cfg = Cfg(**{k: v for k, v in p.items() if k in Cfg.__dataclass_fields__})
    cfg.grad_accum = 16 // cfg.batch_size; cfg.epochs = epochs
    return cfg


# --------------------------------------------------------------------------- cross-validation
def cross_validate(cfg: Cfg, items, cache, k=5, dev="cuda", out_dir="results", save_model_fold=0):
    os.makedirs(out_dir, exist_ok=True)
    folds = speaker_folds(items, k, seed=0); results = []
    for fi, test_actors in enumerate(folds):
        rest = [a for f in folds[:fi] + folds[fi+1:] for a in f]
        val_actors = set(rest[:3])
        tr = [i for i in items if i["actor"] not in set(test_actors) | val_actors]
        va = [i for i in items if i["actor"] in val_actors]
        te = [i for i in items if i["actor"] in set(test_actors)]
        print(f"\n=== fold {fi+1}/{k}  test actors {test_actors}  val {sorted(val_actors)} ===")
        r = train_run(cfg, tr, va, cache, dev, test_items=te)
        print(f"  TEST acc {r.best['acc']:.4f}  UAR {r.best['uar']:.4f}  F1 {r.best['f1']:.4f}  ({r.infer_ms:.1f} ms/clip)")
        if fi == save_model_fold:
            import torch; torch.save(r.model.state_dict(), os.path.join(out_dir, "best_fold_model.pt"))
        del r.model; results.append(asdict(r))
        json.dump(results, open(os.path.join(out_dir, "cv_results.json"), "w"))
    return results


def summarize(results):
    keys = ["acc", "uar", "f1"]
    tab = {k: (np.mean([r["best"][k] for r in results]), np.std([r["best"][k] for r in results])) for k in keys}
    for k, (m, s) in tab.items(): print(f"{k.upper():4s} {100*m:5.2f} +- {100*s:4.2f}")
    return tab


# --------------------------------------------------------------------------- IEEE figures
def ieee_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
        "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
        "legend.frameon": False})


COL, DBL = 3.5, 7.16   # IEEE column widths in inches


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"): fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    print("saved", name)


def fig_optuna(study, out_dir):
    import matplotlib.pyplot as plt, optuna
    trials = [t for t in study.trials if t.value is not None]
    x = [t.number for t in trials]; y = [t.value for t in trials]
    best = np.maximum.accumulate(y)
    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.scatter(x, y, s=10, color=PALETTE[0], label="Trial UAR", zorder=3)
    ax.step(x, best, where="post", color=PALETTE[1], label="Best so far")
    pruned = [t.number for t in study.trials if t.state.name == "PRUNED"]
    if pruned: ax.scatter(pruned, [min(y)] * len(pruned), marker="x", s=10, color="0.5", label="Pruned")
    ax.set_xlabel("Trial"); ax.set_ylabel("Validation UAR"); ax.legend(loc="lower right")
    _save(fig, out_dir, "fig_optuna_history"); plt.close(fig)

    imp = optuna.importance.get_param_importances(study)
    names, vals = list(imp.keys())[::-1], list(imp.values())[::-1]
    fig, ax = plt.subplots(figsize=(COL, 2.6))
    ax.barh(names, vals, color=PALETTE[0], height=0.6)
    for i, v in enumerate(vals): ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=6)
    ax.set_xlabel("Hyper-parameter importance (fANOVA)"); ax.grid(axis="y", visible=False)
    _save(fig, out_dir, "fig_optuna_importance"); plt.close(fig)

    try:
        ax = optuna.visualization.matplotlib.plot_parallel_coordinate(
            study, params=["lr", "dropout", "n_freeze", "warmup", "label_smoothing", "aug_timemask"])
        fig = ax.figure; fig.set_size_inches(DBL, 2.6); ax.set_title("")
        _save(fig, out_dir, "fig_optuna_parallel"); plt.close(fig)
    except Exception as e: print("parallel-coordinate skipped:", e)


def fig_training_curves(history, out_dir, name="fig_training_curves"):
    import matplotlib.pyplot as plt
    ep = [h["epoch"] for h in history]
    fig, axs = plt.subplots(1, 2, figsize=(DBL, 2.2))
    axs[0].plot(ep, [h["train_loss"] for h in history], color=PALETTE[0], label="Train loss")
    axs[0].plot(ep, [h["val_loss"] for h in history], color=PALETTE[1], ls="--", label="Val. loss")
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Cross-entropy"); axs[0].legend()
    axs[1].plot(ep, [100 * h["acc"] for h in history], color=PALETTE[0], label="Accuracy")
    axs[1].plot(ep, [100 * h["uar"] for h in history], color=PALETTE[2], ls="--", label="UAR")
    axs[1].plot(ep, [100 * h["f1"] for h in history], color=PALETTE[3], ls=":", label="Macro-F1")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Validation (%)"); axs[1].legend()
    _save(fig, out_dir, name); plt.close(fig)


def fig_confusion(y, yhat, out_dir, name="fig_confusion"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y, yhat, labels=range(8)); cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(COL, COL))
    im = ax.imshow(cmn, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(8)); ax.set_yticks(range(8))
    ax.set_xticklabels([e.capitalize() for e in EMOTIONS], rotation=45, ha="right"); ax.set_yticklabels([e.capitalize() for e in EMOTIONS])
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{100*cmn[i,j]:.0f}", ha="center", va="center", fontsize=6,
                    color="white" if cmn[i, j] > 0.55 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label("Recall")
    _save(fig, out_dir, name); plt.close(fig)


def fig_per_class(y, yhat, out_dir, name="fig_per_class"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y, yhat, labels=range(8), zero_division=0)
    x = np.arange(8); w = 0.26
    fig, ax = plt.subplots(figsize=(DBL, 2.3))
    for k, (v, lab, c) in enumerate([(p, "Precision", PALETTE[0]), (r, "Recall", PALETTE[1]), (f, "F1", PALETTE[2])]):
        ax.bar(x + (k - 1) * w, 100 * v, w * 0.92, color=c, label=lab)
        for xi, vi in zip(x + (k - 1) * w, v): ax.text(xi, 100 * vi + 1, f"{100*vi:.0f}", ha="center", fontsize=5)
    ax.set_xticks(x); ax.set_xticklabels([e.capitalize() for e in EMOTIONS]); ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)"); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.grid(axis="x", visible=False)
    _save(fig, out_dir, name); plt.close(fig)


def fig_roc(y, probs, out_dir, name="fig_roc"):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    Y = label_binarize(y, classes=range(8)); P = np.asarray(probs)
    fig, ax = plt.subplots(figsize=(COL, COL))
    aucs = []
    for i in range(8):
        fpr, tpr, _ = roc_curve(Y[:, i], P[:, i]); a = auc(fpr, tpr); aucs.append(a)
        ax.plot(fpr, tpr, color=PALETTE[i], lw=1, label=f"{EMOTIONS[i].capitalize()} ({a:.2f})")
    ax.plot([0, 1], [0, 1], color="0.6", ls="--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=6, title=f"Macro AUC = {np.mean(aucs):.3f}", title_fontsize=6)
    _save(fig, out_dir, name); plt.close(fig)
    return float(np.mean(aucs))


def fig_tsne(emb, y, out_dir, name="fig_tsne"):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    Z = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(np.asarray(emb))
    y = np.asarray(y)
    fig, ax = plt.subplots(figsize=(COL, COL))
    for i in range(8):
        ax.scatter(Z[y == i, 0], Z[y == i, 1], s=6, color=PALETTE[i], label=EMOTIONS[i].capitalize(), alpha=0.85,
                   edgecolors="white", linewidths=0.2)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False); ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
    ax.legend(ncol=2, fontsize=6, markerscale=1.5, loc="best")
    _save(fig, out_dir, name); plt.close(fig)


def fig_cv_folds(results, out_dir, name="fig_cv_folds"):
    import matplotlib.pyplot as plt
    x = np.arange(len(results)); w = 0.26
    fig, ax = plt.subplots(figsize=(COL, 2.2))
    for k, (key, lab, c) in enumerate([("acc", "Accuracy", PALETTE[0]), ("uar", "UAR", PALETTE[1]), ("f1", "Macro-F1", PALETTE[2])]):
        v = [100 * r["best"][key] for r in results]
        ax.bar(x + (k - 1) * w, v, w * 0.92, color=c, label=lab)
        ax.axhline(np.mean(v), color=c, lw=0.7, ls=":")
    ax.set_xticks(x); ax.set_xticklabels([f"Fold {i+1}" for i in x]); ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 105); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.grid(axis="x", visible=False)
    _save(fig, out_dir, name); plt.close(fig)


def fig_latency(stages: dict, out_dir, name="fig_latency"):
    """stages = {'Audio capture':3000, 'Inference (GPU)':45, 'Cloud publish':210, 'Cloud -> device':800} in ms"""
    import matplotlib.pyplot as plt
    names, vals = list(stages.keys()), list(stages.values())
    fig, ax = plt.subplots(figsize=(COL, 2.0))
    ax.barh(names[::-1], vals[::-1], color=PALETTE[0], height=0.6)
    for i, v in enumerate(vals[::-1]): ax.text(v * 1.05, i, f"{v:.0f} ms", va="center", fontsize=6)
    ax.set_xscale("log"); ax.set_xlabel("Latency (ms, log scale)"); ax.grid(axis="y", visible=False)
    _save(fig, out_dir, name); plt.close(fig)


def fig_ablation(rows: list, out_dir, name="fig_ablation"):
    """rows = [{'name':'MFCC+SVM','acc':..,'uar':..,'f1':..}, ...] values in [0,1]"""
    import matplotlib.pyplot as plt
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=(DBL, 2.3))
    for k, (key, lab, c) in enumerate([("acc", "Accuracy", PALETTE[0]), ("uar", "UAR", PALETTE[1]), ("f1", "Macro-F1", PALETTE[2])]):
        v = [100 * r[key] for r in rows]
        ax.bar(x + (k - 1) * w, v, w * 0.92, color=c, label=lab)
        for xi, vi in zip(x + (k - 1) * w, v): ax.text(xi, vi + 1, f"{vi:.1f}", ha="center", fontsize=5)
    ax.set_xticks(x); ax.set_xticklabels([r["name"] for r in rows]); ax.set_ylim(0, 105)
    ax.set_ylabel("Score (%)"); ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.grid(axis="x", visible=False)
    _save(fig, out_dir, name); plt.close(fig)


# --------------------------------------------------------------------------- tables
def latex_results_table(tab: dict, results, out_dir, auc=None, caption="Speaker-independent 5-fold cross-validation on RAVDESS (mean $\\pm$ std, \\%)."):
    lines = ["\\begin{table}[t]", "\\centering", f"\\caption{{{caption}}}", "\\label{tab:cv}",
             "\\begin{tabular}{lccc}", "\\toprule", "Fold & Accuracy & UAR & Macro-F1 \\\\", "\\midrule"]
    for i, r in enumerate(results):
        lines.append(f"{i+1} & {100*r['best']['acc']:.2f} & {100*r['best']['uar']:.2f} & {100*r['best']['f1']:.2f} \\\\")
    lines += ["\\midrule", "Mean $\\pm$ std & " + " & ".join(f"{100*m:.2f} $\\pm$ {100*s:.2f}" for m, s in
              (tab["acc"], tab["uar"], tab["f1"])) + " \\\\"]
    if auc is not None: lines.append(f"\\multicolumn{{4}}{{l}}{{Macro one-vs-rest AUC = {auc:.3f}}} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    s = "\n".join(lines); open(os.path.join(out_dir, "table_cv.tex"), "w").write(s); print(s)


def latex_ablation_table(rows, out_dir):
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Ablation on the same speaker-independent folds (mean, \\%).}",
             "\\label{tab:ablation}", "\\begin{tabular}{lccc}", "\\toprule", "Configuration & Accuracy & UAR & Macro-F1 \\\\", "\\midrule"]
    for r in rows: lines.append(f"{r['name']} & {100*r['acc']:.2f} & {100*r['uar']:.2f} & {100*r['f1']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    s = "\n".join(lines); open(os.path.join(out_dir, "table_ablation.tex"), "w").write(s); print(s)


def latex_hparams_table(params: dict, out_dir):
    lines = ["\\begin{table}[t]", "\\centering", "\\caption{Optuna search space and selected values.}", "\\label{tab:hparams}",
             "\\begin{tabular}{lll}", "\\toprule", "Hyper-parameter & Search space & Selected \\\\", "\\midrule"]
    space = {"backbone": "\\{WavLM-base+, wav2vec2-base, HuBERT-base\\}", "pooling": "\\{attentive, mean\\}",
             "layer_weighting": "\\{on, off\\}", "n_freeze": "\\{0,2,\\dots,8\\}", "dropout": "[0.05, 0.4]",
             "lr": "[5e-6, 1e-4] (log)", "head_lr_mult": "\\{1,5,10,20\\}", "warmup": "[0, 0.2]",
             "label_smoothing": "[0, 0.15]", "mixup": "\\{0, 0.2, 0.4\\}", "batch_size": "\\{4, 8\\}",
             "aug_timemask": "[0, 0.2]", "aug_speed": "[0, 0.15]"}
    for k, sp in space.items():
        v = params.get(k, "--"); v = f"{v:.3g}" if isinstance(v, float) else str(v).replace("_", "\\_")
        kk = k.replace("_", "\\_")
        lines.append(f"{kk} & {sp} & {v} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    s = "\n".join(lines); open(os.path.join(out_dir, "table_hparams.tex"), "w").write(s); print(s)


# --------------------------------------------------------------------------- baseline (MFCC + SVM) for ablation
def mfcc_svm_baseline(items, cache, k=5):
    import librosa
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    def feat(w):
        m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=40); d = librosa.feature.delta(m)
        return np.concatenate([m.mean(1), m.std(1), d.mean(1), d.std(1)])
    X = {it["path"]: feat(cache[it["path"]]) for it in items}
    folds = speaker_folds(items, k, seed=0); ys, ps = [], []
    for te in folds:
        tr = [i for i in items if i["actor"] not in set(te)]; ts = [i for i in items if i["actor"] in set(te)]
        clf = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")).fit([X[i["path"]] for i in tr], [i["label"] for i in tr])
        ps += list(clf.predict([X[i["path"]] for i in ts])); ys += [i["label"] for i in ts]
    m = metrics(ys, ps); m["name"] = "MFCC + SVM"; print("MFCC+SVM:", m); return m
