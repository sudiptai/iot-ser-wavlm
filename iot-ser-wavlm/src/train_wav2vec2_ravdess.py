"""
Fine-tune wav2vec 2.0 on RAVDESS (speech, 8 emotions).

Split: speaker-independent when >=6 actors are present (last 2 actors = test,
2 before = val); otherwise a stratified random 80/10/10 split.

Usage:
    python train_wav2vec2_ravdess.py --download          # fetch RAVDESS speech zip (~208 MB)
    python train_wav2vec2_ravdess.py                     # train (uses ./ravdess)
    python train_wav2vec2_ravdess.py --epochs 20 --base facebook/wav2vec2-base

Output: ./wav2vec2-ravdess-ser/  (model + feature extractor, ready for ser_server.py)

Split is speaker-independent: actors 01-20 train, 21-22 val, 23-24 test.
"""
import argparse, glob, os, re, sys, zipfile, urllib.request
import numpy as np
import torch, torchaudio
from torch.utils.data import Dataset
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification,
                          Trainer, TrainingArguments)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

RAVDESS_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LABEL2ID = {e: i for i, e in enumerate(EMOTIONS)}
ID2LABEL = {i: e for e, i in LABEL2ID.items()}
SR = 16000
MAX_SEC = 4.0


def download(root):
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, "Audio_Speech_Actors_01-24.zip")
    if not os.path.exists(zpath):
        print("Downloading RAVDESS speech ...")
        urllib.request.urlretrieve(RAVDESS_URL, zpath)
    print("Extracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    print("Done:", root)


NAME_RE = re.compile(r"^(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)")

def parse(path):
    # 03-01-06-01-02-01-12.wav -> modality-channel-emotion-intensity-statement-rep-actor
    # tolerates Drive duplicates like "03-01-06-01-02-01-12 (1).wav"
    m = NAME_RE.match(os.path.basename(path))
    if not m:
        return None
    p = [int(x) for x in m.groups()]
    return {"path": path, "key": m.group(0), "modality": p[0], "channel": p[1],
            "label": p[2] - 1, "intensity": p[3], "actor": p[6]}


def load_index(root):
    files = sorted(glob.glob(os.path.join(root, "**", "*.wav"), recursive=True))
    if not files:
        sys.exit(f"No wav files under {root}. Run with --download first.")
    seen, items, skipped = set(), [], 0
    for f in files:
        it = parse(f)
        if it is None or it["modality"] != 3 or it["channel"] != 1:   # audio-only speech
            skipped += 1; continue
        if it["key"] in seen:                                          # Drive "(1)" duplicates
            skipped += 1; continue
        seen.add(it["key"]); items.append(it)
    print(f"{len(items)} usable clips, {skipped} skipped (non-speech / duplicates / unparsable)")
    actors = sorted({it["actor"] for it in items})
    print(f"actors present: {actors}")
    if len(actors) >= 6:
        # speaker-independent: last 2 actors -> test, 2 before -> val, rest -> train
        te_a, va_a = set(actors[-2:]), set(actors[-4:-2])
        train = [it for it in items if it["actor"] not in te_a | va_a]
        val   = [it for it in items if it["actor"] in va_a]
        test  = [it for it in items if it["actor"] in te_a]
        print(f"speaker-independent split  val actors {sorted(va_a)}  test actors {sorted(te_a)}")
    else:
        # too few speakers: stratified random 80/10/10 (NOT speaker-independent)
        from sklearn.model_selection import train_test_split
        y = [it["label"] for it in items]
        train, rest = train_test_split(items, test_size=0.2, stratify=y, random_state=42)
        yr = [it["label"] for it in rest]
        val, test = train_test_split(rest, test_size=0.5, stratify=yr, random_state=42)
        print("WARNING: <6 actors, using stratified random split (not speaker-independent)")
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")
    if len(val) < 8 or len(test) < 8:
        sys.exit("val/test too small - need more clips (use --download for the full RAVDESS set)")
    return train, val, test


def load_audio(path):
    wav, sr = torchaudio.load(path)
    wav = wav.mean(0)                                   # mono
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    # trim leading/trailing silence (RAVDESS clips have ~0.5 s padding)
    thr = 0.01 * wav.abs().max()
    idx = torch.nonzero(wav.abs() > thr)
    if len(idx) > 0:
        wav = wav[idx[0, 0]: idx[-1, 0] + 1]
    return wav[: int(MAX_SEC * SR)].numpy()


class RavdessDS(Dataset):
    def __init__(self, items, fe, augment=False):
        self.items, self.fe, self.augment = items, fe, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        wav = load_audio(it["path"])
        if self.augment:
            wav = wav * np.random.uniform(0.7, 1.3)
            wav = wav + np.random.normal(0, 0.003, wav.shape).astype(np.float32)
        feats = self.fe(wav, sampling_rate=SR, return_tensors="np")
        return {"input_values": feats["input_values"][0], "labels": it["label"]}


def collate(batch):
    lens = [len(b["input_values"]) for b in batch]
    L = max(lens)
    x = np.zeros((len(batch), L), dtype=np.float32)
    mask = np.zeros((len(batch), L), dtype=np.int64)
    for i, b in enumerate(batch):
        x[i, : lens[i]] = b["input_values"]
        mask[i, : lens[i]] = 1
    return {"input_values": torch.tensor(x),
            "attention_mask": torch.tensor(mask),
            "labels": torch.tensor([b["labels"] for b in batch])}


def compute_metrics(p):
    pred = p.predictions.argmax(-1)
    return {"accuracy": accuracy_score(p.label_ids, pred),
            "f1_macro": f1_score(p.label_ids, pred, average="macro")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./ravdess")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--base", default="facebook/wav2vec2-base")
    ap.add_argument("--out", default="./wav2vec2-ravdess-ser")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-5)
    args = ap.parse_args()

    if args.download:
        download(args.root)          # extracts Actor_01..Actor_24 folders under --root

    train, val, test = load_index(args.root)
    fe = Wav2Vec2FeatureExtractor.from_pretrained(args.base)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        args.base, num_labels=len(EMOTIONS), label2id=LABEL2ID, id2label=ID2LABEL,
        problem_type="single_label_classification")
    model.freeze_feature_encoder()   # keep the CNN front-end frozen; fine-tune transformer + head

    use_cuda = torch.cuda.is_available()
    targs = TrainingArguments(
        output_dir=args.out + "-ckpt",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        save_total_limit=2,
        fp16=use_cuda,
        logging_steps=20,
        report_to="none",
        dataloader_num_workers=2 if use_cuda else 0,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=RavdessDS(train, fe, augment=True),
                      eval_dataset=RavdessDS(val, fe),
                      data_collator=collate, compute_metrics=compute_metrics)
    trainer.train()

    print("\n=== Held-out test (actors 23-24) ===")
    pred = trainer.predict(RavdessDS(test, fe))
    y, yhat = pred.label_ids, pred.predictions.argmax(-1)
    print(classification_report(y, yhat, target_names=EMOTIONS, digits=3))
    print(confusion_matrix(y, yhat))

    trainer.save_model(args.out)
    fe.save_pretrained(args.out)
    print("Saved to", args.out)


if __name__ == "__main__":
    main()
