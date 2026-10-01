"""
SER inference server: wav2vec2 (fine-tuned on RAVDESS) <-> Arduino UNO R4 bridge.

Audio source options:
    --source arduino  : raw 8 kHz, 8-bit unsigned samples streamed by SER_Bridge.ino over USB
    --source mic      : your laptop/USB microphone at 16 kHz (better quality)

In both modes the predicted label is sent to the Arduino as  "L:<EMOTION>:<conf>\n"
so the LCD shows it.

Usage:
    python ser_server.py --port COM5 --source arduino
    python ser_server.py --port COM5 --source mic
"""
import argparse, sys, time, threading, collections
import numpy as np
import torch, torchaudio
import serial
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

ARD_SR = 8000
SR = 16000
MIN_UTT_SEC, MAX_UTT_SEC = 0.6, 4.0
SILENCE_SEC = 0.5
FRAME = 256


def load_model(path):
    fe = Wav2Vec2FeatureExtractor.from_pretrained(path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(path).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return fe, model.to(dev), dev


@torch.no_grad()
def predict(fe, model, dev, wav16k):
    wav16k = wav16k / (np.abs(wav16k).max() + 1e-6)
    x = fe(wav16k, sampling_rate=SR, return_tensors="pt")["input_values"].to(dev)
    probs = torch.softmax(model(x).logits[0], -1).cpu().numpy()
    i = int(probs.argmax())
    return model.config.id2label[i], float(probs[i]), probs


class UtteranceDetector:
    """Energy-gated utterance segmentation on a float stream at `sr`."""
    def __init__(self, sr):
        self.sr = sr
        self.noise = None
        self.buf, self.silence, self.active = [], 0, False

    def calibrate(self, x):
        rms = np.sqrt(np.mean(x ** 2))
        self.noise = rms if self.noise is None else 0.9 * self.noise + 0.1 * rms

    def push(self, x):
        """Return a finished utterance (np.float32) or None."""
        rms = np.sqrt(np.mean(x ** 2))
        if self.noise is None:
            self.noise = rms
            return None
        thr = max(self.noise * 3.0, 0.004)
        if rms > thr:
            self.active, self.silence = True, 0
            self.buf.append(x)
        elif self.active:
            self.silence += len(x)
            self.buf.append(x)
            if self.silence > SILENCE_SEC * self.sr:
                return self._finish()
        else:
            self.noise = 0.98 * self.noise + 0.02 * rms   # track noise floor while idle
        if self.active and sum(map(len, self.buf)) > MAX_UTT_SEC * self.sr:
            return self._finish()
        return None

    def _finish(self):
        utt = np.concatenate(self.buf).astype(np.float32)
        self.buf, self.silence, self.active = [], 0, False
        return utt if len(utt) >= MIN_UTT_SEC * self.sr else None


def send_label(ser, label, conf):
    if ser:
        ser.write(f"L:{label.upper()}:{conf:.2f}\n".encode())


def run_arduino(ser, fe, model, dev):
    ser.reset_input_buffer()
    det = UtteranceDetector(ARD_SR)
    print("Calibrating noise (stay quiet 2 s) ...")
    t0 = time.time()
    while time.time() - t0 < 2.0:
        raw = ser.read(FRAME)
        if raw:
            det.calibrate((np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0)
    send_label(ser, "READY", 0)
    print("Listening on Arduino mic. Speak with emotion.")
    while True:
        raw = ser.read(FRAME)
        if not raw:
            continue
        x = (np.frombuffer(raw, np.uint8).astype(np.float32) - 128.0) / 128.0
        utt = det.push(x)
        if utt is None:
            continue
        wav = torchaudio.functional.resample(torch.tensor(utt), ARD_SR, SR).numpy()
        label, conf, probs = predict(fe, model, dev, wav)
        print(f"{label:10s} {conf:.2f}   " + "  ".join(
            f"{model.config.id2label[i]}={p:.2f}" for i, p in enumerate(probs)))
        send_label(ser, label, conf)


def run_mic(ser, fe, model, dev):
    import sounddevice as sd
    det = UtteranceDetector(SR)
    q = collections.deque()

    def cb(indata, frames, t, status):
        q.append(indata[:, 0].copy())

    with sd.InputStream(samplerate=SR, channels=1, blocksize=1024, callback=cb):
        print("Calibrating noise (stay quiet 2 s) ...")
        t0 = time.time()
        while time.time() - t0 < 2.0:
            while q: det.calibrate(q.popleft())
            time.sleep(0.05)
        send_label(ser, "READY", 0)
        print("Listening on PC mic. Speak with emotion.")
        while True:
            while q:
                utt = det.push(q.popleft())
                if utt is not None:
                    label, conf, probs = predict(fe, model, dev, utt)
                    print(f"{label:10s} {conf:.2f}")
                    send_label(ser, label, conf)
            time.sleep(0.02)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./wav2vec2-ravdess-ser")
    ap.add_argument("--port", required=True, help="e.g. COM5 or /dev/ttyACM0")
    ap.add_argument("--source", choices=["arduino", "mic"], default="arduino")
    args = ap.parse_args()

    fe, model, dev = load_model(args.model)
    print("Model loaded on", dev)
    ser = serial.Serial(args.port, 115200, timeout=0.05)
    time.sleep(2.5)                       # let the R4 reboot after port open
    try:
        (run_arduino if args.source == "arduino" else run_mic)(ser, fe, model, dev)
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
