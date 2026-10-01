# IoT-Enabled Speech Emotion Recognition with Optuna-Tuned WavLM

End-to-end speech emotion recognition (SER) system: a WavLM-base+ classifier fine-tuned on RAVDESS
runs on a cloud GPU, publishes the predicted emotion to **Arduino Cloud**, and an **Arduino UNO R4 WiFi**
node with a 1.8" ST7735 LCD plus the Arduino IoT Remote mobile app display it in near real time.

Companion code for the paper *An IoT-Enabled Speech Emotion Recognition System Using Optuna-Tuned WavLM
with Cloud-to-Edge Feedback* (S. Bhattacharya, IEM Kolkata).

| Metric (5-fold leave-speakers-out CV, RAVDESS, 8 classes) | Value |
|---|---|
| Accuracy | 78.47 ± 3.65 % |
| UAR (macro recall) | 77.39 ± 4.13 % |
| Macro-F1 | 77.44 ± 3.96 % |
| Macro one-vs-rest AUC | 0.961 |
| Inference (T4 GPU) | 6.5 ms / utterance |
| Cloud publish (REST) | 65 ms |
| Cloud → UNO R4 display | 2.45 s (median, acknowledged) |

![architecture](paper/figs/fig_architecture.png)

## Repository layout

```
arduino/
  SER_CloudOnly/        <- deployed firmware: subscribes to Arduino Cloud, drives the LCD, acks via `detections`
  SER_Cloud/            <- variant that also streams Grove-mic audio over USB to a local PC server
  SER_Bridge/           <- USB audio bridge without cloud (for ser_server.py)
  SpeechEmotion_UnoR4/  <- fully on-device prosody-based baseline (no cloud, no ML model)
notebooks/
  SER_IEEE_Optuna_Colab.ipynb      <- Optuna HPO, 5-fold CV, ablation, IEEE figures, latency measurement
  SER_wav2vec2_RAVDESS_Colab.ipynb <- single-model training + live demo pushing labels to Arduino Cloud
src/
  ser_ieee.py                  <- data, model, training, Optuna, CV, figure and LaTeX-table generation
  train_wav2vec2_ravdess.py    <- standalone fine-tuning script (HF Trainer)
  ser_server.py                <- local PC inference server for the USB-bridge variants
paper/
  main.tex, body.tex, preamble.tex, refs.bib, figs/   <- IEEEtran conference source
  IoT_SER_paper_preview.pdf
results/
  best_params.json             <- Optuna-selected configuration
  cv_results.json              <- per-fold predictions, probabilities, embeddings (proposed model)
  cv_results_ablation_attn_lw.json
  ablation.json, latency.json, table_*.tex
```

Trained weights (`best_fold_model.pt`, 361 MB) are too large for the repository; download them from the
**Releases** page and place the file under `results/`.

## Reproduce

### 1. Model training, HPO and figures (Google Colab, T4 GPU)

Open `notebooks/SER_IEEE_Optuna_Colab.ipynb` in Colab and run the sections in order.
Section 1 downloads RAVDESS (speech subset, 1440 clips) into your Google Drive and caches it in RAM;
Section 2 runs the Optuna study (resumable SQLite on Drive); Section 3 the 5-fold cross-validation;
Section 4 the ablation; Section 5 writes every figure (PDF + 600-dpi PNG) and LaTeX table to `paper_assets/`.

Protocol: speaker-independent throughout. Actors are split into five gender-balanced folds; the
hyper-parameter search uses an inner split whose speakers are disjoint from test fold 1.

### 2. Edge node

1. Arduino Cloud → create a Thing with variables `emotion` (String, R/W), `confidence` (Float, R/W),
   `detections` (Int, read-only, on change). Associate the UNO R4 WiFi and set its network.
2. Open `arduino/SER_CloudOnly/` in the Arduino IDE (or paste the `.ino` into the Cloud Editor, which
   generates `thingProperties.h` for you). Copy `arduino_secrets.h.example` to `arduino_secrets.h` and fill it in.
3. Set the `TFT_*` pin defines to your wiring. Upload over USB or, if the board is already online, over the air.

### 3. Live demo

In `notebooks/SER_wav2vec2_RAVDESS_Colab.ipynb`, Part B records from the browser microphone, classifies on
the GPU, and publishes to Arduino Cloud via its REST API. Create an API key in Arduino Cloud and paste the
Client ID / Secret into the config cell; the notebook looks up the Thing and variable IDs by name.

### 4. Paper

`paper/` compiles with IEEEtran on Overleaf (`pdflatex → bibtex → pdflatex ×2`).
Remove the `\todo{}` placeholders (author details, repository URL) before submission.

## Requirements

Python ≥ 3.10, PyTorch ≥ 2.1, `transformers ≥ 4.40`, `optuna`, `torchaudio`, `scikit-learn`, `librosa`,
`matplotlib`; see the first cell of each notebook. Arduino: UNO R4 Boards core, ArduinoIoTCloud,
Arduino_ConnectionHandler, Adafruit GFX, Adafruit ST7735/ST7789.

## Citation

```bibtex
@inproceedings{bhattacharya2026iotser,
  title     = {An IoT-Enabled Speech Emotion Recognition System Using Optuna-Tuned WavLM with Cloud-to-Edge Feedback},
  author    = {Bhattacharya, Sudipta},
  booktitle = {Proc. IEEE Conference (under review)},
  year      = {2026}
}
```

## License

MIT (code). RAVDESS is distributed under CC BY-NC-SA 4.0 by its authors and is not included here.
