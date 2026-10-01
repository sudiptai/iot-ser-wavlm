/*
  Speech Emotion Detection - Arduino UNO R4 WiFi
  Mic    : Seeed Grove Sound Sensor v1.7  (SIG -> A0)
  Display: SmartElex 1.8" 128x160 SPI LCD (ST7735)

  Method : frame-wise prosodic features (RMS energy, pitch via
           autocorrelation, pitch variation, zero-crossing rate),
           speaker-relative calibration, rule-based classifier.
  Output : emotion on LCD + feature CSV on Serial (115200) for dataset
           collection / training a better classifier later.

  Libraries (Library Manager):
    - Adafruit GFX Library
    - Adafruit ST7735 and ST7789 Library
*/

#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

// ---------------- PINS: change these to match YOUR wiring ----------------
// Hardware SPI on UNO R4 is fixed:  CLK -> D13,  DIN -> D11
#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8
#define TFT_BL    7      // backlight; or wire BL straight to 3.3V and ignore
#define MIC_PIN  A0

// ---------------- TUNABLE THRESHOLDS ----------------
const float AROUSAL_HIGH   = 0.35f;  // above -> Happy/Angry
const float AROUSAL_LOW    = -0.15f; // below -> Sad (if pitch is flat)
const float ANGRY_ENERGY   = 1.5f;   // energy ratio needed for Angry
const float ANGRY_ZCR      = 1.15f;  // ZCR ratio (harsh voice) for Angry
const float VOICED_CORR    = 0.35f;  // autocorrelation peak to accept pitch
const uint8_t CAL_UTTERANCES = 3;    // neutral sentences used as baseline

// ---------------- AUDIO ----------------
#define N 256                         // samples per frame (~32 ms)
const uint32_t SAMPLE_US = 125;       // target 8 kHz
int16_t  buf[N];
float    x[N];
float    fsActual = 8000.0f;

// per-frame features
float fRms, fZcr, fPitch;             // fPitch = 0 if unvoiced

// noise / VAD
float noiseRms = 5, vadThresh = 20;

// utterance accumulators
bool     inUtt = false;
uint16_t nFrames, nVoiced, silence;
float    sumRms, sumZcr, sumP, sumP2;

// speaker baseline
uint8_t  calCount = 0;
float    basePitch = 0, baseRms = 0, baseZcr = 0, baseCv = 0;

Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_RST);

// ---------------- DISPLAY HELPERS ----------------
void header() {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 14, ST77XX_BLUE);
  tft.setTextColor(ST77XX_WHITE);
  tft.setTextSize(1);
  tft.setCursor(22, 3);
  tft.print("SPEECH EMOTION (SER)");
}

void message(const char *l1, const char *l2) {
  header();
  tft.setTextColor(ST77XX_YELLOW);
  tft.setCursor(6, 40);  tft.print(l1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(6, 58);  tft.print(l2);
}

void drawLevel(float rms) {
  float top = vadThresh * 8.0f;
  int w = (int)(156.0f * constrain(rms, 0, top) / top);
  uint16_t c = (rms > vadThresh) ? ST77XX_GREEN : 0x7BEF; // grey when silent
  tft.fillRect(2, 118, w, 8, c);
  tft.fillRect(2 + w, 118, 156 - w, 8, ST77XX_BLACK);
}

void showEmotion(const char *label, uint16_t color,
                 float pitch, float eR, float pR, float cv) {
  header();
  tft.setTextSize(3);
  tft.setTextColor(color);
  int wpx = strlen(label) * 18;
  tft.setCursor((160 - wpx) / 2, 28);
  tft.print(label);

  tft.setTextSize(1);
  tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(6, 66);  tft.print("Pitch : "); tft.print(pitch, 0); tft.print(" Hz");
  tft.setCursor(6, 78);  tft.print("P-ratio: "); tft.print(pR, 2);
  tft.setCursor(6, 90);  tft.print("Energy : "); tft.print(eR, 2); tft.print(" x");
  tft.setCursor(6, 102); tft.print("P-var  : "); tft.print(cv, 2);
}

// ---------------- AUDIO CAPTURE + FEATURES ----------------
void captureFrame() {
  uint32_t t0 = micros(), t = t0;
  for (int i = 0; i < N; i++) {
    while ((uint32_t)(micros() - t) < SAMPLE_US) { }
    t += SAMPLE_US;
    buf[i] = analogRead(MIC_PIN);
  }
  fsActual = (float)N * 1e6f / (float)(micros() - t0);
}

void analyzeFrame(bool wantPitch) {
  float mean = 0;
  for (int i = 0; i < N; i++) mean += buf[i];
  mean /= N;

  float e = 0;
  for (int i = 0; i < N; i++) { x[i] = buf[i] - mean; e += x[i] * x[i]; }
  fRms = sqrt(e / N);

  // zero-crossing rate with hysteresis (ignores tiny noise wiggles)
  float h = 0.2f * fRms;
  int zc = 0, sign = 0;
  for (int i = 0; i < N; i++) {
    int s = (x[i] > h) ? 1 : (x[i] < -h ? -1 : 0);
    if (s != 0) { if (sign != 0 && s != sign) zc++; sign = s; }
  }
  fZcr = (float)zc / N;

  // pitch by autocorrelation, 80..400 Hz
  fPitch = 0;
  if (!wantPitch || e < 1.0f) return;
  int lagMin = (int)(fsActual / 400.0f);
  int lagMax = (int)(fsActual / 80.0f);
  if (lagMax > N / 2) lagMax = N / 2;
  float best = 0; int bestLag = 0;
  for (int lag = lagMin; lag <= lagMax; lag++) {
    float r = 0;
    for (int i = 0; i < N - lag; i++) r += x[i] * x[i + lag];
    r /= e;
    if (r > best) { best = r; bestLag = lag; }
  }
  if (best > VOICED_CORR && bestLag > 0) fPitch = fsActual / bestLag;
}

void calibrateNoise() {
  message("Calibrating noise...", "Stay quiet for 2 sec");
  float s = 0; int n = 60;
  for (int i = 0; i < n; i++) { captureFrame(); analyzeFrame(false); s += fRms; }
  noiseRms  = s / n;
  vadThresh = max(noiseRms * 2.5f, noiseRms + 15.0f);
  Serial.print("# noiseRms="); Serial.print(noiseRms);
  Serial.print(" vadThresh="); Serial.print(vadThresh);
  Serial.print(" fs=");        Serial.println(fsActual);
}

void promptCalibration() {
  char l2[32];
  snprintf(l2, sizeof(l2), "sentence  (%d of %d)", calCount + 1, CAL_UTTERANCES);
  message("Say a NEUTRAL", l2);
}

void resetUtt() {
  inUtt = false; nFrames = nVoiced = silence = 0;
  sumRms = sumZcr = sumP = sumP2 = 0;
}

// ---------------- CLASSIFIER ----------------
void finishUtterance() {
  if (nFrames < 10 || nVoiced < 5) { resetUtt(); return; }   // too short / noise

  float mRms = sumRms / nFrames;
  float mZcr = sumZcr / nFrames;
  float mP   = sumP / nVoiced;
  float var  = sumP2 / nVoiced - mP * mP;
  float cv   = (var > 0 ? sqrt(var) : 0) / mP;               // pitch variation

  if (calCount < CAL_UTTERANCES) {                           // build baseline
    basePitch += mP; baseRms += mRms; baseZcr += mZcr; baseCv += cv;
    calCount++;
    if (calCount == CAL_UTTERANCES) {
      basePitch /= calCount; baseRms /= calCount;
      baseZcr   /= calCount; baseCv  /= calCount;
      if (baseZcr < 0.001f) baseZcr = 0.001f;
      Serial.println("pitch,pRatio,eRatio,zRatio,cv,label");
      message("Ready!", "Speak with emotion...");
    } else promptCalibration();
    resetUtt();
    return;
  }

  float pR = mP / basePitch;
  float eR = mRms / baseRms;
  float zR = mZcr / baseZcr;
  float arousal = 0.5f * (eR - 1.0f) + 1.5f * (pR - 1.0f);

  const char *label; uint16_t color;
  if (arousal > AROUSAL_HIGH) {
    if (eR > ANGRY_ENERGY && zR > ANGRY_ZCR) { label = "ANGRY";  color = ST77XX_RED; }
    else                                     { label = "HAPPY";  color = ST77XX_YELLOW; }
  } else if (arousal < AROUSAL_LOW && cv <= baseCv) {
    label = "SAD";     color = ST77XX_CYAN;
  } else {
    label = "NEUTRAL"; color = ST77XX_GREEN;
  }

  showEmotion(label, color, mP, eR, pR, cv);
  Serial.print(mP, 1); Serial.print(','); Serial.print(pR, 3); Serial.print(',');
  Serial.print(eR, 3); Serial.print(','); Serial.print(zR, 3); Serial.print(',');
  Serial.print(cv, 3); Serial.print(','); Serial.println(label);
  resetUtt();
}

// ---------------- SETUP / LOOP ----------------
void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH);

  tft.initR(INITR_BLACKTAB);   // wrong colours / shifted edges? try INITR_GREENTAB
  tft.setRotation(1);          // landscape 160x128
  tft.setTextWrap(false);

  resetUtt();
  calibrateNoise();
  promptCalibration();
}

void loop() {
  captureFrame();
  analyzeFrame(true);
  drawLevel(fRms);

  if (fRms > vadThresh) {                 // speech frame
    inUtt = true; silence = 0;
    nFrames++; sumRms += fRms; sumZcr += fZcr;
    if (fPitch > 0) { nVoiced++; sumP += fPitch; sumP2 += fPitch * fPitch; }
    if (nFrames > 125) finishUtterance(); // cap at ~4 s
  } else if (inUtt) {
    if (++silence > 15) finishUtterance(); // ~0.5 s pause ends the utterance
  }
}
