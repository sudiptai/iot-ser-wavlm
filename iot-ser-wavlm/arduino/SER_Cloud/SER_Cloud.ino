/*
  SER_Cloud - Arduino UNO R4 WiFi + Arduino Cloud
  Streams Grove mic audio to the PC (ser_server.py runs wav2vec2),
  receives the label back over USB, shows it on the ST7735 LCD and
  publishes it to Arduino Cloud so the Arduino IoT Remote app shows it.

  Files in this sketch folder: SER_Cloud.ino, thingProperties.h, arduino_secrets.h
  Libraries: ArduinoIoTCloud, Arduino_ConnectionHandler, Adafruit GFX, Adafruit ST7735
*/
#include "thingProperties.h"
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

// ---- display pins: EDIT to match your wiring (software SPI, any pins work) ----
#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8
#define TFT_BL    7
#define TFT_MOSI 11      // DIN
#define TFT_SCLK 13      // CLK
#define MIC_PIN  A0

Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

const uint32_t SAMPLE_US = 125;      // 8 kHz
#define CHUNK 128
uint8_t  chunk[CHUNK];
char     line[40];
uint8_t  lineLen = 0;
uint32_t lastCloud = 0;
bool     cloudDirty = false;

uint16_t colourFor(const String &l) {
  if (l == "HAPPY")     return ST77XX_YELLOW;
  if (l == "ANGRY")     return ST77XX_RED;
  if (l == "SAD")       return ST77XX_CYAN;
  if (l == "FEARFUL")   return ST77XX_MAGENTA;
  if (l == "DISGUST")   return 0x8410;
  if (l == "SURPRISED") return ST77XX_ORANGE;
  if (l == "CALM")      return 0x07FF;
  return ST77XX_GREEN;
}

void header() {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 14, ST77XX_BLUE);
  tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(8, 3); tft.print("wav2vec2 SER  + Cloud");
  // cloud status dot, top-right
  tft.fillCircle(152, 7, 4, ArduinoCloud.connected() ? ST77XX_GREEN : ST77XX_RED);
}

void showLabel(const String &label, float conf) {
  header();
  uint8_t sz = label.length() > 7 ? 2 : 3;
  tft.setTextSize(sz); tft.setTextColor(colourFor(label));
  tft.setCursor((160 - label.length() * 6 * sz) / 2, 40);
  tft.print(label);
  if (conf > 0) {
    tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(50, 80); tft.print("conf: "); tft.print(conf, 2);
  }
}

void handleLine() {
  line[lineLen] = 0;
  if (line[0] == 'L' && line[1] == ':') {
    char *lab = line + 2;
    char *c = strchr(lab, ':');
    float conf = 0;
    if (c) { *c = 0; conf = atof(c + 1); }
    String l(lab);
    showLabel(l, conf);
    if (l != "READY") {
      emotion    = l;                 // cloud variables (thingProperties.h)
      confidence = conf * 100.0f;     // as percent
      detections = detections + 1;
      cloudDirty = true;
    }
  }
  lineLen = 0;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH);
  tft.initR(INITR_BLACKTAB);          // try INITR_GREENTAB if colours/offset look wrong
  tft.setRotation(1); tft.setTextWrap(false);

  initProperties();
  ArduinoCloud.begin(ArduinoIoTPreferredConnection);
  setDebugMessageLevel(0);            // keep USB stream clean (debug goes to Serial otherwise)

  header();
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(10, 40); tft.print("Connecting WiFi...");
  uint32_t t0 = millis();
  while (!ArduinoCloud.connected() && millis() - t0 < 20000) ArduinoCloud.update();

  emotion = "READY"; confidence = 0; detections = 0;
  ArduinoCloud.update();
  header();
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(10, 40); tft.print("Waiting for PC...");
  tft.setTextColor(ST77XX_WHITE);  tft.setCursor(10, 56); tft.print("run ser_server.py");
}

void loop() {
  // 1. capture + stream one 16 ms chunk at 8 kHz
  uint32_t t = micros();
  for (int i = 0; i < CHUNK; i++) {
    while ((uint32_t)(micros() - t) < SAMPLE_US) { }
    t += SAMPLE_US;
    chunk[i] = analogRead(MIC_PIN) >> 4;
  }
  Serial.write(chunk, CHUNK);

  // 2. labels from the PC
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') handleLine();
    else if (c != '\r' && lineLen < sizeof(line) - 1) line[lineLen++] = c;
  }

  // 3. cloud sync: right after a new label (speaker is between utterances,
  //    so the short audio gap is harmless), otherwise a keep-alive every 3 s
  if (cloudDirty || millis() - lastCloud > 3000) {
    ArduinoCloud.update();
    lastCloud = millis();
    cloudDirty = false;
  }
}
