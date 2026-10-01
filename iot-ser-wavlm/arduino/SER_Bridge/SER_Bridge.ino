/*
  SER_Bridge - Arduino UNO R4 WiFi
  Streams Grove Sound Sensor audio (8 kHz, 8-bit) to the PC over USB serial
  and shows the emotion label sent back by ser_server.py on the 1.8" ST7735 LCD.

  PC -> Arduino protocol:  "L:<LABEL>:<conf>\n"   e.g.  L:HAPPY:0.87
*/
#include <SPI.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ST7735.h>

// ---- display pins: EDIT to match your wiring ----
#define TFT_CS   10
#define TFT_DC    9
#define TFT_RST   8
#define TFT_BL    7
#define TFT_MOSI 11      // DIN
#define TFT_SCLK 13      // CLK
#define MIC_PIN  A0

// Software SPI: works on ANY pins, so the defines above can be anything you wired.
Adafruit_ST7735 tft(TFT_CS, TFT_DC, TFT_MOSI, TFT_SCLK, TFT_RST);

const uint32_t SAMPLE_US = 125;   // 8 kHz
#define CHUNK 128
uint8_t chunk[CHUNK];
char line[32];
uint8_t lineLen = 0;

uint16_t colourFor(const char *l) {
  if (!strcmp(l, "HAPPY"))     return ST77XX_YELLOW;
  if (!strcmp(l, "ANGRY"))     return ST77XX_RED;
  if (!strcmp(l, "SAD"))       return ST77XX_CYAN;
  if (!strcmp(l, "FEARFUL"))   return ST77XX_MAGENTA;
  if (!strcmp(l, "DISGUST"))   return 0x8410;          // olive
  if (!strcmp(l, "SURPRISED")) return ST77XX_ORANGE;
  if (!strcmp(l, "CALM"))      return 0x07FF;          // light blue
  return ST77XX_GREEN;                                  // NEUTRAL / READY
}

void header() {
  tft.fillScreen(ST77XX_BLACK);
  tft.fillRect(0, 0, 160, 14, ST77XX_BLUE);
  tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
  tft.setCursor(14, 3); tft.print("wav2vec2 SER  (RAVDESS)");
}

void showLabel(const char *label, const char *conf) {
  header();
  uint8_t sz = strlen(label) > 7 ? 2 : 3;
  tft.setTextSize(sz);
  tft.setTextColor(colourFor(label));
  int w = strlen(label) * 6 * sz;
  tft.setCursor((160 - w) / 2, 40);
  tft.print(label);
  if (conf && conf[0]) {
    tft.setTextSize(1); tft.setTextColor(ST77XX_WHITE);
    tft.setCursor(50, 80); tft.print("conf: "); tft.print(conf);
  }
}

void handleLine() {
  line[lineLen] = 0;
  if (line[0] == 'L' && line[1] == ':') {
    char *label = line + 2;
    char *conf = strchr(label, ':');
    if (conf) { *conf = 0; conf++; }
    if (!strcmp(label, "READY")) showLabel("READY", "");
    else showLabel(label, conf);
  }
  lineLen = 0;
}

void setup() {
  Serial.begin(115200);
  analogReadResolution(12);
  pinMode(TFT_BL, OUTPUT); digitalWrite(TFT_BL, HIGH);
  tft.initR(INITR_BLACKTAB);        // try INITR_GREENTAB if colours/offset look wrong
  tft.setRotation(1);
  tft.setTextWrap(false);
  header();
  tft.setTextColor(ST77XX_YELLOW); tft.setCursor(10, 40);
  tft.print("Waiting for PC...");
  tft.setTextColor(ST77XX_WHITE);  tft.setCursor(10, 56);
  tft.print("run ser_server.py");
}

void loop() {
  // --- capture one chunk at 8 kHz ---
  uint32_t t = micros();
  for (int i = 0; i < CHUNK; i++) {
    while ((uint32_t)(micros() - t) < SAMPLE_US) { }
    t += SAMPLE_US;
    chunk[i] = analogRead(MIC_PIN) >> 4;   // 12-bit -> 8-bit unsigned
  }
  Serial.write(chunk, CHUNK);

  // --- read any label lines from the PC ---
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') handleLine();
    else if (c != '\r' && lineLen < sizeof(line) - 1) line[lineLen++] = c;
  }
}
