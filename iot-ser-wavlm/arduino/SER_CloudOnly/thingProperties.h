// thingProperties.h for SER_CloudOnly.
// If you create the Thing in the Arduino Cloud web editor it generates this file
// for you - use THAT one, just make sure the variables and callbacks match.
#include <ArduinoIoTCloud.h>
#include <Arduino_ConnectionHandler.h>
#include "arduino_secrets.h"

const char DEVICE_LOGIN_NAME[] = SECRET_DEVICE_ID;
const char SSID[]              = SECRET_SSID;
const char PASS[]              = SECRET_OPTIONAL_PASS;
const char DEVICE_KEY[]        = SECRET_DEVICE_KEY;

void onEmotionChange();
void onConfidenceChange();

// emotion and confidence are written by Colab -> must be READWRITE
String emotion;       // "HAPPY", "ANGRY", ...
float  confidence;    // 0-100 %
int    detections;    // counter kept by the board

void initProperties() {
  ArduinoCloud.setBoardId(DEVICE_LOGIN_NAME);
  ArduinoCloud.setSecretDeviceKey(DEVICE_KEY);
  ArduinoCloud.addProperty(emotion,    READWRITE, ON_CHANGE, onEmotionChange);
  ArduinoCloud.addProperty(confidence, READWRITE, ON_CHANGE, onConfidenceChange);
  ArduinoCloud.addProperty(detections, READ,      ON_CHANGE, NULL);
}

WiFiConnectionHandler ArduinoIoTPreferredConnection(SSID, PASS);
