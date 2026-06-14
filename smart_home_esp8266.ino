#include <ESP8266WiFi.h>
#include <PubSubClient.h>

const char* WIFI_SSID     = "International University 2.4Ghz";
const char* WIFI_PASSWORD = "";

const char* MQTT_BROKER   = "khiet1111.cloud.shiftr.io";
const int   MQTT_PORT     = 1883;
const char* MQTT_USER     = "khiet1111";
const char* MQTT_PASSWORD = "khiet";
const char* MQTT_CLIENT   = "ESP8266_SmartHome";

const int          DOOR_PULSE_MS       = 500;
const int          WINDOW_RUN_MS       = 3000;
const unsigned long STATUS_INTERVAL_MS = 15000;

const int PIN_LIGHTS   = 5;
const int PIN_AC       = 0;
const int PIN_DOOR     = 15;
const int PIN_WINDOW_F = 12;
const int PIN_WINDOW_R = 13;
const int PIN_LED      = 2;

const int RELAY_ON  = HIGH;
const int RELAY_OFF = LOW;

const char* T_LIGHTS_ON   = "/smart_home/lights/on";
const char* T_LIGHTS_OFF  = "/smart_home/lights/off";
const char* T_AC_ON       = "/smart_home/ac/on";
const char* T_AC_OFF      = "/smart_home/ac/off";
const char* T_DOOR_TOGGLE = "/smart_home/door/toggle";
const char* T_WINDOW_ON   = "/smart_home/window/on";
const char* T_WINDOW_OFF  = "/smart_home/window/off";

const char* S_LIGHTS = "/smart_home/status/lights";
const char* S_AC     = "/smart_home/status/ac";
const char* S_DOOR   = "/smart_home/status/door";
const char* S_WINDOW = "/smart_home/status/window";
const char* S_ESP    = "/smart_home/status/esp8266";

WiFiClient wifiClient;
PubSubClient mqtt(wifiClient);

bool lightsOn = false;
bool acOn = false;
int windowDir = 0;
unsigned long lastStatus = 0;

void relayOn(int pin) { digitalWrite(pin, RELAY_ON); }
void relayOff(int pin) { digitalWrite(pin, RELAY_OFF); }

void relayPulse(int pin, int ms) {
  relayOn(pin);
  delay(ms);
  relayOff(pin);
}

void allRelaysOff() {
  relayOff(PIN_LIGHTS);
  relayOff(PIN_AC);
  relayOff(PIN_DOOR);
  relayOff(PIN_WINDOW_F);
  relayOff(PIN_WINDOW_R);
}

void ledBlink(int times = 1) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_LED, LOW);
    delay(80);
    digitalWrite(PIN_LED, HIGH);
    delay(80);
  }
}

void publishStatus() {
  if (!mqtt.connected()) return;

  mqtt.publish(S_LIGHTS, lightsOn ? "ON" : "OFF");
  mqtt.publish(S_AC, acOn ? "ON" : "OFF");
  mqtt.publish(S_DOOR, "READY");

  const char* w =
    (windowDir == 1) ? "GOING_UP" :
    (windowDir == -1) ? "GOING_DOWN" :
    "STOPPED";

  mqtt.publish(S_WINDOW, w);
  mqtt.publish(S_ESP, "ONLINE");
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  if (strstr(topic, "/status/")) return;

  String t = String(topic);

  if (t == T_LIGHTS_ON) {
    relayOn(PIN_LIGHTS);
    lightsOn = true;
    mqtt.publish(S_LIGHTS, "ON");
  }
  else if (t == T_LIGHTS_OFF) {
    relayOff(PIN_LIGHTS);
    lightsOn = false;
    mqtt.publish(S_LIGHTS, "OFF");
  }
  else if (t == T_AC_ON) {
    relayOn(PIN_AC);
    acOn = true;
    mqtt.publish(S_AC, "ON");
  }
  else if (t == T_AC_OFF) {
    relayOff(PIN_AC);
    acOn = false;
    mqtt.publish(S_AC, "OFF");
  }
  else if (t == T_DOOR_TOGGLE) {
    mqtt.publish(S_DOOR, "PULSING");
    relayPulse(PIN_DOOR, DOOR_PULSE_MS);
    delay(1500);
    mqtt.publish(S_DOOR, "READY");
  }
  else if (t == T_WINDOW_ON) {
    relayOff(PIN_WINDOW_R);
    delay(50);

    relayOn(PIN_WINDOW_F);
    windowDir = 1;
    mqtt.publish(S_WINDOW, "GOING_UP");

    delay(WINDOW_RUN_MS);

    relayOff(PIN_WINDOW_F);
    windowDir = 0;
    mqtt.publish(S_WINDOW, "STOPPED");
  }
  else if (t == T_WINDOW_OFF) {
    relayOff(PIN_WINDOW_F);
    delay(50);

    relayOn(PIN_WINDOW_R);
    windowDir = -1;
    mqtt.publish(S_WINDOW, "GOING_DOWN");

    delay(WINDOW_RUN_MS);

    relayOff(PIN_WINDOW_R);
    windowDir = 0;
    mqtt.publish(S_WINDOW, "STOPPED");
  }
}

bool ensureWiFi() {
  if (WiFi.status() == WL_CONNECTED) return true;

  static unsigned long lastAttempt = 0;
  static bool connecting = false;

  if (!connecting) {
    WiFi.mode(WIFI_STA);

    if (strlen(WIFI_PASSWORD) == 0)
      WiFi.begin(WIFI_SSID);
    else
      WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

    connecting = true;
    lastAttempt = millis();
  }

  digitalWrite(PIN_LED, !digitalRead(PIN_LED));

  if (millis() - lastAttempt > 20000) {
    WiFi.disconnect();
    connecting = false;
  }

  return false;
}

bool ensureMQTT() {
  if (mqtt.connected()) return true;

  static unsigned long lastAttempt = 0;
  if (millis() - lastAttempt < 5000) return false;

  lastAttempt = millis();

  if (mqtt.connect(MQTT_CLIENT, MQTT_USER, MQTT_PASSWORD)) {
    mqtt.subscribe("/smart_home/lights/#");
    mqtt.subscribe("/smart_home/ac/#");
    mqtt.subscribe("/smart_home/door/#");
    mqtt.subscribe("/smart_home/window/#");

    publishStatus();
    ledBlink(3);
    return true;
  }

  return false;
}

void setup() {
  Serial.begin(115200);

  digitalWrite(PIN_LIGHTS, RELAY_OFF);
  digitalWrite(PIN_AC, RELAY_OFF);
  digitalWrite(PIN_DOOR, RELAY_OFF);
  digitalWrite(PIN_WINDOW_F, RELAY_OFF);
  digitalWrite(PIN_WINDOW_R, RELAY_OFF);
  digitalWrite(PIN_LED, HIGH);

  pinMode(PIN_LIGHTS, OUTPUT);
  pinMode(PIN_AC, OUTPUT);
  pinMode(PIN_DOOR, OUTPUT);
  pinMode(PIN_WINDOW_F, OUTPUT);
  pinMode(PIN_WINDOW_R, OUTPUT);
  pinMode(PIN_LED, OUTPUT);

  allRelaysOff();

  WiFi.mode(WIFI_STA);

  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(mqttCallback);

  if (strlen(WIFI_PASSWORD) == 0)
    WiFi.begin(WIFI_SSID);
  else
    WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
}

void loop() {
  if (!ensureWiFi()) return;

  ensureMQTT();
  mqtt.loop();

  unsigned long now = millis();

  if (now - lastStatus > STATUS_INTERVAL_MS) {
    publishStatus();
    lastStatus = now;
  }
}