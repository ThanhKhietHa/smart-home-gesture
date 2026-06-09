import paho.mqtt.client as mqtt
import time
import threading
import config


class MQTTHandler:
    def __init__(self):
        self._client    = mqtt.Client()
        self._connected = False
        self._lock      = threading.Lock()

        # Device states — always driven by ESP status topics, never assumed
        self.state = {
            "lights": None,   # "ON" | "OFF"
            "ac":     None,   # "ON" | "OFF"
            "door":   None,   # "READY" | "PULSING"
            "window": None,   # "STOPPED" | "GOING_UP" | "GOING_DOWN"
            "esp":    None,   # "ONLINE"
        }

        self.on_state_change = None

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message    = self._on_message

        user = getattr(config, 'MQTT_USER',     '')
        pwd  = getattr(config, 'MQTT_PASSWORD', '')
        if user:
            self._client.username_pw_set(user, pwd)
            print(f"[MQTT] Using credentials for: {user}")
        else:
            print("[MQTT] No credentials — local broker mode")

        self._client.reconnect_delay_set(min_delay=1, max_delay=10)

        t = threading.Thread(target=self._connect_loop, daemon=True)
        t.start()

    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {config.MQTT_BROKER}:{config.MQTT_PORT}")
            client.subscribe(f"{config.MQTT_TOPIC_BASE}status/#", qos=1)
            print("[MQTT] Subscribed to status topics")
        else:
            self._connected = False
            codes = {
                1: "wrong protocol version",
                2: "invalid client ID",
                3: "broker unavailable",
                4: "wrong username/password — check MQTT_USER/MQTT_PASSWORD",
                5: "not authorized",
            }
            print(f"[MQTT] Connection failed: {codes.get(rc, f'rc={rc}')}")

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print("[MQTT] Unexpected disconnect — will retry")

    def _on_message(self, client, userdata, msg):
        """
        Handles /smart_home/status/<device> payloads published by the ESP.
        Updates self.state so the Python side always mirrors hardware reality.
        """
        topic   = msg.topic
        payload = msg.payload.decode().strip()

        # topic example: /smart_home/status/lights
        base   = f"{config.MQTT_TOPIC_BASE}status/"
        if not topic.startswith(base):
            return

        device = topic[len(base):]  
        
        key = "esp" if device == "esp8266" else device

        if key not in self.state:
            return

        old = self.state[key]
        self.state[key] = payload

        if old != payload:
            print(f"[STATUS] {key}: {old} → {payload}")
            if callable(self.on_state_change):
                self.on_state_change(key, payload)

    def _connect_loop(self):
        """
        Tries to connect once; paho's loop_start() handles reconnection.
        """
        self._client.loop_start()
        while not self._connected:
            try:
                self._client.connect(config.MQTT_BROKER, config.MQTT_PORT, 60)
                time.sleep(1.5)
                if self._connected:
                    break
            except Exception as e:
                print(f"[MQTT] Cannot reach broker: {e} "
                      f"— retry in {config.MQTT_RECONNECT_DELAY}s")
                time.sleep(config.MQTT_RECONNECT_DELAY)

    def wait_for_sync(self, timeout=5.0):
        """
        On app restart, ESP is already running and publishing status every 15s.
        Wait up to `timeout` seconds for all states to arrive.
        If timeout — default everything to OFF/READY (safe state).
        """
        print("[MQTT] Syncing with ESP state...")
        deadline = time.time() + timeout

        # Wait for connection first before expecting status messages
        while time.time() < deadline:
            if self._connected:
                break
            time.sleep(0.1)
        else:
            print("[MQTT] Sync failed — broker not reachable")
            self._apply_defaults()
            return False

        # Now wait for ESP status topics
        while time.time() < deadline:
            if all(self.state[k] is not None
                   for k in ("lights", "ac", "door", "window")):
                print(f"[MQTT] Sync OK → {self.state}")
                return True
            time.sleep(0.1)

        self._apply_defaults()
        return False

    def _apply_defaults(self):
        defaults = {"lights": "OFF", "ac": "OFF", "door": "READY", "window": "STOPPED"}
        for k, v in defaults.items():
            if self.state[k] is None:
                self.state[k] = v
        print(f"[MQTT] Sync timeout — using defaults: {self.state}")

    def publish(self, device: str, action: str):
        """
        Sends a command topic to the ESP, e.g. publish("lights", "on").

        The payload is intentionally empty ("") because the ESP only cares
        about the topic name, not the payload content.  State is NOT updated
        here — it is updated only when the ESP confirms via a status message.
        """
        if not self._connected:
            print(f"[MQTT] Not connected — dropped: {device}/{action}")
            return False

        topic = f"{config.MQTT_TOPIC_BASE}{device}/{action}"
        with self._lock:
            # Empty payload — the ESP acts on topic name alone
            result = self._client.publish(topic, "", qos=1)

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] ✓  {topic}")
            return True

        print(f"[MQTT] ✗  publish failed (rc={result.rc})")
        return False

    def get_state(self, device: str):
        """
        Returns the last known state for a device, or None if not yet
        received from the ESP.  Never returns a locally-assumed value.
        """
        return self.state.get(device)

    def is_connected(self) -> bool:
        return self._connected

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
        print("[MQTT] Disconnected.")
