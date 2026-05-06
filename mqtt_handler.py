"""
mqtt_handler.py — MQTT publisher with auto-reconnect
=====================================================
Wraps paho-mqtt so the rest of the program never crashes
if the broker is offline or connection drops mid-session.

Supports:
  - Local Mosquitto (no auth) - set MQTT_USER = "" in config.py
  - Shiftr.io cloud broker (with auth) - set MQTT_USER and MQTT_PASSWORD
"""

import paho.mqtt.client as mqtt
import time
import threading
import config


class MQTTHandler:
    """
    Thread-safe MQTT publisher.
    - Connects in background — program starts immediately without waiting
    - Auto-reconnects if broker goes offline
    - publish() silently drops messages if not connected (no crash)
    - All connection errors are printed but never raised
    - Supports Shiftr.io authentication when credentials provided
    """

    def __init__(self):
        self._client      = mqtt.Client()
        self._connected   = False
        self._lock        = threading.Lock()

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

        # ===== ADDED: Shiftr.io authentication support =====
        # Check if username/password are configured in config.py
        if hasattr(config, 'MQTT_USER') and config.MQTT_USER:
            self._client.username_pw_set(config.MQTT_USER, config.MQTT_PASSWORD)
            print(f"[MQTT] Using Shiftr.io authentication for user: {config.MQTT_USER}")
        else:
            print("[MQTT] No credentials - using local broker (no authentication)")
        # ===== END OF ADDED SECTION =====

        # Start background connection thread
        t = threading.Thread(target=self._connect_loop, daemon=True)
        t.start()

    # ── Internal callbacks ─────────────────────────────────────────────
    def _on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            self._connected = True
            print(f"[MQTT] Connected to {config.MQTT_BROKER}:{config.MQTT_PORT}")
        else:
            self._connected = False
            # ===== IMPROVED: Better error messages =====
            error_codes = {
                1: "wrong protocol version",
                2: "invalid client ID",
                3: "broker unavailable",
                4: "wrong username/password (check MQTT_USER/MQTT_PASSWORD)",
                5: "not authorized"
            }
            error_msg = error_codes.get(rc, f"error code {rc}")
            print(f"[MQTT] Connection failed: {error_msg}")
            # ===== END OF IMPROVED SECTION =====

    def _on_disconnect(self, client, userdata, rc):
        self._connected = False
        if rc != 0:
            print(f"[MQTT] Unexpected disconnect (rc={rc}) — will retry")

    def _connect_loop(self):
        """Keeps trying to connect/reconnect in background forever."""
        self._client.loop_start()
        while True:
            if not self._connected:
                try:
                    # ===== FIXED: Use reconnect_delay_set for better reconnection =====
                    self._client.reconnect_delay_set(min_delay=1, max_delay=30)
                    self._client.connect(config.MQTT_BROKER,
                                         config.MQTT_PORT, 60)
                except Exception as e:
                    print(f"[MQTT] Cannot reach broker: {e} — retry in "
                          f"{config.MQTT_RECONNECT_DELAY}s")
            time.sleep(config.MQTT_RECONNECT_DELAY)

    # ── Public API ─────────────────────────────────────────────────────
    def publish(self, device: str, action: str):
        """
        Publish a device command.
        Topic format: /smart_home/<device>/<action>
        Payload     : <action>
        """
        if not self._connected:
            print(f"[MQTT] Not connected — dropped: {device}/{action}")
            return False

        topic = f"{config.MQTT_TOPIC_BASE}{device}/{action}"
        with self._lock:
            # ===== IMPROVED: QoS 1 for guaranteed delivery =====
            result = self._client.publish(topic, action, qos=1)
            # ===== END OF IMPROVED SECTION =====

        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            print(f"[MQTT] ✓  {topic}")
            return True
        else:
            print(f"[MQTT] ✗  publish failed (rc={result.rc})")
            return False

    def is_connected(self) -> bool:
        return self._connected

    def stop(self):
        self._client.loop_stop()
        self._client.disconnect()
        print("[MQTT] Disconnected.")
