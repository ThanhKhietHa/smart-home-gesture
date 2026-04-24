"""
mqtt_handler.py — MQTT publisher with auto-reconnect
=====================================================
Wraps paho-mqtt so the rest of the program never crashes
if the broker is offline or connection drops mid-session.
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
    """

    def __init__(self):
        self._client      = mqtt.Client()
        self._connected   = False
        self._lock        = threading.Lock()

        self._client.on_connect    = self._on_connect
        self._client.on_disconnect = self._on_disconnect

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
            print(f"[MQTT] Connection failed (rc={rc})")

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
            result = self._client.publish(topic, action)

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
