
import serial
import time
import threading


class ArduinoController:

    def __init__(self, port="COM7", baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.connection = None
        self.lock = threading.Lock()
        self.alert_active = False
        self.status = "Disconnected"

    def connect(self):
        with self.lock:

            if self.connection and self.connection.is_open:
                return True

            try:
                self.connection = serial.Serial(
                    port=self.port,
                    baudrate=self.baud_rate,
                    timeout=1
                )

                # Arduino Uno resets when serial opens
                time.sleep(2)

                self.status = "Connected"

                # Start with normal driver status
                self.connection.write(b"CLEAR\n")
                self.connection.flush()

                self.alert_active = False

                print(
                    f"[ARDUINO] Connected on {self.port}",
                    flush=True
                )

                return True

            except (serial.SerialException, OSError) as exc:
                self.connection = None
                self.status = "Disconnected"

                print(
                    f"[ARDUINO ERROR] {exc}",
                    flush=True
                )

                return False

    def send(self, command):

        if command not in ("ALERT", "NORMAL", "CLEAR"):
            return False

        with self.lock:

            if not self.connection or not self.connection.is_open:
                self.status = "Disconnected"
                return False

            try:
                self.connection.write(
                    (command + "\n").encode("utf-8")
                )

                self.connection.flush()

                self.alert_active = command == "ALERT"

                print(
                    f"[ARDUINO] Command sent: {command}",
                    flush=True
                )

                return True

            except (serial.SerialException, OSError) as exc:
                self.status = "Disconnected"
                print(f"[ARDUINO ERROR] {exc}", flush=True)
                return False

    def get_status(self):

        connected = (
            self.connection is not None
            and self.connection.is_open
        )

        return {
            "connected": connected,
            "status": "Connected" if connected else "Disconnected",
            "alert_active": self.alert_active if connected else False,
            "port": self.port
        }

    def close(self):

        with self.lock:

            if self.connection and self.connection.is_open:

                try:
                    self.connection.write(b"CLEAR\n")
                    self.connection.flush()
                    self.connection.close()

                except (serial.SerialException, OSError) as exc:
                    print(f"[ARDUINO ERROR] {exc}", flush=True)

            self.connection = None
            self.status = "Disconnected"
            self.alert_active = False

            print("[ARDUINO] Disconnected", flush=True)
