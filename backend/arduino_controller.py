
import serial
import time


class ArduinoController:
    def __init__(self, port="COM7", baud_rate=9600):
        self.connection = None

        try:
            self.connection = serial.Serial(
                port=port,
                baudrate=baud_rate,
                timeout=1
            )

            # Arduino Uno resets briefly when serial opens.
            time.sleep(2)
            self.send("RESET")
            print(f"[ARDUINO] Connected on {port}")

        except serial.SerialException as exc:
            print(f"[ARDUINO ERROR] {exc}")

    def send(self, command):
        if self.connection and self.connection.is_open:
            try:
                self.connection.write(
                    (command + "\n").encode("utf-8")
                )
            except serial.SerialException as exc:
                print(f"[ARDUINO ERROR] {exc}")

    def close(self):
        if self.connection and self.connection.is_open:
            self.send("RESET")
            self.connection.close()
            print("[ARDUINO] Disconnected")
