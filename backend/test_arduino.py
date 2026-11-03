from backend.arduino_controller import ArduinoController
import time

arduino = ArduinoController("COM7")

if arduino.connection and arduino.connection.is_open:
    arduino.send("ALERT")
    time.sleep(3)

    arduino.send("DROWSY")
    time.sleep(3)

    arduino.send("WARNING")
    time.sleep(3)

    arduino.close()