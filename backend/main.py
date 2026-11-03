import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend.camera_service import camera_service
from backend.voice_state import get_voice_status
from backend.arduino_controller import ArduinoController


app = FastAPI(
    title="Driver Drowsiness Detection API",
    version="1.0.0",
)

# One Arduino controller is shared by the API and camera service.
arduino_controller = ArduinoController(port="COM7", baud_rate=9600)
camera_service.arduino_controller = arduino_controller


@app.on_event("startup")
def initialize_hardware():
    # Clear a previously latched alert when the backend starts.
    connected = arduino_controller.connect()
    if connected:
        arduino_controller.send("CLEAR")
        print("[STARTUP] Arduino connected; alert output cleared.", flush=True)
    else:
        print(
            "[STARTUP] Arduino unavailable. Camera can still run.",
            flush=True,
        )


@app.on_event("shutdown")
def shutdown_hardware():
    try:
        camera_service.stop()
    except Exception as exc:
        print(f"[SHUTDOWN] Camera stop error: {exc}", flush=True)
    arduino_controller.close()


@app.post("/api/arduino/connect")
def connect_arduino():
    connected = arduino_controller.connect()
    if connected:
        arduino_controller.send("CLEAR")
    return {
        "connected": connected,
        **arduino_controller.get_status(),
    }


@app.get("/api/arduino/status")
def arduino_status():
    return arduino_controller.get_status()


@app.post("/api/arduino/test-alert")
def arduino_test_alert():
    if not arduino_controller.get_status()["connected"]:
        arduino_controller.connect()
    return {"success": arduino_controller.send("ALERT")}


@app.post("/api/arduino/test-clear")
def arduino_test_clear():
    if not arduino_controller.get_status()["connected"]:
        arduino_controller.connect()
    return {"success": arduino_controller.send("CLEAR")}


@app.get("/api/voice/status")
def voice_status():
    return get_voice_status()


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Driver Drowsiness Detection API is running"}


@app.get("/api/health")
def health():
    return {
        "status": "online",
        "project": "Driver Drowsiness Detection",
        "model": "Random Forest",
        "camera": camera_service.get_status()["camera"],
        "arduino": arduino_controller.get_status(),
    }


@app.post("/api/camera/start")
def start_camera():
    try:
        # Ensure the LED is OFF before detection begins.
        if not arduino_controller.get_status()["connected"]:
            arduino_controller.connect()

        if arduino_controller.get_status()["connected"]:
            arduino_controller.send("CLEAR")

        result = camera_service.start()
        return {
            **result,
            "arduino": arduino_controller.get_status(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/camera/stop")
def stop_camera():
    result = camera_service.stop()
    if arduino_controller.get_status()["connected"]:
        arduino_controller.send("CLEAR")
    return {
        **result,
        "arduino": arduino_controller.get_status(),
    }


@app.get("/api/camera/status")
def camera_status():
    return camera_service.get_status()


def generate_frames():
    while True:
        frame = camera_service.get_frame()

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )
        elif not camera_service.running:
            break

        time.sleep(0.04)


@app.get("/api/camera/video")
def camera_video():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
