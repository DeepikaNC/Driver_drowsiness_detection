
import threading

_lock = threading.Lock()

_state = {
    "status": "Idle",
    "message": "Voice assistant is waiting",
    "active": False,
}


def set_voice_status(status, message=None, active=None):
    with _lock:
        _state["status"] = status

        if message is not None:
            _state["message"] = message

        if active is not None:
            _state["active"] = active


def get_voice_status():
    with _lock:
        return _state.copy()
