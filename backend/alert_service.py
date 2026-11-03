import threading


class AlertService:
    def __init__(self):
        self.stop_event = threading.Event()
        self.new_alert_event = threading.Event()
        self.thread = None
        self.lock = threading.Lock()

    def trigger_alert(self):
        """
        Start the voice assistant for the first alert.
        If it is already active, signal it about the new alert.
        """
        with self.lock:
            if self.thread is not None and self.thread.is_alive():
                self.new_alert_event.set()
                print(
                    "[ALERT] New alert sent to the active voice assistant.",
                    flush=True,
                )
                return

            self.stop_event.clear()
            self.new_alert_event.clear()

            from backend.voice_service import run_conversation

            self.thread = threading.Thread(
                target=self._run_voice,
                args=(run_conversation,),
                daemon=True,
                name="DriverVoiceAssistant",
            )
            self.thread.start()
            print("[ALERT] Voice assistant started.", flush=True)

    def _run_voice(self, run_conversation):
        try:
            run_conversation(
                self.new_alert_event,
                self.stop_event,
            )
        except Exception as exc:
            print(
                f"[ALERT ERROR] {type(exc).__name__}: {exc}",
                flush=True,
            )

    def stop(self):
        self.stop_event.set()
        print("[ALERT] Voice assistant stop requested.", flush=True)


alert_service = AlertService()
