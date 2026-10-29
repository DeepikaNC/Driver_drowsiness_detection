
import speech_recognition as sr


class MicrophoneService:
    def __init__(self, timeout=20, phrase_time_limit=12):
        self.recognizer = sr.Recognizer()
        self.timeout = timeout
        self.phrase_time_limit = phrase_time_limit
        self.calibrated = False

        # Give the driver more time to finish speaking.
        self.recognizer.pause_threshold = 1.5
        self.recognizer.non_speaking_duration = 0.5
        self.recognizer.dynamic_energy_threshold = True

    def listen(self):
        """Listen for a complete response and return recognized text."""
        try:
            with sr.Microphone() as source:
                if not self.calibrated:
                    print("[MIC] Calibrating microphone...", flush=True)
                    self.recognizer.adjust_for_ambient_noise(
                        source, duration=1.0
                    )
                    self.calibrated = True

                print(
                    "[MIC] You have time to respond. Speak now.",
                    flush=True,
                )

                audio = self.recognizer.listen(
                    source,
                    timeout=self.timeout,
                    phrase_time_limit=self.phrase_time_limit,
                )

            print("[MIC] Recognizing speech...", flush=True)
            text = self.recognizer.recognize_google(audio).strip()
            print(f"[MIC] Recognized: {text}", flush=True)
            return text or None

        except sr.WaitTimeoutError:
            print("[MIC] No speech detected. Listening again.", flush=True)
            return None

        except sr.UnknownValueError:
            print("[MIC] Speech was not understood. Try again.", flush=True)
            return None

        except sr.RequestError as exc:
            print(f"[MIC ERROR] Speech recognition failed: {exc}", flush=True)
            return None

        except Exception as exc:
            print(
                f"[MIC ERROR] {type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
