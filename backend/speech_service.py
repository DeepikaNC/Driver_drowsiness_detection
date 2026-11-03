
import threading
import win32com.client


class SpeechService:
    def __init__(self, rate=0, volume=100):
        self.rate = rate
        self.volume = volume
        self.lock = threading.Lock()
        self.speaker = win32com.client.Dispatch("SAPI.SpVoice")
        self.speaker.Rate = rate
        self.speaker.Volume = volume

    def speak(self, text):
        """Speak a complete response before returning."""
        if not text or not text.strip():
            return

        with self.lock:
            try:
                self.speaker.Speak(text.strip())
            except Exception as exc:
                print(f"Speech error: {exc}")
                raise


if __name__ == "__main__":
    speaker = SpeechService()

    speaker.speak("Hello. I am your driver safety assistant.")
    speaker.speak("I can now speak multiple responses.")
    speaker.speak("How are you feeling right now?")

    print("Speech test completed.")
