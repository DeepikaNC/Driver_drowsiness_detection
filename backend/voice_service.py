import os
import threading

from dotenv import load_dotenv
from google import genai
from google.genai import types

from backend.speech_service import SpeechService
from backend.microphone_service import MicrophoneService


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found in the project .env file."
    )

# Limit how long the application waits for an API request.
client = genai.Client(
    api_key=API_KEY,
    http_options=types.HttpOptions(timeout=20000),
)

speaker = SpeechService()
microphone = MicrophoneService()


# --------------------------------------------------
# SYSTEM INSTRUCTIONS
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a friendly, calm driver-assistance voice companion.

Safety rules:
- If drowsiness is detected, warn the driver to pull over safely
  and rest.
- Ask how the driver feels, using one short question at a time.
- If the driver feels sleepy, advise them not to continue driving.
- Never encourage a sleepy driver to continue driving.
- If the driver says they feel alert, acknowledge this politely.
- Do not repeatedly ask whether the driver is sleepy.
- Do not claim that conversation, coffee, or music makes
  drowsy driving safe.
- If another alert occurs, prioritize safety and recommend
  stopping somewhere safe.
- Do not claim to monitor the camera yourself.
- Keep responses brief, calm, and natural when spoken aloud.
"""


# --------------------------------------------------
# GEMINI RESPONSE GENERATION
# --------------------------------------------------

def generate_reply(user_message, previous_interaction_id=None):
    """Send the driver's response to Gemini and return its reply."""

    request = {
        "model": MODEL,
        "input": user_message,
        "system_instruction": SYSTEM_PROMPT,
    }

    if previous_interaction_id:
        request["previous_interaction_id"] = previous_interaction_id

    print(
        f"[VOICE] Sending request to Gemini ({MODEL})...",
        flush=True,
    )

    interaction = client.interactions.create(**request)

    reply = interaction.output_text

    if not reply or not reply.strip():
        raise RuntimeError("Gemini returned an empty response.")

    print("[VOICE] Gemini request completed.", flush=True)

    return reply.strip(), interaction.id


def start_gemini_reply(message, previous_id):
    """Generate a response and provide a safe fallback on failure."""

    try:
        reply, interaction_id = generate_reply(
            message,
            previous_id,
        )

        print("[VOICE] Gemini reply received.", flush=True)

        return reply, interaction_id

    except Exception as exc:
        print(
            f"[VOICE GEMINI ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )

        # Keep the conversation usable after an API failure.
        # Retain the previous valid interaction ID.
        fallback = (
            "I'm having trouble responding right now. "
            "If you feel sleepy, please pull over safely and rest."
        )

        return fallback, previous_id


# --------------------------------------------------
# SPEECH OUTPUT
# --------------------------------------------------

def speak_and_log(message):
    """Print a message and attempt to speak it."""

    print(f"\n[VOICE] Assistant: {message}", flush=True)

    try:
        speaker.speak(message)
        print("[VOICE] Speech playback completed.", flush=True)

    except Exception as exc:
        print(
            f"[VOICE SPEECH ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )


def announce_alert():
    """Announce a drowsiness warning."""

    message = (
        "Warning. The system is detecting signs of drowsiness. "
        "Please pull over somewhere safe and rest. "
        "Do not continue driving while sleepy."
    )

    speak_and_log(message)


# --------------------------------------------------
# VOICE CONVERSATION
# --------------------------------------------------

def run_conversation(new_alert_event, stop_event):
    """
    Run the voice companion until shutdown.

    new_alert_event:
        Set when the drowsiness detector raises a new alert.

    stop_event:
        Set when the application should stop.
    """

    previous_id = None

    print("[VOICE] Conversation function entered.", flush=True)

    try:
        # Initial warning and question.
        announce_alert()

        if stop_event.is_set():
            return

        speak_and_log(
            "How are you feeling right now? "
            "Do you feel sleepy, or are you feeling alert?"
        )

        while not stop_event.is_set():

            # Handle an alert before listening.
            if new_alert_event.is_set():
                new_alert_event.clear()
                announce_alert()

                if stop_event.is_set():
                    break

            print(
                "[VOICE] Waiting for microphone input...",
                flush=True,
            )

            user_message = microphone.listen()

            if stop_event.is_set():
                break

            # Handle an alert that occurred while listening.
            if new_alert_event.is_set():
                new_alert_event.clear()
                announce_alert()

            if not user_message:
                print(
                    "[VOICE] No usable response received. "
                    "Listening again.",
                    flush=True,
                )
                continue

            user_message = user_message.strip()

            if not user_message:
                continue

            print(
                f"[VOICE] Processing driver response: {user_message}",
                flush=True,
            )

            # Allow the user to end the conversation.
            if user_message.lower() in {
                "exit",
                "quit",
                "stop conversation",
            }:
                speak_and_log("Okay. Ending our conversation.")
                break

            prompt = (
                f"The driver responded: {user_message}\n"
                "Respond briefly and naturally. Ask at most one "
                "relevant follow-up question when appropriate. "
                "Prioritize driver safety. If the driver feels "
                "sleepy, tell them to pull over safely and rest. "
                "Do not suggest that continuing to drive is safe."
            )

            print(
                "[VOICE] Sending recognized response for processing...",
                flush=True,
            )

            reply, previous_id = start_gemini_reply(
                prompt,
                previous_id,
            )

            print(
                f"[VOICE] Reply ready: {reply!r}",
                flush=True,
            )

            if stop_event.is_set():
                break

            speak_and_log(reply)

            # Handle alerts raised while Gemini was responding.
            if new_alert_event.is_set():
                new_alert_event.clear()
                announce_alert()

    except KeyboardInterrupt:
        stop_event.set()
        print("\n[VOICE] Conversation interrupted.", flush=True)

    except Exception as exc:
        print(
            f"[VOICE ERROR] {type(exc).__name__}: {exc}",
            flush=True,
        )

    finally:
        print(
            "[VOICE] Conversation thread finished.",
            flush=True,
        )


# --------------------------------------------------
# STANDALONE TEST
# --------------------------------------------------

if __name__ == "__main__":

    alert_event = threading.Event()
    stop_event = threading.Event()

    try:
        run_conversation(alert_event, stop_event)

    except KeyboardInterrupt:
        stop_event.set()
        print(
            "\n[VOICE] Standalone test stopped.",
            flush=True,
        )
