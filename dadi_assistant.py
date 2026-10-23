"""
Dadi Sathi - Bhojpuri/Hindi voice assistant for grandparents.

Pipeline:  mic -> speech-to-text (Hindi model, understands Bhojpuri roughly)
           -> Claude (understands intent, replies in Bhojpuri)
           -> text-to-speech (Hindi voice reading Bhojpuri text)

Run:   python dadi_assistant.py
Use:   press SPACE (or click the big button) and speak.
       Type in the box + Enter if the mic misbehaves.   F11 = fullscreen.

Set your key first (optional - without it the app runs in OFFLINE keyword mode):
   Windows:  setx ANTHROPIC_API_KEY "sk-ant-..."   (then reopen terminal)
   Mac/Linux: export ANTHROPIC_API_KEY="sk-ant-..."
"""
import os
import queue
import tempfile
import threading
import time
import datetime as dt
import tkinter as tk

import numpy as np
import sounddevice as sd
import speech_recognition as sr
from gtts import gTTS
import pygame

from logic import extract_json, offline_intent, compute_due

MODEL = os.environ.get("DADI_MODEL", "claude-3-5-haiku-20241022")   # fast model = short pauses on stage
RATE = 16000
CONTACTS = ["बेटा", "बहू", "डॉक्टर"]
INBOX = ["बेटा राहुल का संदेश: पापा, हम शाम के छह बजे घरे आ जाएब।",
         "डॉक्टर साहब का संदेश: काल्ह सबेरे दस बजे जाँच खातिर आईं।"]

SYSTEM = """You are 'Dadi Sathi', a warm voice assistant for elderly Bhojpuri-speaking people in eastern Uttar Pradesh.
The user's speech was transcribed by a Hindi speech recogniser, so it may be a Hindi/Bhojpuri mix with small errors. Be forgiving.
Current date and time: {NOW}. Known contacts: {CONTACTS}.
Reply with ONE JSON object only, no markdown, no extra text:
{"intent": "set_reminder" | "call" | "read_messages" | "chat",
 "reminder_text": short string or null,
 "time": "HH:MM" in 24-hour format or null,
 "in_minutes": integer or null (use when the user says 'X minute baad'),
 "contact": one name from the contacts list or null,
 "reply": string}
Rules for "reply": write in Bhojpuri using Devanagari script, 1-2 short sentences, respectful (रउआ / राउर),
simple words, no English, no emojis. Examples:
 "ठीक बा, रात के नौ बजे दवाई के याद दिला देब।"
 "बेटा के फोन लगावत बानी, तनी रुकीं।"
 "राउर संदेश सुनावत बानी।"
 "हम ठीक बानी, रउआ कइसन बानी?"
If the request is unclear, ask one short clarifying question in Bhojpuri with intent "chat"."""

# Load .env file if present
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip("'\""))

# ---------- LLM clients (Groq primary, Claude secondary) ----------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")

groq_client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
    except Exception as e:
        print("Groq disabled:", e)

client = None
if os.environ.get("ANTHROPIC_API_KEY"):
    try:
        import anthropic
        client = anthropic.Anthropic()
    except Exception as e:   # package missing etc.
        print("Claude disabled:", e)

# ---------- audio ----------
pygame.mixer.init()
speak_lock = threading.Lock()
ui_q = queue.Queue()
reminders = []          # dicts: {"text", "due", "fired"}
busy = False


def ui(kind, value=""):
    ui_q.put((kind, value))


def speak(text):
    """Hindi voice reading Bhojpuri (Devanagari) text."""
    with speak_lock:
        path = os.path.join(tempfile.gettempdir(), f"dadi_{time.time_ns()}.mp3")
        try:
            gTTS(text=text, lang="hi").save(path)
            pygame.mixer.music.load(path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.05)
            pygame.mixer.music.unload()
        except Exception as e:
            ui("status", f"आवाज़ नहीं चली (internet?): {e}")
        finally:
            try:
                os.remove(path)
            except OSError:
                pass


def record(max_seconds=8):
    """Record until ~1s of silence after speech. Returns int16 numpy array or None."""
    block = int(RATE * 0.1)
    frames, started, silence = [], False, 0
    with sd.InputStream(samplerate=RATE, channels=1, dtype="int16", blocksize=block) as stream:
        ambient = np.mean([np.abs(stream.read(block)[0]).mean() for _ in range(3)])
        threshold = max(ambient * 2.5, 250)
        for _ in range(int(max_seconds * 10)):
            data, _ = stream.read(block)
            frames.append(data.copy())
            if np.abs(data).mean() > threshold:
                started, silence = True, 0
            elif started:
                silence += 1
                if silence >= 10:
                    break
            elif len(frames) > 50:          # 5 s of nothing
                return None
    return np.concatenate(frames) if started else None


def transcribe(audio):
    rec = sr.Recognizer()
    data = sr.AudioData(audio.tobytes(), RATE, 2)
    return rec.recognize_google(data, language="hi-IN")


# ---------- brain ----------
def think(text):
    if groq_client:
        try:
            system = (SYSTEM.replace("{NOW}", dt.datetime.now().strftime("%A %d %B %Y, %H:%M"))
                            .replace("{CONTACTS}", ", ".join(CONTACTS)))
            resp = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": text}
                ],
                temperature=0.2,
                max_tokens=250,
                response_format={"type": "json_object"}
            )
            return extract_json(resp.choices[0].message.content)
        except Exception as e:
            ui("status", f"Groq offline ({type(e).__name__}), fallback चल रहल बा")
    elif client:
        try:
            system = (SYSTEM.replace("{NOW}", dt.datetime.now().strftime("%A %d %B %Y, %H:%M"))
                            .replace("{CONTACTS}", ", ".join(CONTACTS)))
            msg = client.messages.create(model=MODEL, max_tokens=300, system=system,
                                         messages=[{"role": "user", "content": text}])
            return extract_json(msg.content[0].text)
        except Exception as e:
            ui("status", f"Claude नहीं मिला, offline mode चल रहल बा ({type(e).__name__})")
    return offline_intent(text)


def act(result):
    intent = result.get("intent", "chat")
    reply = result.get("reply") or "ठीक बा।"
    ui("reply", reply)

    if intent == "set_reminder":
        due = compute_due(result)
        if due:
            label = result.get("reminder_text") or "दवाई"
            reminders.append({"text": label, "due": due, "fired": False})
            ui("reminders")
    elif intent == "call":
        ui("action", f"📞  {result.get('contact') or 'बेटा'} को फोन लग रहा है...")
    speak(reply)
    if intent == "read_messages":
        for m in INBOX:
            ui("action", "✉️  " + m)
            speak(m)


def handle_text(text):
    global busy
    try:
        ui("heard", text)
        ui("status", "सोच रहल बानी...")
        act(think(text))
        ui("status", "तैयार - SPACE दबाईं आ बोलीं")
    finally:
        busy = False


def listen_flow():
    global busy
    try:
        ui("status", "🎙️ सुन रहल बानी... बोलीं")
        audio = record()
        if audio is None:
            ui("status", "कुछ सुनाई ना दिहल, फेर से कोशिश करीं")
            busy = False
            return
        ui("status", "समझ रहल बानी...")
        text = transcribe(audio)
    except sr.UnknownValueError:
        ui("status", "ठीक से समझ ना आइल, फेर से बोलीं")
        busy = False
        return
    except Exception as e:
        ui("status", f"गड़बड़: {type(e).__name__} - नीचे लिख के भेजीं")
        busy = False
        return
    handle_text(text)


def start_listening(_event=None):
    global busy
    if busy:
        return
    busy = True
    threading.Thread(target=listen_flow, daemon=True).start()


def start_typed(text):
    global busy
    if busy or not text.strip():
        return
    busy = True
    threading.Thread(target=handle_text, args=(text.strip(),), daemon=True).start()


def reminder_watcher():
    while True:
        now = dt.datetime.now()
        for r in reminders:
            if not r["fired"] and r["due"] <= now:
                r["fired"] = True
                ui("reminders")
                msg = f"दादी, {r['text']} के समय हो गइल बा। खा लीं।"
                ui("action", "⏰  " + msg)
                speak(msg)
        time.sleep(1)


# ---------- GUI ----------
BG, FG, ACCENT = "#101820", "#F2F2F2", "#FFB000"
root = tk.Tk()
root.title("Dadi Sathi - AI Bootcamp")
root.configure(bg=BG)
root.geometry("1100x700")
FONT = "Nirmala UI" if os.name == "nt" else "Noto Sans Devanagari"

mode = "ONLINE: Groq (Ultra-Fast)" if groq_client else ("ONLINE: Claude" if client else "OFFLINE: keyword mode")
tk.Label(root, text=f"दादी साथी   ({mode})", font=(FONT, 26, "bold"), bg=BG, fg=ACCENT).pack(pady=10)
status = tk.Label(root, text="तैयार - SPACE दबाईं आ बोलीं", font=(FONT, 20), bg=BG, fg="#9AD1FF")
status.pack(pady=4)
heard = tk.Label(root, text="", font=(FONT, 28), bg=BG, fg=FG, wraplength=1000, justify="center")
heard.pack(pady=14)
reply = tk.Label(root, text="", font=(FONT, 32, "bold"), bg=BG, fg="#7CFFB2", wraplength=1000, justify="center")
reply.pack(pady=14)
action = tk.Label(root, text="", font=(FONT, 22), bg=BG, fg="#FFD166", wraplength=1000, justify="center")
action.pack(pady=8)
rem_label = tk.Label(root, text="", font=(FONT, 20), bg=BG, fg=FG, justify="center")
rem_label.pack(pady=8)
tk.Button(root, text="🎙️  बोलीं  (SPACE)", font=(FONT, 24, "bold"), bg=ACCENT, fg="black",
          command=start_listening).pack(pady=10, ipadx=20, ipady=6)
entry = tk.Entry(root, font=(FONT, 16), width=60)
entry.pack(pady=4)
tk.Label(root, text="backup: type here + Enter", font=(FONT, 11), bg=BG, fg="#666").pack()


def on_enter(_e):
    start_typed(entry.get())
    entry.delete(0, "end")


def on_space(e):
    if root.focus_get() is not entry:
        start_listening()


def toggle_fs(_e=None):
    root.attributes("-fullscreen", not root.attributes("-fullscreen"))


root.bind("<space>", on_space)
root.bind("<F11>", toggle_fs)
entry.bind("<Return>", on_enter)


def poll_ui():
    try:
        while True:
            kind, val = ui_q.get_nowait()
            if kind == "status":
                status.config(text=val)
            elif kind == "heard":
                heard.config(text="🗣️  " + val)
                reply.config(text="")
                action.config(text="")
            elif kind == "reply":
                reply.config(text=val)
            elif kind == "action":
                action.config(text=val)
            elif kind == "reminders":
                lines = [f"{'✅' if r['fired'] else '⏰'} {r['text']} - {r['due'].strftime('%I:%M %p')}"
                         for r in reminders[-3:]]
                rem_label.config(text="\n".join(lines))
    except queue.Empty:
        pass
    root.after(100, poll_ui)


if __name__ == "__main__":
    threading.Thread(target=reminder_watcher, daemon=True).start()
    poll_ui()
    root.mainloop()
