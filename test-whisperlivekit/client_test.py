"""
WhisperLiveKit test client — microphone → real-time transcription with speaker labels.

Usage:
    source .venv/bin/activate
    python client_test.py

Stop with Ctrl+C.

API used: whisper_live.client.Client / TranscriptionClient / TranscriptionTeeClient
  - transcription_callback(text, segments) fires on every server update and
    returns early, bypassing the built-in clear-screen log_transcription path.
  - Segment dict keys: text, start, end, completed (bool), speaker (str|None).
  - Language detection: the server sends {"uid":..., "language":"fr",
    "language_prob":0.92} messages; Client.on_message stores the last detected
    language in self.language but discards the probability.
  - hf_token is not forwarded in the WS handshake; pyannote falls back to the
    cached huggingface-cli login token.

Two-pass language strategy
  Phase 1 (LANG_DETECT_WINDOW seconds): lang=None, no diarization.
    A Timer stops the recording loop by setting client.recording=False, which
    causes TranscriptionTeeClient.record() to break out and return cleanly.
    A _LangDetectClient subclass intercepts language messages and tracks the
    highest probability observed per language code.
  Phase 2 (until Ctrl+C): lang=<best detected>, diarization on, max_speakers
    capped to MAX_SPEAKERS.
"""

import json
import os
import sys
import threading

# ── Tunable constants ─────────────────────────────────────────────────────────

# Set dynamically based on number of participants in Radical integration.
MAX_SPEAKERS = 2

# How long (seconds) to run with lang=None before locking to the detected
# language.  Longer gives more signal; 10s is a good default for most meetings.
LANG_DETECT_WINDOW = 10.0

# ── Terminal helpers ──────────────────────────────────────────────────────────

_CLEAR_LINE = "\r" + " " * 100 + "\r"


def _format_ts(secs: float) -> str:
    m, s = divmod(int(secs), 60)
    return f"{m:02d}:{s:02d}"


# ── Output state ──────────────────────────────────────────────────────────────

# Start times of completed segments we have already printed, to avoid reprints
# when the server re-sends the same segments in later updates.
_printed: set[float] = set()

# Whether the current terminal line holds an interim preview that must be
# erased before the next completed segment is printed.
_interim_on_line = False


def _on_segment(text: str, segments: list) -> None:
    """
    Callback invoked by WhisperLive on every segments update.
    Prints completed segments once (deduplicated by start time), with speaker
    label and timestamp.  Shows the last non-completed segment as a live
    inline preview overwriting the same terminal line.
    """
    global _interim_on_line

    completed_new = []
    last_interim = None

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        if not seg_text:
            continue
        if seg.get("completed", False):
            start = float(seg.get("start", 0))
            if start not in _printed:
                completed_new.append(seg)
        else:
            last_interim = seg

    for seg in completed_new:
        start = float(seg.get("start", 0))
        _printed.add(start)

        if _interim_on_line:
            print(_CLEAR_LINE, end="")
            _interim_on_line = False

        end = float(seg.get("end", start))
        ts = f"[{_format_ts(start)} → {_format_ts(end)}]"
        speaker = seg.get("speaker")
        spk = f"  {speaker}" if speaker else ""
        print(f"{ts}{spk}  {seg.get('text', '').strip()}")

    if last_interim:
        preview = last_interim.get("text", "").strip()
        if preview:
            print(f"\r⟳  {preview}", end="", flush=True)
            _interim_on_line = True
    elif not completed_new and _interim_on_line:
        print(_CLEAR_LINE, end="", flush=True)
        _interim_on_line = False


# ── HuggingFace token ─────────────────────────────────────────────────────────

def _hf_token() -> str | None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import HfFolder
            token = HfFolder.get_token()
        except ImportError:
            pass
    return token


# ── Phase-1 client subclass ───────────────────────────────────────────────────

class _LangDetectClient:
    """
    Thin wrapper around whisper_live.client.Client that intercepts
    language-detection server messages to track the highest-confidence
    language seen during the phase-1 window.

    The server sends a dedicated message type:
        {"uid": "...", "language": "fr", "language_prob": 0.92}
    Client.on_message() stores only the *last* detected language.  This
    subclass stores the *best* (highest probability) per language code so
    that a brief noisy mis-detection at the start does not win out.
    """

    def __new__(cls, **kwargs):
        # Lazy import so a missing package gives a clear error at call time.
        from whisper_live.client import Client

        class _Inner(Client):
            def __init__(inner_self, **kw):
                inner_self._lang_votes: dict[str, float] = {}
                super().__init__(**kw)

            def on_message(inner_self, ws, raw: str) -> None:
                try:
                    data = json.loads(raw)
                    # Language detection messages carry "language" + "language_prob"
                    # but NOT a "status" key (those are WAIT/ERROR/WARNING messages).
                    if "language" in data and "status" not in data:
                        lang = data.get("language")
                        prob = float(data.get("language_prob", 0.0))
                        if lang and prob > inner_self._lang_votes.get(lang, 0.0):
                            inner_self._lang_votes[lang] = prob
                except Exception:
                    pass
                super().on_message(ws, raw)

            @property
            def best_language(inner_self) -> str | None:
                if not inner_self._lang_votes:
                    return None
                return max(inner_self._lang_votes, key=lambda k: inner_self._lang_votes[k])

        return _Inner(**kwargs)


# ── Phase cleanup helper ──────────────────────────────────────────────────────

def _teardown_phase1(tee, inner_client) -> None:
    """
    Close the WebSocket and release the PyAudio stream after phase 1 ends.
    Must be called regardless of whether the phase ended via timer or Ctrl+C
    so that the microphone device is free for the phase-2 client.
    """
    try:
        inner_client.close_websocket()
    except Exception:
        pass
    try:
        if getattr(tee, "stream", None):
            tee.stream.stop_stream()
            tee.stream.close()
        tee.p.terminate()
    except Exception:
        pass


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    token = _hf_token()
    if not token:
        print(
            "⚠️   No HuggingFace token found — diarization may fail if the\n"
            "    speaker embedding model is not cached locally.\n"
            "    Run: huggingface-cli login\n",
            file=sys.stderr,
        )
    else:
        print("✅  HuggingFace token found (used by pyannote from cache)")

    try:
        from whisper_live.client import TranscriptionClient, TranscriptionTeeClient
    except ImportError:
        print(
            "❌  whisper-live not found. Install dependencies:\n"
            "    pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    # ── Phase 1: language detection window ───────────────────────────────────

    print(f"\nPhase 1 — detecting language ({LANG_DETECT_WINDOW:.0f}s, no diarization)...")

    p1_inner = _LangDetectClient(
        host="localhost",
        port=9090,
        lang=None,              # auto-detect
        model="medium",
        use_vad=True,
        # Raise min_speech_duration_ms (default 250 ms) so that brief noise or
        # silence at microphone open doesn't satisfy the VAD and trigger a
        # spurious language-detection call before real speech begins.
        vad_parameters={"min_speech_duration_ms": 600},
        log_transcription=False,
        transcription_callback=_on_segment,
        enable_diarization=False,   # skip in phase 1 — model load adds latency
    )
    p1 = TranscriptionTeeClient([p1_inner])

    # A daemon Timer sets recording=False after LANG_DETECT_WINDOW seconds.
    # TranscriptionTeeClient.record() checks `any(c.recording for c in self.clients)`
    # at each chunk (~0.26 s), so the loop exits within one read cycle.
    stop_timer = threading.Timer(
        LANG_DETECT_WINDOW,
        lambda: setattr(p1_inner, "recording", False),
    )
    stop_timer.daemon = True
    stop_timer.start()

    try:
        p1()                    # blocks until recording=False or Ctrl+C
    except KeyboardInterrupt:
        stop_timer.cancel()
        _teardown_phase1(p1, p1_inner)
        if _interim_on_line:
            print()
        print("\n✅  Stopped.")
        return

    stop_timer.cancel()         # in case p1() returned before the timer fired
    _teardown_phase1(p1, p1_inner)

    # Pick the language with the highest single-observation probability
    detected = p1_inner.best_language
    if detected:
        best_prob = p1_inner._lang_votes[detected]
        all_votes = ", ".join(
            f"{lang}={prob:.0%}" for lang, prob in sorted(
                p1_inner._lang_votes.items(), key=lambda kv: -kv[1]
            )
        )
        print(f"\n🔒  Language locked to: {detected} ({best_prob:.0%} — all: {all_votes})")
    else:
        print("\n⚠️   No language detected in phase 1, continuing with auto-detect")

    # ── Phase 2: full transcription with locked language + diarization ────────

    print(
        f"Phase 2 — transcribing "
        f"(lang={detected or 'auto'}, max_speakers={MAX_SPEAKERS})...\n"
    )

    client = TranscriptionClient(
        host="localhost",
        port=9090,
        model="medium",
        lang=detected,          # locked language from phase 1 (or None)
        use_vad=True,
        # ── Diarization ───────────────────────────────────────────────────
        # max_speakers is sent in the WS handshake as "max_speakers".
        # The SpeakerDiarizer on the server side caps new-speaker creation
        # once this limit is reached, forcing closer segments into existing
        # speakers instead of spinning up new ones.
        enable_diarization=True,
        max_speakers=MAX_SPEAKERS,
        # ── Output ────────────────────────────────────────────────────────
        log_transcription=False,
        transcription_callback=_on_segment,
    )

    try:
        client()                # blocks until Ctrl+C
    except KeyboardInterrupt:
        pass

    if _interim_on_line:
        print()
    print("\n✅  Stopped.")


if __name__ == "__main__":
    main()
