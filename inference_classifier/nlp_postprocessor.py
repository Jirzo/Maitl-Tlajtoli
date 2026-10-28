"""
NLP post-processor for Maitl Tlajtoli.

Architecture:
  LSTM word predictions → word buffer → LLaMA 3 (local via Ollama) → Spanish sentence

Why a local LLM instead of an online API?
  - The system is designed for real-time, offline use with deaf/hard-of-hearing users.
  - Privacy: no sign language gestures or transcriptions are sent to the cloud.
  - Latency: a local quantized model (e.g. LLaMA 3 8B Q4) runs in ~200 ms on a
    mid-range GPU, which is acceptable for a conversational display.

Setup prerequisites:
  1. Install Ollama: https://ollama.com/download
  2. Pull the model: `ollama pull llama3`     (or llama3:8b for smaller footprint)
  3. Start the server: `ollama serve`         (runs on http://localhost:11434)

Usage:
  from inference_classifier.nlp_postprocessor import WordBuffer, sentence_from_words

  buf = WordBuffer(max_words=12, silence_timeout_s=2.0)
  buf.push("HOLA")
  buf.push("YO")
  buf.push("COMER")
  sentence = buf.flush_to_sentence()  # "Hola, yo quiero comer."
"""

import time
import threading
import requests
from collections import deque
from typing import Callable


# ---------------------------------------------------------------------------
# Ollama connection
# ---------------------------------------------------------------------------

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "llama3"

# System prompt — tune this as you learn more about LSM grammar
_SYSTEM_PROMPT = """
Eres un intérprete de Lengua de Señas Mexicana (LSM).
Recibirás una lista de palabras predichas por un reconocedor de señas.
Las palabras siguen el orden gramatical típico de LSM: Topic-Comment, no Sujeto-Verbo-Objeto.
Tu tarea es convertir esas palabras en UNA sola oración natural en español.
Reglas:
  - Mantén el significado, añade artículos, conjugaciones y preposiciones faltantes.
  - Si las palabras no forman una oración coherente, devuelve la mejor aproximación.
  - Responde SOLO con la oración en español, sin explicaciones ni comillas.
""".strip()


def ollama_available(timeout_s: float = 2.0) -> bool:
    """Quick health-check to see if the Ollama server is running."""
    try:
        r = requests.get("http://localhost:11434/", timeout=timeout_s)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def sentence_from_words(words: list[str], model: str = DEFAULT_MODEL) -> str:
    """
    Send a list of LSM words to LLaMA 3 and return a grammatical Spanish sentence.

    Falls back gracefully to a simple join if Ollama is not running.

    Args:
        words:  List of predicted words/signs in sequence order, e.g. ["YO", "COMER", "MANZANA"]
        model:  Ollama model name (default: "llama3")

    Returns:
        A natural Spanish sentence string.
    """
    if not words:
        return ""

    if not ollama_available():
        print("[NLP] Ollama not running — using raw word join as fallback.")
        return " ".join(w.lower().capitalize() for w in words) + "."

    user_message = "Palabras: " + " ".join(words)

    payload = {
        "model": model,
        "system": _SYSTEM_PROMPT,
        "prompt": user_message,
        "stream": False,
        "options": {
            "temperature": 0.3,   # low temperature = more deterministic grammar
            "top_p": 0.9,
            "num_predict": 80,    # cap response length for latency
        },
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=10.0)
        response.raise_for_status()
        sentence = response.json().get("response", "").strip()
        return sentence if sentence else " ".join(words) + "."
    except requests.exceptions.Timeout:
        print("[NLP] Ollama request timed out.")
        return " ".join(words) + "."
    except Exception as e:
        print(f"[NLP] Error calling Ollama: {e}")
        return " ".join(words) + "."


# ---------------------------------------------------------------------------
# Word buffer — accumulates predictions between pauses
# ---------------------------------------------------------------------------

class WordBuffer:
    """
    Accumulates word predictions from the LSTM and flushes them to the LLM
    when the signer pauses (no new word for `silence_timeout_s` seconds).

    Integration example (inside the live inference loop):

        buf = WordBuffer(max_words=12, silence_timeout_s=2.5,
                         on_sentence=lambda s: print("SENTENCE:", s))

        # After each stable LSTM prediction:
        buf.push(predicted_word)

        # Call buf.tick() every frame to trigger auto-flush on silence.
        buf.tick()
    """

    def __init__(
        self,
        max_words: int = 12,
        silence_timeout_s: float = 2.5,
        on_sentence: Callable[[str], None] | None = None,
        model: str = DEFAULT_MODEL,
    ):
        """
        Args:
            max_words:         Auto-flush when this many words accumulate.
            silence_timeout_s: Auto-flush if no new word arrives within this window.
            on_sentence:       Callback invoked with the generated sentence string.
                               If None, the sentence is just printed.
            model:             Ollama model name.
        """
        self._words: deque[str] = deque(maxlen=max_words)
        self._max_words = max_words
        self._timeout = silence_timeout_s
        self._last_push = time.time()
        self._on_sentence = on_sentence or (lambda s: print(f"[LSM→NLP] {s}"))
        self._model = model
        self._flushing = False  # prevent concurrent flushes

    def push(self, word: str) -> None:
        """Add a newly predicted word. Triggers flush if buffer is full."""
        # Deduplicate consecutive identical predictions (stabilization)
        if self._words and self._words[-1] == word:
            return

        self._words.append(word)
        self._last_push = time.time()

        if len(self._words) >= self._max_words:
            self._flush_async()

    def tick(self) -> None:
        """Call once per inference frame to enable silence-based auto-flush."""
        if (
            self._words
            and not self._flushing
            and (time.time() - self._last_push) >= self._timeout
        ):
            self._flush_async()

    def flush_to_sentence(self) -> str:
        """Synchronous flush — blocks until the LLM responds. Returns the sentence."""
        if not self._words:
            return ""
        words = list(self._words)
        self._words.clear()
        self._last_push = time.time()
        sentence = sentence_from_words(words, model=self._model)
        self._on_sentence(sentence)
        return sentence

    def _flush_async(self) -> None:
        """Non-blocking flush — runs LLM call on a daemon thread."""
        if self._flushing or not self._words:
            return
        self._flushing = True
        words = list(self._words)
        self._words.clear()
        self._last_push = time.time()

        def _worker():
            try:
                sentence = sentence_from_words(words, model=self._model)
                self._on_sentence(sentence)
            finally:
                self._flushing = False

        threading.Thread(target=_worker, daemon=True).start()

    @property
    def pending_words(self) -> list[str]:
        """Return a copy of the current buffer without clearing it."""
        return list(self._words)

    def __len__(self) -> int:
        return len(self._words)


# ---------------------------------------------------------------------------
# Quick standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_words = ["YO", "QUERER", "AGUA", "POR_FAVOR"]
    print(f"Words: {test_words}")
    print(f"Ollama available: {ollama_available()}")
    result = sentence_from_words(test_words)
    print(f"Sentence: {result}")
