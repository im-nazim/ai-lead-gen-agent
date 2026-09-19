import threading
import time

_lock = threading.Lock()
_last_call_start = [0.0]


def throttled_invoke(chain, inputs: dict, interval: float):
    """
    Ensures at least `interval` seconds pass between the START of one
    Gemini call and the START of the next, project-wide, regardless of
    what else happened in between (web searches, processing time).
    If a rate-limit error slips through anyway, waits longer and retries
    once before giving up.
    """
    with _lock:
        now = time.time()
        elapsed = now - _last_call_start[0]
        if elapsed < interval:
            time.sleep(interval - elapsed)
        _last_call_start[0] = time.time()

    try:
        return chain.invoke(inputs)
    except Exception as e:
        if "429" in str(e) or "ResourceExhausted" in str(e):
            time.sleep(interval * 3)
            with _lock:
                _last_call_start[0] = time.time()
            return chain.invoke(inputs)
        raise
