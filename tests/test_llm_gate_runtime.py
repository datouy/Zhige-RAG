import threading
import time

from src.llm_concurrency import LLMConcurrencyGate, LLMConcurrencyLimitError


def test_sync_gate_limits_peak_concurrency():
    gate = LLMConcurrencyGate(max_concurrent=2, max_queue=10, queue_timeout=2)
    current = 0
    peak = 0
    lock = threading.Lock()

    def work():
        nonlocal current, peak
        with gate.blocking():
            with lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.05)
            with lock:
                current -= 1

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert peak == 2
    assert gate.stats()["active"] == 0
    assert gate.stats()["waiting"] == 0


def test_gate_rejects_when_queue_is_full():
    gate = LLMConcurrencyGate(max_concurrent=1, max_queue=0, queue_timeout=0.1)
    with gate.blocking():
        try:
            with gate.blocking():
                pass
        except LLMConcurrencyLimitError:
            pass
        else:
            raise AssertionError("queue full must reject")
