"""
fast_sandbox.py — Fast in-process evaluator used ONLY to make the
large-scale batch experiment (run_experiments.py) computationally
feasible. Not a production component; see sandbox.py and the paper's
disclosure (Methodology: Sandbox Isolation) for the reference design.
"""

from __future__ import annotations

import io
import signal
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List

from hybrid_optimizer.sandbox import ExecutionResult


class _Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _Timeout()


class FastInProcessSandbox:
    def __init__(self, timeout_s: float = 2.0):
        self.timeout_s = timeout_s

    def run(self, source: str, entry_point: str, *args, **kwargs) -> ExecutionResult:
        out, err = io.StringIO(), io.StringIO()
        start = time.perf_counter()
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, self.timeout_s)
        try:
            namespace: Dict[str, Any] = {"__name__": "__sandbox__"}
            with redirect_stdout(out), redirect_stderr(err):
                exec(compile(source, "<candidate>", "exec"), namespace)
                fn = namespace.get(entry_point)
                if fn is None:
                    raise NameError(f"entry point '{entry_point}' not defined by candidate")
                ret = fn(*args, **kwargs)
            return ExecutionResult(success=True, return_value=ret,
                                    stdout=out.getvalue(), stderr=err.getvalue(),
                                    wall_time_s=time.perf_counter() - start)
        except _Timeout:
            return ExecutionResult(success=False, timed_out=True,
                                    exception="TimeoutError: execution exceeded sandbox limit",
                                    wall_time_s=self.timeout_s)
        except Exception:
            return ExecutionResult(success=False, exception=traceback.format_exc(limit=4),
                                    stdout=out.getvalue(), stderr=err.getvalue(),
                                    wall_time_s=time.perf_counter() - start)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def run_test_suite(self, source: str, entry_point: str,
                        test_cases: List[Dict[str, Any]]) -> List[ExecutionResult]:
        return [self.run(source, entry_point, *c.get("args", ()), **c.get("kwargs", {}))
                for c in test_cases]
