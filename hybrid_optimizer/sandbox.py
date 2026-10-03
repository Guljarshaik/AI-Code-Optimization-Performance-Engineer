"""sandbox.py — Isolated, process-based execution harness (reference/production design)."""

from __future__ import annotations

import io
import multiprocessing as mp
import time
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionResult:
    success: bool
    return_value: Any = None
    stdout: str = ""
    stderr: str = ""
    exception: Optional[str] = None
    wall_time_s: float = 0.0
    timed_out: bool = False


def _worker(source: str, entry_point: str, args: tuple, kwargs: dict, conn) -> None:
    out, err = io.StringIO(), io.StringIO()
    result: Dict[str, Any] = {"success": False}
    start = time.perf_counter()
    try:
        namespace: Dict[str, Any] = {"__name__": "__sandbox__"}
        with redirect_stdout(out), redirect_stderr(err):
            exec(compile(source, "<candidate>", "exec"), namespace)
            fn = namespace.get(entry_point)
            if fn is None:
                raise NameError(f"entry point '{entry_point}' not defined by candidate")
            ret = fn(*args, **kwargs)
        result.update(success=True, return_value=ret)
    except Exception:
        result["exception"] = traceback.format_exc(limit=4)
    finally:
        result["wall_time_s"] = time.perf_counter() - start
        result["stdout"] = out.getvalue()
        result["stderr"] = err.getvalue()
        conn.send(result)
        conn.close()


class Sandbox:
    def __init__(self, timeout_s: float = 3.0):
        self.timeout_s = timeout_s

    def run(self, source: str, entry_point: str, *args, **kwargs) -> ExecutionResult:
        ctx = mp.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe()
        proc = ctx.Process(target=_worker, args=(source, entry_point, args, kwargs, child_conn))
        proc.start()
        proc.join(self.timeout_s)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return ExecutionResult(success=False, timed_out=True,
                                    exception="TimeoutError: execution exceeded sandbox limit",
                                    wall_time_s=self.timeout_s)

        if parent_conn.poll():
            payload = parent_conn.recv()
            return ExecutionResult(
                success=payload["success"],
                return_value=payload.get("return_value"),
                stdout=payload.get("stdout", ""),
                stderr=payload.get("stderr", ""),
                exception=payload.get("exception"),
                wall_time_s=payload.get("wall_time_s", 0.0),
            )
        return ExecutionResult(success=False, exception="No result returned (process crashed)")

    def run_test_suite(self, source: str, entry_point: str,
                        test_cases: List[Dict[str, Any]]) -> List[ExecutionResult]:
        results = []
        for case in test_cases:
            r = self.run(source, entry_point, *case.get("args", ()), **case.get("kwargs", {}))
            results.append(r)
        return results
