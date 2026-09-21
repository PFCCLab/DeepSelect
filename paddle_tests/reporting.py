from __future__ import annotations

import json
from pathlib import Path
import time


class Reporter:
    def __init__(self, directory: Path, manifest):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.results_path = directory / "results.jsonl"
        self.counts = {}
        with (directory / "manifest.jsonl").open("w") as file:
            for item in manifest:
                file.write(json.dumps(item, sort_keys=True) + "\n")

    def emit(self, **record):
        record.setdefault("timestamp", time.time())
        status = record.get("status", "UNKNOWN")
        self.counts[status] = self.counts.get(status, 0) + 1
        with self.results_path.open("a") as file:
            file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        print(json.dumps(record, sort_keys=True, default=str), flush=True)

    def finish(self, planned: int):
        executed = sum(self.counts.values())
        if self.counts.get("ERROR"):
            status, code = "ERROR", 2
        elif self.counts.get("FAIL"):
            status, code = "FAIL", 1
        elif self.counts.get("SKIP_RESOURCE") or executed < planned:
            status, code = "INCOMPLETE", 3
        elif executed == planned:
            status, code = "PASS", 0
        else:
            status, code = "ERROR", 2
        summary = {"status": status, "planned": planned, "executed": executed, "counts": self.counts}
        (self.directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(json.dumps(summary, sort_keys=True))
        return code
