"""Simple visualization server for legibility trial logs."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Dict, List, Optional, Sequence

from dictdiffer import diff as dictdiffer_diff

from .legibility import _parse as parse_serialized


def _stringify(value: Any) -> str:
    """Return a readable multi-line representation preserving structure."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, indent=2)
    except TypeError:
        return str(value)


def _path_to_string(path: Sequence[Any]) -> str:
    if not path:
        return "(root)"
    return ".".join(str(component) for component in path)


def _format_diff(expected: Any, actual: Any) -> str:
    try:
        changes = list(dictdiffer_diff(expected, actual))
    except Exception as exc:  # noqa: BLE001
        return f"diff error: {exc}"
    if not changes:
        return "structures match"
    lines: List[str] = []
    for op, path, values in changes:
        lines.append(f"{op} {_path_to_string(path)}: {_stringify(values)}")
    return "\n".join(lines)


class TrialStore:
    """Caches parsed trial records with modification-time tracking."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._mtime: float = -1.0
        self._lock = threading.Lock()
        self._cache: List[Dict[str, Any]] = []

    def _load_file(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        with self._path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("success") is None:
                    continue
                records.append(record)
        return records

    def get_records(self) -> List[Dict[str, Any]]:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            return []

        with self._lock:
            if mtime != self._mtime:
                self._cache = self._load_file()
                self._mtime = mtime
        return self._cache


def _summarize(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(records)
    successes = sum(1 for item in records if item.get("success"))
    jaccard_sum = sum(float(item.get("jaccard") or 0.0) for item in records)
    dictdiff_sum = sum(float(item.get("dictdiff_score") or 0.0) for item in records)
    by_format: Dict[str, Dict[str, Any]] = {}
    for item in records:
        fmt = item.get("format", "unknown")
        bucket = by_format.setdefault(
            fmt,
            {"total": 0, "success": 0, "jaccard_sum": 0.0, "dictdiff_sum": 0.0},
        )
        bucket["total"] += 1
        if item.get("success"):
            bucket["success"] += 1
        bucket["jaccard_sum"] += float(item.get("jaccard") or 0.0)
        bucket["dictdiff_sum"] += float(item.get("dictdiff_score") or 0.0)
    for bucket in by_format.values():
        total_fmt = bucket["total"]
        bucket["mean_jaccard"] = round(bucket["jaccard_sum"] / total_fmt, 4) if total_fmt else 0.0
        bucket["mean_dictdiff"] = round(bucket["dictdiff_sum"] / total_fmt, 4) if total_fmt else 0.0
        bucket.pop("jaccard_sum", None)
        bucket.pop("dictdiff_sum", None)
    summary = {
        "total_trials": total,
        "successes": successes,
        "failures": total - successes,
        "success_rate": round(successes / total, 4) if total else 0.0,
        "mean_jaccard": round(jaccard_sum / total, 4) if total else 0.0,
        "mean_dictdiff": round(dictdiff_sum / total, 4) if total else 0.0,
        "by_format": by_format,
    }
    return summary


def _sanitize(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    trimmed: List[Dict[str, Any]] = []
    for item in records:
        expected_obj = item.get("expected_output")
        model_raw = item.get("model_response")
        observation = item.get("observation")
        model_text = _stringify(model_raw)
        parsed_actual = None
        if isinstance(model_raw, (dict, list)):
            parsed_actual = model_raw
        elif isinstance(model_raw, str) and model_raw.strip() and observation is None:
            try:
                parsed_actual = parse_serialized(item.get("format", ""), model_raw)
            except Exception as exc:  # noqa: BLE001
                parsed_actual = None
                model_text = f"{model_text}\n\n(parse error: {exc})"
        diff_text = ""
        if expected_obj is not None and parsed_actual is not None:
            diff_text = _format_diff(expected_obj, parsed_actual)
        elif expected_obj is not None:
            diff_text = "unable to parse model response for diff"
        trimmed.append(
            {
                "timestamp": item.get("timestamp"),
                "format": item.get("format"),
                "input_nodes": item.get("input_nodes"),
                "output_nodes": item.get("output_nodes"),
                "trial_index": item.get("trial_index"),
                "combo_index": item.get("combo_index"),
                "success": item.get("success"),
                "jaccard": item.get("jaccard"),
                "dictdiff": item.get("dictdiff_score"),
                "prompt_length": item.get("prompt_length"),
                "response_length": item.get("response_length"),
                "observation": _stringify(observation),
                "expected_text": _stringify(expected_obj),
                "model_response_text": model_text,
                "diff": diff_text,
            }
        )
    return trimmed


def _build_payload(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = _summarize(records)
    sanitized = _sanitize(records)
    return {"summary": summary, "trials": sanitized}


def _html_page() -> bytes:
    html = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Legibility Trials</title>
  <style>
    body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #f9fafb; color: #111827; }
    header { margin-bottom: 1rem; }
    table { border-collapse: collapse; width: 100%; margin-top: 1rem; background: #fff; }
    th, td { border: 1px solid #e5e7eb; padding: 0.5rem; text-align: left; vertical-align: top; }
    td.multiline { white-space: pre-wrap; font-family: 13px/1.4 Menlo, Consolas, monospace; }
    th { background: #f3f4f6; position: sticky; top: 0; }
    tr:nth-child(even) { background: #f9fafb; }
    code { background: #f3f4f6; padding: 0.1rem 0.25rem; border-radius: 4px; }
    .summary-cards { display: flex; flex-wrap: wrap; gap: 1rem; }
    .card { background: #fff; padding: 1rem; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); min-width: 180px; }
    .by-format { margin-top: 1rem; }
  </style>
</head>
<body>
  <header>
    <h1>Legibility Trial Dashboard</h1>
    <p>Visualization of JSONL trial logs. Data refreshes when the page reloads.</p>
  </header>
  <section id="summary">
    <div class="summary-cards">
      <div class="card">
        <strong>Total trials</strong>
        <div id="totalTrials">—</div>
      </div>
      <div class="card">
        <strong>Success rate</strong>
        <div id="successRate">—</div>
      </div>
      <div class="card">
        <strong>Mean Jaccard</strong>
        <div id="meanJaccard">—</div>
      </div>
      <div class="card">
        <strong>Mean DictDiff</strong>
        <div id="meanDictdiff">—</div>
      </div>
      <div class="card">
        <strong>Successes</strong>
        <div id="successCount">—</div>
      </div>
      <div class="card">
        <strong>Failures</strong>
        <div id="failureCount">—</div>
      </div>
    </div>
    <div class="by-format" id="byFormat"></div>
  </section>

  <section>
    <h2>Trials</h2>
    <table id="trialTable">
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Format</th>
          <th>Input Nodes</th>
          <th>Output Nodes</th>
          <th>Trial Index</th>
          <th>Combo Index</th>
          <th>Success</th>
          <th>Jaccard</th>
          <th>DictDiff</th>
          <th>Prompt Chars</th>
          <th>Response Chars</th>
          <th>Expected</th>
          <th>Model Response</th>
          <th>Diff</th>
          <th>Observation</th>
        </tr>
      </thead>
      <tbody id="trialBody">
      </tbody>
    </table>
  </section>

  <script>
    async function fetchData() {
      const response = await fetch('/api/trials');
      if (!response.ok) {
        throw new Error('Failed to load trials');
      }
      return await response.json();
    }

    function renderSummary(summary) {
      const rate = summary.success_rate ? (summary.success_rate * 100).toFixed(1) + '%' : '0%';
      const meanJaccard = typeof summary.mean_jaccard === 'number' ? summary.mean_jaccard.toFixed(3) : '0.000';
      const meanDictdiff = typeof summary.mean_dictdiff === 'number' ? summary.mean_dictdiff.toFixed(3) : '0.000';
      document.getElementById('totalTrials').textContent = summary.total_trials;
      document.getElementById('successRate').textContent = rate;
      document.getElementById('meanJaccard').textContent = meanJaccard;
      document.getElementById('meanDictdiff').textContent = meanDictdiff;
      document.getElementById('successCount').textContent = summary.successes;
      document.getElementById('failureCount').textContent = summary.failures;

      const formatContainer = document.getElementById('byFormat');
      formatContainer.innerHTML = '<h3>By Format</h3>';
      const list = document.createElement('ul');
      for (const [fmt, stats] of Object.entries(summary.by_format)) {
        const li = document.createElement('li');
        const fmtRate = stats.total ? ((stats.success / stats.total) * 100).toFixed(1) + '%' : '0%';
        const fmtMeanJaccard = stats.total ? (stats.mean_jaccard ?? 0).toFixed(3) : '0.000';
        const fmtMeanDictdiff = stats.total ? (stats.mean_dictdiff ?? 0).toFixed(3) : '0.000';
        li.textContent = fmt + ': ' + stats.success + '/' + stats.total + ' (' + fmtRate + '), Jaccard ' + fmtMeanJaccard + ', DictDiff ' + fmtMeanDictdiff;
        list.appendChild(li);
      }
      formatContainer.appendChild(list);
    }

    function renderTrials(trials) {
      const body = document.getElementById('trialBody');
      body.innerHTML = '';
      for (const trial of trials) {
        const row = document.createElement('tr');
        const cells = [
          { value: trial.timestamp || '' },
          { value: trial.format || '' },
          { value: trial.input_nodes ?? '' },
          { value: trial.output_nodes ?? '' },
          { value: trial.trial_index ?? '' },
          { value: trial.combo_index ?? '' },
          { value: trial.success ? '✅' : '❌' },
          { value: trial.jaccard !== undefined && trial.jaccard !== null ? Number(trial.jaccard).toFixed(3) : '' },
          { value: trial.dictdiff !== undefined && trial.dictdiff !== null ? Number(trial.dictdiff).toFixed(3) : '' },
          { value: trial.prompt_length ?? '' },
          { value: trial.response_length ?? '' },
          { value: trial.expected_text || '', className: 'multiline' },
          { value: trial.model_response_text || '', className: 'multiline' },
          { value: trial.diff || '', className: 'multiline' },
          { value: trial.observation || '' }
        ];
        for (const cell of cells) {
          const td = document.createElement('td');
          td.textContent = cell.value;
          if (cell.className) {
            td.className = cell.className;
          }
          row.appendChild(td);
        }
        body.appendChild(row);
      }
    }

    fetchData()
      .then(data => {
        renderSummary(data.summary);
        renderTrials(data.trials);
      })
      .catch(error => {
        document.body.innerHTML = '<p style="color: red;">' + error.message + '</p>';
      });
  </script>
</body>
</html>
"""
    return html.encode("utf-8")


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def _handler_factory(store: TrialStore):
    class VisualizationHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/" or self.path.startswith("/index"):
                self._serve_index()
                return
            if self.path.startswith("/api/trials"):
                self._serve_api()
                return
            self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return  # Suppress default stdout logging for cleanliness.

        def _serve_index(self) -> None:
            payload = _html_page()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _serve_api(self) -> None:
            records = store.get_records()
            payload = json.dumps(_build_payload(records), ensure_ascii=False).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return VisualizationHandler


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve legibility trial visualizations.")
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to the JSONL trials log file to load. Defaults to the most recent file in data/legibility_logs.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind (default: 8765).")
    return parser.parse_args(argv)


def _find_latest_log(directory: Path) -> Optional[Path]:
    if not directory.exists() or not directory.is_dir():
        return None
    candidates = sorted(
        directory.glob("trials_*.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    fallback = directory / "trials_latest.jsonl"
    return fallback if fallback.exists() else None


def _resolve_log_path(cli_value: Optional[Path]) -> Path:
    if cli_value is not None:
        return cli_value
    base_dir = Path("data") / "legibility_logs"
    latest = _find_latest_log(base_dir)
    if latest:
        return latest
    raise FileNotFoundError(
        f"No legibility logs found in {base_dir}. Generate a log or specify --log-file explicitly."
    )


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    log_path = _resolve_log_path(args.log_file)
    store = TrialStore(log_path)
    handler_cls = _handler_factory(store)

    server = ThreadedHTTPServer((args.host, args.port), handler_cls)
    address = f"http://{args.host}:{args.port}"
    print(f"Serving trial dashboard on {address} (log: {log_path})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
