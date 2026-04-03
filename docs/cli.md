# CLI Reference

LongTracer installs a `longtracer` command for verification and trace inspection.

## `longtracer check` — Verify from the command line

```bash
longtracer check "The Eiffel Tower is in Berlin." "The Eiffel Tower is in Paris."
```

Output:
```
✗ FAIL  trust=0.50  hallucinations=1
  0/1 claims supported, 1 hallucination(s) detected.

  ✗ The Eiffel Tower is in Berlin. [HALLUCINATION]
    ↳ source: The Eiffel Tower is in Paris.
```

### Options

| Option | Description |
|--------|-------------|
| `--json` | Output results as JSON |
| `--threshold 0.7` | Set verification threshold (default: 0.5) |

### JSON output

```bash
longtracer check "response" "source" --json
```

```json
{
  "verdict": "FAIL",
  "trust_score": 0.5,
  "summary": "0/1 claims supported, 1 hallucination(s) detected.",
  "hallucination_count": 1,
  "claims": [...]
}
```

---

## Commands

### `longtracer view`

List recent traces:

```bash
longtracer view
longtracer view --limit 20
longtracer view --project chatbot-prod
```

### View a specific trace

```bash
longtracer view --id <trace_id>
```

### View the most recent trace

```bash
longtracer view --last
```

### Export to JSON

```bash
longtracer view --export <trace_id>
# Saves to trace_<id>.json
```

### Export to HTML report

```bash
longtracer view --html <trace_id>
# Saves to trace_<id>.html — open in any browser
```

---

## Options

| Option | Description |
|--------|-------------|
| `--id <trace_id>` | View a specific trace by ID |
| `--last` | View the most recent trace |
| `--project <name>` | Filter traces by project name |
| `--limit N` | Max number of traces to list (default: 10) |
| `--export <trace_id>` | Export trace to JSON file |
| `--html <trace_id>` | Export trace to self-contained HTML report |

---

## HTML Report

The HTML report is a self-contained single file with:

- Summary cards: verdict, trust score, total duration, claim count
- Waterfall timeline of all pipeline spans
- Per-claim verification table with scores and source evidence
- No external dependencies — works offline

```bash
longtracer view --html abc123
open trace_abc123.html
```

---

## Environment Variables

The CLI respects the same environment variables as the SDK:

```bash
TRACE_CACHE_BACKEND=sqlite longtracer view
TRACE_PROJECT=chatbot-prod longtracer view
```
