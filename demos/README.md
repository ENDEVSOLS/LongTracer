# LongTracer Demos

This directory contains interactive demonstrations of LongTracer's hallucination detection capabilities.

## Prerequisites

To run the interactive TUI demo, you just need `longtracer` and `rich` installed:

```bash
pip install longtracer rich
```

*(If you are working within the repository, the included `.venv` already has everything installed).*

## Running the Demo

```bash
python demos/hallucination_detection.py
```

### What to expect:

The demo will showcase the core workflow:
1. **Model Loading:** Initializes the local `sentence-transformers` and `DeBERTa-v3` models.
2. **Clean Pass:** Verifies a factually correct response (Trust Score ≈ 1.0).
3. **Obvious Hallucination:** Catches a clear factual contradiction (e.g., Paris vs Berlin).
4. **Subtle Fabrication:** Catches nuanced errors (e.g., name swaps).
5. **Batch Summary:** Displays aggregated metrics across multiple verification tasks.

> **Note:** LongTracer runs 100% locally. It does not use OpenAI or any external API keys. All verification happens on your machine using lightweight NLI models.
