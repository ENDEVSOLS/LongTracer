#!/usr/bin/env bash
set -euo pipefail

echo "🎬 Generating LongTracer demo GIFs..."
mkdir -p demo-vhs-gifs

for tape in demo-vhs-tapes/*.tape; do
  name=$(basename "$tape" .tape)
  echo "  Recording: $name"
  vhs "$tape"
  echo "  ✓ Generated: demo-vhs-gifs/$name.gif"
done

echo ""
echo "✅ All GIFs generated in demo-vhs-gifs/"
ls -lh demo-vhs-gifs/*.gif
