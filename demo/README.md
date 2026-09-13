---
title: Visual Product Search (OOD-aware)
emoji: 🔍
colorFrom: indigo
colorTo: green
sdk: gradio
sdk_version: "6.27.0"
app_file: app.py
pinned: false
---

Upload a product photo → top-k catalog matches, or **No match** when the query is
out-of-catalog. See the main repo README for training, results and the refusal curve.

## Deploying to HuggingFace Spaces

The Space needs, beside `app.py`:

```
app.py
requirements.txt
vpse/serve/engine.py      (copy src/vpse/serve/engine.py + an empty vpse/__init__.py, vpse/serve/__init__.py)
bundle/                   (from the Phase 5 export notebook: bundle_demo.zip, unzipped)
```

`scripts/stage_space.py` assembles that folder for you.
