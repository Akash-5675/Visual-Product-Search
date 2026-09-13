"""Gradio demo. Runs anywhere with a bundle next to it -- built for HuggingFace Spaces.

    python demo/app.py                      # uses ./bundle or $VPSE_BUNDLE
"""
import os
import sys
from pathlib import Path

import gradio as gr
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))       # repo layout
sys.path.insert(0, str(HERE))                       # Spaces layout (vpse/ copied beside app.py)
from vpse.serve.engine import SearchEngine  # noqa: E402

BUNDLE = Path(os.environ.get("VPSE_BUNDLE", HERE / "bundle"))
engine = SearchEngine(BUNDLE)

# HuggingFace's free tier runs Gradio Spaces on ZeroGPU, which refuses to start
# unless some function is decorated with @spaces.GPU -- even though this app is
# CPU-only ONNX and never touches a GPU. The placeholder satisfies the check and
# is never called. Locally the `spaces` package is absent and this is skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass


def search(img: Image.Image, k: int):
    if img is None:
        return "Upload an image to search.", []
    out = engine.search(img, k=int(k))
    conf = out["confidence"]
    thr = out["threshold"]
    if out["match"]:
        status = (f"### Match &nbsp; <span style='color:#2a2'>●</span> "
                  f"confidence {conf:.3f} (threshold {thr:.3f})")
    elif out.get("refusal_reason") == "blank_image":
        status = ("### No match &nbsp; <span style='color:#c33'>●</span> "
                  "the image is blank or a single flat colour — nothing to search for.")
    else:
        status = (f"### No match &nbsp; <span style='color:#c33'>●</span> "
                  f"nearest item only {conf:.3f} similar (threshold {thr:.3f}) — "
                  f"this doesn't look like anything in the catalog. "
                  f"Closest items shown for reference.")
    gallery = []
    for r in out["results"]:
        p = engine.thumb_path(r["gallery_index"])
        if p is None:
            continue
        gallery.append((str(p), f"#{r['rank']}  {r['category']}  ·  product {r['product_id']}  ·  {r['similarity']:.3f}"))
    return status, gallery


with gr.Blocks(title="Visual Product Search") as demo:
    gr.Markdown(
        "# Visual Product Search with OOD-aware refusal\n"
        "Upload a product photo. Returns the closest catalog products, or **No match** "
        "when the query is out-of-catalog. Model: ResNet50 + batch-hard triplet, "
        "hflip TTA, ONNX. Catalog: Stanford Online Products (unseen test products)."
    )
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="query image")
            k = gr.Slider(1, 10, value=5, step=1, label="results")
            btn = gr.Button("Search", variant="primary")
        with gr.Column(scale=2):
            status = gr.Markdown()
            gal = gr.Gallery(label="top matches", columns=5, height=260)
    btn.click(search, [inp, k], [status, gal])
    inp.change(search, [inp, k], [status, gal])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
