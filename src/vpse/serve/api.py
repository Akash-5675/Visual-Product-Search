"""FastAPI endpoint.

    VPSE_BUNDLE=path/to/bundle uvicorn vpse.serve.api:app --port 8000

POST /search   multipart field "file", optional query k (default 5)
    -> {"match": bool, "confidence": float, "threshold": float, "results": [...]}
GET  /thumb/{gallery_index}   thumbnail, if the bundle carries them
GET  /health
"""
import io
import os
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError

from vpse.serve.engine import SearchEngine

app = FastAPI(title="Visual Product Search", version="1.0")
_engine: SearchEngine | None = None


def engine() -> SearchEngine:
    global _engine
    if _engine is None:
        _engine = SearchEngine(Path(os.environ.get("VPSE_BUNDLE", "serve/bundle")))
    return _engine


@app.get("/health")
def health():
    e = engine()
    return {"ok": True, "gallery_size": int(len(e.gallery)),
            "threshold": e.threshold, "views": list(e.views)}


@app.post("/search")
async def search(file: UploadFile = File(...), k: int = Query(5, ge=1, le=50)):
    data = await file.read()
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "not a decodable image")
    return engine().search(img, k=k)


@app.get("/thumb/{gallery_index}")
def thumb(gallery_index: int):
    p = engine().thumb_path(gallery_index)
    if p is None:
        raise HTTPException(404, "no thumbnail")
    return FileResponse(p, media_type="image/jpeg")
