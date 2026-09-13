"""Assemble a HuggingFace Space folder from the repo + a demo bundle.

    python scripts/stage_space.py --bundle path/to/bundle_demo --out space/

Then in `space/`:  git init, connect to your Space, commit, push.
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="demo bundle dir (with thumbs/)")
    ap.add_argument("--out", default="space")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    for f in ("app.py", "requirements.txt", "README.md"):
        shutil.copy(ROOT / "demo" / f, out / f)
    pkg = out / "vpse" / "serve"
    pkg.mkdir(parents=True)
    (out / "vpse" / "__init__.py").touch()
    (pkg / "__init__.py").touch()
    shutil.copy(ROOT / "src" / "vpse" / "serve" / "engine.py", pkg / "engine.py")
    shutil.copytree(args.bundle, out / "bundle")

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 2**20
    print(f"staged {out}/  ({size:.0f} MB)")


if __name__ == "__main__":
    main()
