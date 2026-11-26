"""Run the code cells of the Jupyter notebooks, in order, and fail if any notebook errors.

Each notebook's code cells are joined into one Julia script, which runs in its own process with the
working directory set to notebooks/. The setup cell then finds the clone (it walks up to the
Project.toml named SpatialEconometrics) and uses it, and the data helper reads notebooks/data. So this
checks the notebooks against the working tree, not against the release named in REV.

The seven Pluto notebooks (notebooks/*.jl) are not run: they need Pluto and PlutoUI.

Python standard library only, so CI needs no extra Julia package to read .ipynb files.

Run:  python -I notebooks/check_notebooks.py [--julia julia] [--jobs N] [notebook ...]
      (no notebook names: all notebooks/*.ipynb)
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def script_of(path):
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    return "\n\n".join(cells) + "\n"


def run_one(path, julia, tmp):
    name = os.path.basename(path)
    jl = os.path.join(tmp, name.replace(".ipynb", ".jl"))
    with open(jl, "w", encoding="utf-8") as f:
        f.write(script_of(path))
    t0 = time.time()
    p = subprocess.run([julia, f"--project={ROOT}", "--color=no", jl], cwd=HERE,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", errors="replace")
    return name, p.returncode, time.time() - t0, out


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser()
    ap.add_argument("notebooks", nargs="*")
    ap.add_argument("--julia", default="julia")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    paths = a.notebooks or sorted(glob.glob(os.path.join(HERE, "*.ipynb")))
    failed = []
    with tempfile.TemporaryDirectory() as tmp, cf.ThreadPoolExecutor(a.jobs) as ex:
        for name, rc, dt, out in ex.map(lambda p: run_one(p, a.julia, tmp), paths):
            status = "ok" if rc == 0 else f"FAILED (exit {rc})"
            print(f"{name:45s} {dt:7.1f} s  {status}", flush=True)
            if rc != 0:
                failed.append(name)
                tail = "\n".join(out.rstrip().splitlines()[-40:])
                print("---- last 40 lines ----\n" + tail + "\n----", flush=True)
    print(f"\n{len(paths) - len(failed)} of {len(paths)} notebooks ran cleanly.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
