#!/usr/bin/env python3
"""EURS reproduction check: run it on your repository BEFORE you hand in.

    python repro_check.py .                              # layout, repro.yaml, weights link, run.py/export_onnx.py signatures
    python repro_check.py . --run --input DIR [--n 5]    # also run main() on a few of your degraded images and check outputs
    python repro_check.py . --export-onnx                # Track 2: also run export_onnx.py and check the ONNX file

One file, no installation. It needs only the Python standard library for the static checks; `--run` needs
numpy and pillow (and whatever your own code needs), `--export-onnx` needs onnx. Run the `--run` check inside
the organizers' base image if you can (pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime plus numpy, pillow,
opencv-python-headless, timm, einops, onnx, pyyaml, scipy, scikit-image) with the network switched off:

    docker run --rm --network none --gpus all -v "$PWD":/pkg:ro -v /path/to/images:/in:ro -v /tmp/out:/out \\
        pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime python /pkg/repro_check.py /pkg --run --input /in --output /out

Exit code 0 = no errors (warnings are only advice). This tool checks the form of your hand-in; it cannot tell whether
your results will reproduce, which the organizers decide by actually re-running them.
"""
import argparse
import ast
import importlib.util
import os
import random
import re
import shutil
import sys
import tempfile
import time
import urllib.parse

TRACKS = ("eurs-t1", "eurs-t2")
TEAM_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAIN_PARAMS = ["model_dir", "input_dir", "output_dir", "device"]
ONNX_PARAMS = ["model_dir", "out_onnx"]
LINK_FILE = os.path.join("weights", "weights_link.yaml")
MAX_PACKAGE_BYTES = 5 * 1024 ** 3
MAX_SIDE_WARN = 1024
MAX_SIDE_ERR = 2048
NUM_CLASSES = 5
IMG_EXTS = (".png", ".jpg", ".jpeg")

errors, warnings = [], []


def err(msg):
    errors.append(msg)
    print(f"ERROR   {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"warning {msg}")


def ok(msg):
    print(f"ok      {msg}")


# ------------------------------------------------------------------ tiny YAML reader (flat key: value)
def read_flat_yaml(path):
    try:
        import yaml  # optional
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}
    except ImportError:
        pass
    data = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].rstrip()
            if ":" not in line or line.startswith((" ", "\t")):
                continue
            key, val = line.split(":", 1)
            val = val.strip().strip("'\"")
            if val.isdigit():
                val = int(val)
            data[key.strip()] = val
    return data


def function_params(path, name):
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError, OSError) as exc:
        return "invalid", str(exc)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return [a.arg for a in node.args.args + node.args.kwonlyargs], None
    return None, None


# ------------------------------------------------------------------ static checks
def check_repro_yaml(pkg):
    path = os.path.join(pkg, "repro.yaml")
    before = len(errors)
    if not os.path.isfile(path):
        err("repro.yaml is missing")
        return None
    d = read_flat_yaml(path)
    if not isinstance(d.get("team"), str) or not TEAM_RE.match(d["team"]):
        err("repro.yaml: team must be your team name as on macvi.org (letters, digits, . _ - only, max 64)")
    if d.get("track") not in TRACKS:
        err(f"repro.yaml: track must be one of {', '.join(TRACKS)}")
    sid = d.get("submission_id")
    if isinstance(sid, bool) or not isinstance(sid, int) or sid <= 0:
        err("repro.yaml: submission_id must be the positive integer ID of your best-scoring TESTING-PHASE submission in this track")
    if not isinstance(d.get("contact"), str) or "@" not in d["contact"]:
        err("repro.yaml: contact must be an e-mail address")
    if d.get("team") == "your-team-name" or d.get("submission_id") == 12345:
        warn("repro.yaml still has the template values (the organizers rebuild it from your factsheet, but fill it in anyway)")
    if len(errors) == before:
        ok("repro.yaml")
    return d


def check_method(pkg):
    p = os.path.join(pkg, "METHOD.md")
    if not os.path.isfile(p):
        err("METHOD.md is missing (a short description of the method is required)")
        return
    text = open(p, encoding="utf-8", errors="ignore").read()
    if not text.strip():
        err("METHOD.md is empty")
    elif "<name>" in text and "<1 / 2 / both>" in text:
        warn("METHOD.md still contains the template placeholders")
    else:
        ok("METHOD.md")


def check_entry_points(pkg, track):
    run_py = os.path.join(pkg, "run.py")
    if not os.path.isfile(run_py):
        err("run.py is missing")
    else:
        params, why = function_params(run_py, "main")
        if params == "invalid":
            err(f"run.py does not parse as Python: {why}")
        elif params is None:
            err("run.py must define main(model_dir, input_dir, output_dir, device)")
        elif params[:4] != MAIN_PARAMS:
            err(f"run.py main() must take ({', '.join(MAIN_PARAMS)}), found ({', '.join(params)})")
        else:
            ok("run.py main() signature")
    onnx_py = os.path.join(pkg, "export_onnx.py")
    if track == "eurs-t2":
        if not os.path.isfile(onnx_py):
            err("export_onnx.py is missing (required for Track 2)")
        else:
            params, why = function_params(onnx_py, "main")
            if params in (None, "invalid") or params[:2] != ONNX_PARAMS:
                err(f"export_onnx.py must define main({', '.join(ONNX_PARAMS)})")
            else:
                ok("export_onnx.py main() signature")


def check_weights(pkg):
    wdir = os.path.join(pkg, "weights")
    link = os.path.join(pkg, LINK_FILE)
    real = [f for f in (os.listdir(wdir) if os.path.isdir(wdir) else [])
            if f not in ("weights_link.yaml", "README.md", ".gitkeep")]
    if os.path.isfile(link):
        d = read_flat_yaml(link)
        url = d.get("url")
        sha = str(d.get("sha256") or "").lower()
        if not isinstance(url, str) or urllib.parse.urlparse(url).scheme != "https":
            err(f"{LINK_FILE}: `url:` must be an https link")
        elif "example.org" in url:
            err(f"{LINK_FILE} still has the template link (https://example.org/...)")
        elif "drive.google.com" in url and "/folders/" in url:
            err(f"{LINK_FILE}: Google Drive folder links are not supported; link to one file or one .zip")
        elif "1drv.ms" in url or "onedrive" in url:
            err(f"{LINK_FILE}: OneDrive links are not supported; use a direct link, GitHub release, Hugging Face, Dropbox or Google Drive file")
        else:
            ok("weights link")
        if not re.fullmatch(r"[0-9a-f]{64}", sha) or set(sha) == {"0"}:
            warn(f"{LINK_FILE}: no valid `sha256:` (strongly recommended: sha256sum of the downloaded file)")
    elif not real:
        err("no weights: put them in weights/ (small models) or describe the download in weights/weights_link.yaml")
    else:
        ok(f"weights committed in weights/ ({len(real)} file(s))")
    for f in real:
        p = os.path.join(wdir, f)
        if os.path.isfile(p):
            with open(p, "rb") as fh:
                if fh.read(60).startswith(b"version https://git-lfs"):
                    err(f"weights/{f} is a Git LFS pointer, not the weights: Git LFS is not supported, use a download link")


def check_tree(pkg):
    total = 0
    abs_paths = []
    unpinned = []
    for root, dirs, files in os.walk(pkg):
        dirs[:] = [d for d in dirs if d not in (".git", "__pycache__")]
        for n in dirs + files:
            p = os.path.join(root, n)
            if os.path.islink(p):
                err(f"symlinks are not allowed: {os.path.relpath(p, pkg)}")
        for n in files:
            p = os.path.join(root, n)
            if os.path.islink(p):
                continue
            total += os.path.getsize(p)
            if n.endswith(".py") and n != "repro_check.py":
                try:
                    text = open(p, encoding="utf-8", errors="ignore").read()
                except OSError:
                    continue
                if re.search(r"""['"](/home/|/Users/|/data/|[A-Za-z]:\\\\)""", text):
                    abs_paths.append(os.path.relpath(p, pkg))
                if re.search(r"\b(requests\.(get|post)|urllib\.request|urlopen|wget |curl |gdown|snapshot_download|hf_hub_download)\b", text):
                    warn(f"{os.path.relpath(p, pkg)} seems to use the network; the reproduction runs with NO network "
                         "(weights must come from weights/)")
    if total > MAX_PACKAGE_BYTES:
        err(f"the repository is {total / 1024 ** 3:.1f} GB, over the {MAX_PACKAGE_BYTES / 1024 ** 3:.0f} GB cap: put big weights behind a link")
    for p in sorted(set(abs_paths)):
        warn(f"{p} contains absolute paths (/home, /Users, /data, C:\\): the code must run from any folder")
    req = os.path.join(pkg, "requirements.txt")
    if not os.path.isfile(req):
        warn("no requirements.txt: only the packages of the base image will be available")
    else:
        for line in open(req, encoding="utf-8", errors="ignore"):
            s = line.split("#", 1)[0].strip()
            if s and not s.startswith("-") and "==" not in s and "@" not in s:
                unpinned.append(s)
        if unpinned:
            warn("requirements.txt: pin exact versions (name==x.y.z): " + ", ".join(unpinned))


# ------------------------------------------------------------------ dynamic checks
def _load_module(pkg, filename, modname):
    for extra in (pkg, os.path.join(pkg, "src")):
        if extra not in sys.path:
            sys.path.insert(0, extra)
    spec = importlib.util.spec_from_file_location(modname, os.path.join(pkg, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _seed():
    random.seed(0)
    os.environ.setdefault("PYTHONHASHSEED", "0")
    try:
        import numpy as np
        np.random.seed(0)
    except Exception:
        pass
    try:
        import torch
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def _scoring_size(w, h, cap=MAX_SIDE_WARN):
    s = cap / max(w, h)
    return (w, h) if s >= 1 else (max(1, round(w * s)), max(1, round(h * s)))


def check_run(pkg, input_dir, n, output_dir, device):
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        err("--run needs numpy and pillow installed (they are in the base image)")
        return
    names = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(IMG_EXTS))
    if not names:
        err(f"no images (.png/.jpg) found in {input_dir}")
        return
    step = max(1, len(names) // n)
    chosen = names[::step][:n]
    tmp_in = tempfile.mkdtemp(prefix="reprocheck_in_")
    for f in chosen:
        shutil.copy(os.path.join(input_dir, f), os.path.join(tmp_in, f))
    out = output_dir or tempfile.mkdtemp(prefix="reprocheck_out_")
    os.makedirs(out, exist_ok=True)
    if device == "cuda":
        try:
            import torch
            if not torch.cuda.is_available():
                warn("device cuda requested but not available here: running on cpu")
                device = "cpu"
        except ImportError:
            device = "cpu"
    _seed()
    try:
        mod = _load_module(pkg, "run.py", "team_run")
        t0 = time.time()
        mod.main(model_dir=os.path.join(pkg, "weights"), input_dir=tmp_in, output_dir=out, device=device)
        dt = time.time() - t0
    except Exception as exc:
        err(f"run.py main() failed: {type(exc).__name__}: {exc}")
        return
    ok(f"run.py main() ran on {len(chosen)} image(s) in {dt:.1f} s ({dt / len(chosen):.2f} s/image including start-up)")

    stems = [os.path.splitext(f)[0] for f in chosen]
    hashes = {}
    for f, stem in zip(chosen, stems):
        rp = os.path.join(out, "restored", stem + ".png")
        mp = os.path.join(out, "seg_masks", stem + ".png")
        if not os.path.isfile(rp):
            err(f"missing output restored/{stem}.png")
            continue
        if not os.path.isfile(mp):
            err(f"missing output seg_masks/{stem}.png")
            continue
        with Image.open(os.path.join(input_dir, f)) as im:
            exp = _scoring_size(*im.size)
        with Image.open(rp) as r, Image.open(mp) as m:
            if r.mode != "RGB":
                err(f"restored/{stem}.png must be RGB, found {r.mode}")
            if m.mode not in ("L", "P"):
                err(f"seg_masks/{stem}.png must be single-channel (mode L), found {m.mode}")
            if r.size != m.size:
                err(f"{stem}: restored {r.size} and mask {m.size} must have the same size")
            if max(r.size) > MAX_SIDE_ERR:
                err(f"{stem}: output longest side {max(r.size)} px is over {MAX_SIDE_ERR}")
            elif r.size != exp:
                warn(f"{stem}: output size {r.size}, the scoring size for this input is {exp} "
                     f"(longest side at most {MAX_SIDE_WARN} px, aspect kept); other sizes are resized by the evaluator")
            vals = np.unique(np.asarray(m))
            if vals.size and (vals.min() < 0 or vals.max() >= NUM_CLASSES):
                err(f"seg_masks/{stem}.png has labels outside 0..{NUM_CLASSES - 1}: {vals.tolist()}")
            hashes[stem] = (hash(np.asarray(r).tobytes()), hash(np.asarray(m).tobytes()))
    if len(set(hashes.values())) < len(hashes) and len(hashes) > 1:
        warn("two different inputs produced identical outputs: every output must come from its own input image only")
    extra = sorted(set(os.listdir(os.path.join(out, "restored"))) - {s + ".png" for s in stems}) \
        if os.path.isdir(os.path.join(out, "restored")) else []
    if extra:
        warn(f"restored/ has files that do not belong to the {len(chosen)} inputs: {extra[:3]}")
    shutil.rmtree(tmp_in, ignore_errors=True)
    if output_dir is None:
        shutil.rmtree(out, ignore_errors=True)


def check_export(pkg):
    try:
        import onnx
    except ImportError:
        err("--export-onnx needs the onnx package (it is in the base image)")
        return
    tmp = tempfile.mkdtemp(prefix="reprocheck_onnx_")
    target = os.path.join(tmp, "model.onnx")
    _seed()
    try:
        mod = _load_module(pkg, "export_onnx.py", "team_export")
        mod.main(model_dir=os.path.join(pkg, "weights"), out_onnx=target)
    except Exception as exc:
        err(f"export_onnx.py main() failed: {type(exc).__name__}: {exc}")
        return
    if not os.path.isfile(target):
        err("export_onnx.py main() did not write the ONNX file to out_onnx")
        return
    model = onnx.load(target)
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        err(f"onnx.checker failed: {exc}")
    inits = {i.name for i in model.graph.initializer}
    inputs = [i for i in model.graph.input if i.name not in inits]
    if len(inputs) != 1:
        err(f"the ONNX model must have exactly one input, found {len(inputs)}")
    else:
        dims = [d.dim_value if d.HasField("dim_value") else None for d in inputs[0].type.tensor_type.shape.dim]
        if len(dims) != 4 or None in dims or dims[0] != 1 or dims[1] != 3:
            err(f"the input must have a static shape (1, 3, H, W), found {dims}")
        else:
            ok(f"static input shape {tuple(dims)}")
    outs = [o.name for o in model.graph.output]
    if outs != ["restored_image", "seg_logits"]:
        err(f"outputs must be named restored_image, seg_logits (in this order), found {outs}")
    else:
        ok("output names")
    params = 0
    for init in model.graph.initializer:
        n = 1
        for d in init.dims:
            n *= d
        params += n
    if params > 1_000_000:
        err(f"{params / 1e6:.2f} M parameters, over the 1.0 M limit")
    else:
        ok(f"{params / 1e6:.3f} M parameters (limit 1.0 M); GFLOPs (limit 12) are measured by the organizers' pipeline")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("package", nargs="?", default=".")
    ap.add_argument("--run", action="store_true", help="run main() on a few images and check the outputs")
    ap.add_argument("--input", help="folder of degraded images for --run")
    ap.add_argument("--n", type=int, default=5, help="how many images --run uses (default 5)")
    ap.add_argument("--output", help="keep --run outputs here (default: temporary folder)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--export-onnx", action="store_true", help="Track 2: run export_onnx.py and check the file")
    args = ap.parse_args()

    pkg = os.path.abspath(args.package)
    if not os.path.isdir(pkg):
        print(f"{pkg} is not a folder")
        return 2
    print(f"checking {pkg}\n")
    spec = check_repro_yaml(pkg)
    track = (spec or {}).get("track")
    check_method(pkg)
    check_entry_points(pkg, track)
    check_weights(pkg)
    check_tree(pkg)
    if args.run:
        if not args.input or not os.path.isdir(args.input):
            err("--run needs --input DIR with some of the degraded images")
        else:
            check_run(pkg, args.input, args.n, args.output, args.device)
    if args.export_onnx:
        check_export(pkg)
    print()
    print(f"{len(errors)} error(s), {len(warnings)} warning(s)")
    print("READY to hand in (as far as this check can tell)" if not errors else "NOT READY: fix the errors above")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
