"""Colab bootstrap for INF8225 polyp segmentation project.

Called from the first cell of each notebook. No-op when not on Colab.
Mounts Drive, installs MM* deps (without ever triggering a source build),
and symlinks heavy assets from Drive into the paths the notebooks use.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

DRIVE_FOLDER_ID = "1BAcGyja2SHP3t2OFOleN2cND4QlXb3fW"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

# local path in the repo  ->  path relative to the Drive project folder
SYMLINK_MAP = {
    "data": "data",
    "work_dirs": "work_dirs",
    "MedSAM/work_dir": "MedSAM/work_dir",
    "grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth":
        "grounding_dino_swin-t_pretrain_obj365_goldg_20231122_132602-4ea751ce.pth",
}

# Places the user is likely to have dropped the shared folder.
DRIVE_CANDIDATES = [
    "/content/drive/MyDrive/Projet_Medsam",
    "/content/drive/MyDrive/INF8225_Projet",
    "/content/drive/MyDrive/INF8225",
    "/content/drive/MyDrive/Colab Notebooks/Projet_Medsam",
]

# Output subdirs the notebooks write to — created on Drive so results persist.
OUTPUT_SUBDIRS = [
    "data/outputs",
    "data/outputs_medsam",
    "data/outputs_medsam_dino",
    "data/results",
    "work_dirs/polyp_config",
]


def is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _mount_drive() -> None:
    from google.colab import drive  # type: ignore
    if not os.path.ismount("/content/drive"):
        drive.mount("/content/drive")


def _find_drive_folder(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        raise FileNotFoundError(f"Drive folder not found at {explicit}")
    for c in DRIVE_CANDIDATES:
        if Path(c).exists():
            return Path(c)
    raise FileNotFoundError(
        "\nProject folder not found under /content/drive/MyDrive.\n"
        f"1. Open {DRIVE_FOLDER_URL}\n"
        "2. Click 'Add shortcut to Drive' → 'My Drive'.\n"
        "   or pass the path explicitly: setup(drive_folder='/content/drive/MyDrive/your_folder')\n"
    )


def _run(cmd: list[str], desc: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess; on failure, surface the tail of stdout/stderr."""
    print(f"→ {desc}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and check:
        tail_out = "\n".join((proc.stdout or "").splitlines()[-40:])
        tail_err = "\n".join((proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(
            f"\ncommand failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout (last 40 lines) ---\n{tail_out}\n"
            f"--- stderr (last 40 lines) ---\n{tail_err}\n"
        )
    return proc


def _torch_meta() -> tuple[str, str] | None:
    """Return ('<maj>.<min>', 'cuNNN') for current torch build, or None."""
    try:
        import torch
    except ImportError:
        return None
    mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])
    cu = getattr(torch.version, "cuda", None)
    if not cu:
        return None
    return mm, "cu" + cu.replace(".", "")[:3]


def _deps_installed() -> bool:
    """True iff every MM* package imports AND mmcv's CUDA ops load."""
    try:
        for pkg in ("mmengine", "mmcv", "mmdet", "transformers"):
            importlib.import_module(pkg)
        from mmcv.ops import nms  # noqa: F401  triggers _ext import
        return True
    except Exception:
        return False


def _install_mmcv(pip: list[str]) -> str:
    """Install mmcv from a prebuilt wheel only (never source build).

    Returns the version string that was installed (e.g. "2.2.0").

    Strategy:
      - on py≤3.11, prefer the strictly-compatible mmcv 2.1.0 wheel
      - on py3.12+, OpenMMLab only ships mmcv ≥2.2.0 wheels; we install
        mmcv 2.2.x and patch mmdet's version ceiling afterwards
      - walk down a list of torch tags so minor torch/mmcv ABI skew is handled
    """
    meta = _torch_meta()
    if meta is None:
        raise RuntimeError(
            "CUDA-enabled torch not detected. Switch Colab runtime to a GPU "
            "(Runtime → Change runtime type → T4/A100/L4)."
        )
    torch_mm, cu = meta

    candidates: list[tuple[str, str]] = []
    if sys.version_info < (3, 12):
        for t in (torch_mm, "2.1", "2.0"):
            candidates.append(("2.1.0", t))
    # mmcv 2.2.0 ships cp312 wheels for torch 2.1 → 2.4.
    for t in (torch_mm, "2.4", "2.3", "2.2", "2.1"):
        candidates.append(("2.2.0", t))

    tried: list[str] = []
    for ver, torch_tag in candidates:
        url = f"https://download.openmmlab.com/mmcv/dist/{cu}/torch{torch_tag}.0/index.html"
        label = f"mmcv=={ver} (torch{torch_tag}/{cu})"
        tried.append(label)
        proc = _run(
            pip + [f"mmcv=={ver}", "-f", url, "--only-binary", "mmcv"],
            f"try {label}",
            check=False,
        )
        if proc.returncode == 0:
            return ver

    raise RuntimeError(
        "No prebuilt mmcv wheel matched this runtime.\n"
        "Tried:\n  - " + "\n  - ".join(tried) + "\n"
        "Workarounds:\n"
        "  - Runtime → Restart session (Colab may have switched torch/CUDA)\n"
        "  - or pin torch first:  !pip install -q torch==2.4.0 torchvision==0.19.0 "
        "--index-url https://download.pytorch.org/whl/cu121\n"
        "  - then rerun this cell."
    )


def _patch_mmdet_mmcv_ceiling(installed_mmcv: str) -> None:
    """mmdet 3.3.0 asserts `mmcv < 2.2.0` at import. Bump the ceiling when needed."""
    if installed_mmcv.startswith("2.1."):
        return
    spec = importlib.util.find_spec("mmdet")
    if spec is None or spec.origin is None:
        return
    init = Path(spec.origin)
    src = init.read_text()
    patched = src.replace(
        "mmcv_maximum_version = '2.2.0'",
        "mmcv_maximum_version = '3.0.0'",
    )
    if patched != src:
        init.write_text(patched)
        print(f"✓  patched mmdet ceiling → accepts mmcv {installed_mmcv}")


def _install_deps(reinstall: bool = False) -> None:
    if not reinstall and _deps_installed():
        print("✓ deps already installed, skipping")
        return

    pip = [sys.executable, "-m", "pip", "install", "-q"]

    # Colab's Py 3.12 ships a setuptools whose pkg_resources uses
    # pkgutil.ImpImporter (removed in 3.12). Force-reinstall to overwrite it.
    _run(
        pip + ["--force-reinstall", "--no-deps", "-U", "setuptools>=70", "wheel"],
        "setuptools / wheel (Py 3.12 fix)",
    )

    _run(pip + ["mmengine==0.10.7"], "mmengine==0.10.7")

    installed_mmcv = _install_mmcv(pip)

    # --no-deps: mmdet's setup.py pins mmcv<2.2.0, which would make the resolver
    # fight us on Py 3.12. We install mmdet alone, then handle its runtime deps.
    _run(pip + ["--no-deps", "mmdet==3.3.0"], "mmdet==3.3.0 (no-deps)")
    _patch_mmdet_mmcv_ceiling(installed_mmcv)
    _run(
        pip + [
            "pycocotools", "shapely", "terminaltables", "tabulate",
            "scipy", "matplotlib",
        ],
        "mmdet runtime deps",
    )

    reqs = Path(__file__).resolve().parent / "requirements-colab.txt"
    if reqs.exists():
        _run(pip + ["-r", str(reqs)], "project extras (transformers, nltk, …)")


def _link(local: Path, target: Path) -> None:
    if not target.exists():
        print(f"⚠  {target} missing on Drive — skipping {local}")
        return
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.is_symlink():
        if local.resolve() == target.resolve():
            print(f"✓  {local} already linked")
            return
        local.unlink()
    elif local.exists():
        print(f"⚠  {local} already exists as real file/dir — leaving untouched")
        return
    os.symlink(target, local)
    print(f"✓  linked {local} → {target}")


def _make_output_dirs(drive_folder: Path) -> None:
    for rel in OUTPUT_SUBDIRS:
        (drive_folder / rel).mkdir(parents=True, exist_ok=True)


def _print_summary(project_root: Path, drive_root: Path) -> None:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    except ImportError:
        gpu = "?"
    print()
    print(f"Project : {project_root}")
    print(f"Drive   : {drive_root}")
    print(f"Device  : {gpu}")


def setup(drive_folder: str | None = None, install: bool = True) -> None:
    """Run once per Colab session at the top of a notebook."""
    if not is_colab():
        print("Not running in Colab — setup skipped.")
        return

    project_root = Path.cwd()
    if not (project_root / "polyp_config.py").exists():
        raise RuntimeError(
            f"{project_root} does not look like INF8225_Projet (polyp_config.py missing). "
            "Clone the repo and %cd into it before calling setup()."
        )

    _mount_drive()
    if install:
        _install_deps()

    drive_root = _find_drive_folder(drive_folder)
    _make_output_dirs(drive_root)
    for local_rel, drive_rel in SYMLINK_MAP.items():
        _link(project_root / local_rel, drive_root / drive_rel)

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        nltk.download("averaged_perceptron_tagger_eng", quiet=True)
    except ImportError:
        pass

    _print_summary(project_root, drive_root)
