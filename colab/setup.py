"""Colab bootstrap for INF8225 polyp segmentation project.

Called from the first cell of each notebook. No-op when not on Colab.
Mounts Drive, installs minimal deps, and symlinks heavy assets from Drive
into the paths the notebooks already use (data/, work_dirs/, MedSAM/work_dir/).
"""
from __future__ import annotations

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


def _run(cmd: list[str], desc: str) -> None:
    """Run a subprocess; on failure, surface the tail of stdout/stderr (which mim/pip -q swallow)."""
    print(f"→ {desc}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        tail_out = "\n".join((proc.stdout or "").splitlines()[-40:])
        tail_err = "\n".join((proc.stderr or "").splitlines()[-40:])
        raise RuntimeError(
            f"\ncommand failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"--- stdout (last 40 lines) ---\n{tail_out}\n"
            f"--- stderr (last 40 lines) ---\n{tail_err}\n"
        )


def _install_deps(reinstall: bool = False) -> None:
    if not reinstall:
        try:
            import mmdet, mmcv, transformers  # noqa: F401
            print("✓ deps already installed, skipping")
            return
        except ImportError:
            pass

    pip = [sys.executable, "-m", "pip", "install", "-q"]

    # Colab Python 3.12 ships a setuptools whose pkg_resources uses the
    # removed pkgutil.ImpImporter. Any tool that imports setuptools crashes.
    # Force-reinstall to overwrite it; plain -U skips because pip considers
    # the pinned-by-Colab version "already up to date".
    _run(pip + ["--force-reinstall", "-U", "setuptools>=70", "wheel"],
         "setuptools / wheel (Py 3.12 fix)")

    # mmengine and mmdet are pure-Python — pip direct is simpler and avoids
    # openmim, which is itself broken by the setuptools issue above.
    _run(pip + ["mmengine==0.10.7"], "mmengine==0.10.7")

    # mmcv has compiled CUDA ops → prefer a pre-built wheel matching Colab's torch.
    index = _mmcv_wheel_index()
    if index is not None:
        _run(pip + ["mmcv==2.1.0", "-f", index],
             f"mmcv==2.1.0 (prebuilt wheel @ {index})")
    else:
        _run(pip + ["mmcv==2.1.0"], "mmcv==2.1.0 (source build, slow)")

    _run(pip + ["mmdet==3.3.0"], "mmdet==3.3.0")

    reqs = Path(__file__).resolve().parent / "requirements-colab.txt"
    _run(pip + ["-r", str(reqs)], "extras (transformers, nltk, ...)")


def _mmcv_wheel_index() -> str | None:
    """OpenMMLab prebuilt-wheel index matching Colab's torch/CUDA, or None."""
    try:
        import torch
    except ImportError:
        return None
    t = ".".join(torch.__version__.split("+")[0].split(".")[:2])  # "2.5"
    cu = getattr(torch.version, "cuda", None)
    if not cu:
        return None
    cu_tag = "cu" + cu.replace(".", "")[:3]  # "121" / "124"
    return f"https://download.openmmlab.com/mmcv/dist/{cu_tag}/torch{t}/index.html"


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
