"""Colab bootstrap for INF8225 polyp segmentation project.

Called from the first cell of each notebook. No-op when not on Colab.
Mounts Drive, pins a known-good OpenMMLab stack, and symlinks heavy assets
from Drive into the paths the notebooks use.
"""
from __future__ import annotations

import importlib.util
from importlib import metadata
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .drive_paths import (
    DRIVE_CANDIDATES,
    OUTPUT_PIPELINES,
    STANDARD_OUTPUT_SUBDIRS,
    ensure_drive_layout,
)

DRIVE_FOLDER_ID = "1BAcGyja2SHP3t2OFOleN2cND4QlXb3fW"
DRIVE_FOLDER_URL = f"https://drive.google.com/drive/folders/{DRIVE_FOLDER_ID}"

# local path in the repo  ->  path relative to the Drive project folder
SYMLINK_MAP = {
    "data": "data",
    "work_dirs": "work_dirs",
    "outputs": "outputs",
    "MedSAM/work_dir": "work_dir",
    "models_weights": "models_weights",
}

COPY_TO_DRIVE_MAP = {}

REPLACE_WITH_SYMLINK = {
    "data",
    "work_dirs",
    "outputs",
    "MedSAM/work_dir",
    "models_weights",
}

OUTPUT_SUBDIRS = [
    "work_dirs/polyp_config_v2",
    "work_dirs/tumor_config_v3",
    "work_dirs/pancreas_unet",
    "work_dir/MedSAM",
    "models_weights",
]
for implementation, pipelines in OUTPUT_PIPELINES.items():
    for pipeline in pipelines:
        OUTPUT_SUBDIRS.append(f"outputs/{implementation}/{pipeline}")
        for subdir in STANDARD_OUTPUT_SUBDIRS:
            OUTPUT_SUBDIRS.append(f"outputs/{implementation}/{pipeline}/{subdir}")

# Colab's preinstalled torch can jump ahead of OpenMMLab wheel support
# (for example, torch 2.10 + CUDA 12.8). Installing a known-good stack is
# more reliable than trying to match whatever Colab happened to ship today.
SUPPORTED_STACKS = {
    "gpu": {
        "label": "GPU / CUDA 12.1",
        "torch": "2.4.0",
        "torchvision": "0.19.0",
        "torchaudio": "2.4.0",
        "torch_index": "https://download.pytorch.org/whl/cu121",
        "mmcv": "2.2.0",
        "mmcv_index": (
            "https://download.openmmlab.com/mmcv/dist/cu121/torch2.4.0/index.html"
        ),
    },
    "cpu": {
        "label": "CPU",
        "torch": "2.3.1",
        "torchvision": "0.18.1",
        "torchaudio": "2.3.1",
        "torch_index": "https://download.pytorch.org/whl/cpu",
        "mmcv": "2.2.0",
        "mmcv_index": (
            "https://download.openmmlab.com/mmcv/dist/cpu/torch2.3.0/index.html"
        ),
    },
}

# Keep the scientific Python stack deterministic on Colab. Newer base images can
# otherwise pull in a NumPy/SciPy pair that is import-incompatible with MMDetection
# in the same runtime, especially after partial in-session upgrades.
SCIENTIFIC_STACK = {
    "numpy": "1.26.4",
    "scipy": "1.13.1",
    "matplotlib": "3.8.4",
}

MM_RUNTIME = {
    "mmengine": "0.10.7",
    "mmdet": "3.3.0",
}


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
        if c.exists():
            return c
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


def _dist_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _has_gpu() -> bool:
    return _run(["nvidia-smi", "-L"], "detect GPU", check=False).returncode == 0


def _target_stack() -> dict[str, str]:
    return SUPPORTED_STACKS["gpu" if _has_gpu() else "cpu"]


def _torch_stack_matches(stack: dict[str, str]) -> bool:
    versions = {
        "torch": _dist_version("torch"),
        "torchvision": _dist_version("torchvision"),
        "torchaudio": _dist_version("torchaudio"),
    }
    return all(
        versions[pkg] is not None and versions[pkg].startswith(stack[pkg])
        for pkg in versions
    )


def _versions_match(required: dict[str, str]) -> bool:
    versions = {pkg: _dist_version(pkg) for pkg in required}
    return all(
        versions[pkg] is not None and versions[pkg].startswith(required[pkg])
        for pkg in required
    )


def _ensure_torch_stack(pip: list[str], stack: dict[str, str]) -> None:
    if _torch_stack_matches(stack):
        print(f"✓ torch stack already compatible ({stack['label']})")
        return

    if "torch" in sys.modules:
        raise RuntimeError(
            "torch was imported before setup() could pin a compatible stack.\n"
            "Restart the Colab runtime, then rerun the first notebook cell."
        )

    _run(
        pip
        + [
            "--force-reinstall",
            "--no-cache-dir",
            "--index-url",
            stack["torch_index"],
            f"torch=={stack['torch']}",
            f"torchvision=={stack['torchvision']}",
            f"torchaudio=={stack['torchaudio']}",
        ],
        f"torch stack ({stack['label']})",
    )


def _deps_installed(stack: dict[str, str]) -> bool:
    """True iff the pinned stack is present and deep MMDetection imports work."""
    required = {
        **SCIENTIFIC_STACK,
        **MM_RUNTIME,
        "mmcv": stack["mmcv"],
    }
    if not _versions_match(required):
        return False

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib;"
                "importlib.import_module('transformers');"
                "from mmdet.apis import init_detector, inference_detector;"
                "from mmdet.utils import register_all_modules;"
                "from mmcv.ops import nms;"
                "from scipy.sparse import csr_matrix"
            ),
        ],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        print("↺ existing MMDetection imports failed, reinstalling pinned deps")
        return False
    return True


def _install_mmcv(pip: list[str], stack: dict[str, str]) -> str:
    _run(
        pip
        + [
            f"mmcv=={stack['mmcv']}",
            "-f",
            stack["mmcv_index"],
            "--only-binary",
            "mmcv",
        ],
        f"mmcv=={stack['mmcv']} ({stack['label']})",
    )
    return stack["mmcv"]


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
    pip = [sys.executable, "-m", "pip", "install", "-q"]
    stack = _target_stack()

    if not reinstall and _deps_installed(stack):
        print("✓ deps already installed, skipping")
        return

    # Colab's Py 3.12 ships a setuptools whose pkg_resources uses
    # pkgutil.ImpImporter (removed in 3.12). Force-reinstall to overwrite it.
    _run(
        pip + ["--force-reinstall", "--no-deps", "-U", "setuptools>=70", "wheel"],
        "setuptools / wheel (Py 3.12 fix)",
    )

    _ensure_torch_stack(pip, stack)
    _run(pip + [f"mmengine=={MM_RUNTIME['mmengine']}"], f"mmengine=={MM_RUNTIME['mmengine']}")

    installed_mmcv = _install_mmcv(pip, stack)

    # --no-deps: mmdet's setup.py pins mmcv<2.2.0, which would make the resolver
    # fight us on Py 3.12. We install mmdet alone, then handle its runtime deps.
    _run(
        pip + ["--no-deps", f"mmdet=={MM_RUNTIME['mmdet']}"],
        f"mmdet=={MM_RUNTIME['mmdet']} (no-deps)",
    )
    _patch_mmdet_mmcv_ceiling(installed_mmcv)
    _run(
        pip + [
            "pycocotools", "shapely", "terminaltables", "tabulate",
        ],
        "mmdet runtime deps",
    )

    reqs = Path(__file__).resolve().parent / "requirements-colab.txt"
    if reqs.exists():
        _run(pip + ["-r", str(reqs)], "project extras (transformers, nltk, …)")

    # Pin scientific stack LAST so nothing above (pycocotools, transformers,
    # fairscale, …) can pull numpy 2.x back in. `pip uninstall` alone can
    # leave stale .so files from a broken 2.x install next to 1.26.4 .py
    # files, yielding `numpy.dtype size changed (expected 96, got 88)` — a
    # 2.x-vs-1.x ABI mismatch inside numpy itself. Nuke the package dirs
    # outright so the reinstall starts from a clean slate.
    pip_uninstall = [sys.executable, "-m", "pip", "uninstall", "-y", "-q"]
    _run(
        pip_uninstall + list(SCIENTIFIC_STACK.keys()),
        "remove stale numpy / scipy / matplotlib",
        check=False,
    )
    import site
    import shutil
    for pkg_name in SCIENTIFIC_STACK:
        for site_dir in site.getsitepackages():
            pkg_path = Path(site_dir) / pkg_name
            if pkg_path.exists():
                shutil.rmtree(pkg_path, ignore_errors=True)
                print(f"  rm -rf {pkg_path}")
    _run(
        pip
        + [
            "--no-cache-dir",
            *(f"{pkg}=={version}" for pkg, version in SCIENTIFIC_STACK.items()),
        ],
        "numpy / scipy / matplotlib (pinned last, clean install)",
    )


def _link(local: Path, target: Path, replace_existing: bool = False) -> None:
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
        if not replace_existing:
            print(f"⚠  {local} already exists as real file/dir — leaving untouched")
            return
        backup = local.with_name(f"{local.name}_repo")
        if backup.exists():
            if local.is_dir():
                shutil.rmtree(local)
            else:
                local.unlink()
            print(f"✓  removed existing {local} (backup already present)")
        else:
            shutil.move(str(local), str(backup))
            print(f"✓  moved existing {local} → {backup}")
    os.symlink(target, local)
    print(f"✓  linked {local} → {target}")


def _copy_to_drive(local: Path, target: Path) -> None:
    if not local.exists():
        print(f"⚠  local config missing — skipping {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"✓  {target} already present")
        return
    shutil.copy2(local, target)
    print(f"✓  copied {local} → {target}")


def _make_output_dirs(drive_folder: Path) -> None:
    for rel in OUTPUT_SUBDIRS:
        (drive_folder / rel).mkdir(parents=True, exist_ok=True)


def _print_summary(project_root: Path, drive_root: Path) -> None:
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
        torch_version = torch.__version__
    except ImportError:
        gpu = "?"
        torch_version = "?"
    print()
    print(f"Project : {project_root}")
    print(f"Drive   : {drive_root}")
    print(f"Device  : {gpu}")
    print(f"Torch   : {torch_version}")


def setup(
    drive_folder: str | None = None,
    install: bool = True,
    reinstall: bool = False,
) -> None:
    """Run once per Colab session at the top of a notebook."""
    if not is_colab():
        print("Not running in Colab — setup skipped.")
        return

    project_root = Path.cwd()
    expected_paths = [
        project_root / "colab" / "setup.py",
        project_root / "kvasir_implementation",
        project_root / "msd_implementation",
    ]
    if not all(path.exists() for path in expected_paths):
        raise RuntimeError(
            f"{project_root} does not look like INF8225_Projet. "
            "Clone the repo and %cd into it before calling setup()."
        )

    _mount_drive()
    if install:
        _install_deps(reinstall=reinstall)

    drive_root = _find_drive_folder(drive_folder)
    os.environ["INF8225_DRIVE_ROOT"] = str(drive_root)
    os.environ["INF8225_OUTPUTS_ROOT"] = str(drive_root / "outputs")
    ensure_drive_layout(drive_root)
    _make_output_dirs(drive_root)
    for local_rel, drive_rel in COPY_TO_DRIVE_MAP.items():
        _copy_to_drive(project_root / local_rel, drive_root / drive_rel)
    for local_rel, drive_rel in SYMLINK_MAP.items():
        _link(
            project_root / local_rel,
            drive_root / drive_rel,
            replace_existing=local_rel in REPLACE_WITH_SYMLINK,
        )

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # MedSAM's vendored segment_anything uses absolute imports
    # (`from segment_anything.modeling import Sam`), so the MedSAM folder
    # itself needs to be on sys.path — not just the repo root.
    medsam_dir = project_root / "MedSAM"
    if medsam_dir.is_dir() and str(medsam_dir) not in sys.path:
        sys.path.insert(0, str(medsam_dir))

    # Run nltk downloads in a subprocess: if the kernel already had numpy
    # preloaded (Colab's default), the in-memory numpy can mismatch the
    # freshly-installed on-disk numpy ABI. A subprocess uses the clean
    # on-disk install and can't propagate ValueError into setup().
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import nltk;"
                "nltk.download('punkt_tab', quiet=True);"
                "nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
            ),
        ],
        check=False,
    )

    _print_summary(project_root, drive_root)

    # If numpy was imported before setup reinstalled it, the kernel still has
    # the stale module and the next `import mmdet` will fail with an ABI
    # ValueError. Warn loudly so the user restarts before continuing.
    if install and "numpy" in sys.modules:
        print()
        print("⚠  numpy was already loaded in this kernel before setup reinstalled it.")
        print("   Runtime → Restart session, then run your imports again")
        print("   (no need to rerun setup — deps are pinned on disk).")
