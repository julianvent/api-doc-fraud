"""Tampering detection engine wrappers."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np

_HERE = Path(__file__).resolve().parent
_WEIGHTS_DIR = _HERE.parent / "weights"
_VENDORED_ROOT = _HERE / "models"

_DEFAULT_DTD_WEIGHTS = _WEIGHTS_DIR / "dtd_doctamper.pth"
_DEFAULT_VPH_WEIGHTS = _WEIGHTS_DIR / "vph_imagenet.pt"
_DEFAULT_SWIN_WEIGHTS = _WEIGHTS_DIR / "swin_imagenet.pt"
_DEFAULT_QT_TABLE = _WEIGHTS_DIR / "qt_table.pk"
_DEFAULT_MVSS_WEIGHTS = _WEIGHTS_DIR / "mvssnet_casia.pt"

# MVSS-Net input spatial size — matches the upstream demo / inference script.
_MVSS_INPUT_SIZE = 512

# Model input spatial size. DocTamper was trained on 512x512 crops; we
# resize inputs to this resolution to stay aligned with the training
# distribution (Swin position biases, FPH block assumptions).
_MODEL_INPUT_SIZE = 512

# JPEG quality used to extract DCT coefficients at inference time. 75 is the
# default lower bound used in the upstream Colab demo and matches the model's
# robustness range. Higher quality preserves more detail but narrows the
# distribution the model was exposed to during training.
_JPEG_QUALITY = 85

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class TamperingEngine(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run detection on a RGB image.

        Returns (heatmap, score):
            heatmap: HxW float32 in [0, 1], same spatial size as the input.
            score:   float in [0, 1], 0 = clean / 1 = tampered.
        """
        ...


class MockEngine:
    """Low-uniform heatmap plus deterministic noise. Exercises the pipeline only."""

    name = "mock-clean"

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        h, w = image.shape[:2]
        rng = np.random.default_rng(seed=int(image.mean()))
        heatmap = np.clip(
            0.05 + rng.normal(0, 0.02, size=(h, w)).astype(np.float32),
            0.0, 1.0,
        )
        return heatmap, float(min(heatmap.mean() * 1.5, 1.0))


class DocTamperEngine:
    """Wrapper around the official DocTamper (DTD) model for document tampering.

    The model takes three inputs: an ImageNet-normalized RGB image, the JPEG
    quantized DCT coefficients of the grayscale channel, and the JPEG
    quantization table used to produce those coefficients.
    """

    name = "doctamper"

    def __init__(
        self,
        dtd_weights: Path = _DEFAULT_DTD_WEIGHTS,
        vph_weights: Path = _DEFAULT_VPH_WEIGHTS,
        swin_weights: Path = _DEFAULT_SWIN_WEIGHTS,
        qt_table_path: Path = _DEFAULT_QT_TABLE,
        device: str = "cpu",
        jpeg_quality: int = _JPEG_QUALITY,
        input_size: int = _MODEL_INPUT_SIZE,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is not installed. Run: pip install -e '.[doctamper]'"
            ) from exc

        for path in (dtd_weights, vph_weights, swin_weights, qt_table_path):
            if not Path(path).is_file():
                raise RuntimeError(f"DocTamper asset not found: {path}")

        self._torch = torch
        self._device = torch.device(device)
        self._jpeg_quality = int(jpeg_quality)
        self._input_size = int(input_size)
        self._qt_table = self._load_qt_table(Path(qt_table_path))
        self._model = self._load_model(
            Path(dtd_weights), Path(vph_weights), Path(swin_weights),
        )

    def _load_qt_table(self, path: Path) -> np.ndarray:
        with path.open("rb") as fh:
            tables = pickle.load(fh)
        if self._jpeg_quality not in tables:
            raise RuntimeError(
                f"Quality {self._jpeg_quality} not in qt_table.pk "
                f"(available: {sorted(tables.keys())})"
            )
        return np.asarray(tables[self._jpeg_quality], dtype=np.int64)

    def _load_model(
        self, dtd_weights: Path, vph_weights: Path, swin_weights: Path,
    ):
        torch = self._torch
        _register_vendored_modules(_VENDORED_ROOT)
        from dtd import seg_dtd

        model = seg_dtd(
            model_name="",
            n_class=2,
            vph_path=str(vph_weights),
            swin_path=str(swin_weights),
            device=str(self._device),
        )
        checkpoint = torch.load(
            str(dtd_weights),
            map_location=self._device,
            weights_only=False,
        )
        state = checkpoint.get("state_dict", checkpoint)
        # Upstream checkpoints may be wrapped in DataParallel (`module.` prefix).
        state = {
            (k[len("module."):] if k.startswith("module.") else k): v
            for k, v in state.items()
        }
        model.load_state_dict(state, strict=False)
        _patch_legacy_submodules(model)
        model.eval().to(self._device)
        return model

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        torch = self._torch
        orig_h, orig_w = image.shape[:2]
        rgb_tensor, dct_tensor, qt_tensor = self._preprocess(image)

        with torch.no_grad():
            logits = self._model(rgb_tensor, dct_tensor, qt_tensor)
        probs = torch.softmax(logits, dim=1)[0, 1]
        heatmap = probs.cpu().numpy().astype(np.float32)

        if heatmap.shape != (orig_h, orig_w):
            heatmap = cv2.resize(
                heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
            )

        # Global score as the mean tampered probability: bounded in [0, 1] and
        # grows with both the size and confidence of suspicious regions.
        global_score = float(heatmap.mean())
        return heatmap, global_score

    def _preprocess(self, image_rgb: np.ndarray):
        torch = self._torch
        size = self._input_size
        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)

        rgb = resized.astype(np.float32) / 255.0
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        rgb_tensor = torch.from_numpy(
            np.ascontiguousarray(rgb.transpose(2, 0, 1))
        ).float().unsqueeze(0).to(self._device)

        gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)
        dct_map = _approximate_quantized_dct(
            gray, self._qt_table, self._jpeg_quality,
        )
        dct_map = np.clip(np.abs(dct_map), 0, 20).astype(np.int64)
        dct_tensor = torch.from_numpy(dct_map).long().unsqueeze(0).to(self._device)

        qt_tensor = (
            torch.from_numpy(self._qt_table).long()
            .unsqueeze(0).unsqueeze(0).to(self._device)
        )
        return rgb_tensor, dct_tensor, qt_tensor


def _register_vendored_modules(root: Path) -> None:
    """Expose vendored model code and legacy `timm.models.layers.*` paths
    under the names expected by upstream DocTamper code and its pickled
    checkpoints.
    """
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    _install_timm_legacy_aliases()

    from . import models as _models_pkg

    _alias_module("models", _models_pkg)

    loaded: dict[str, object] = {}
    for submodule_name in ("dtd", "swins", "fph"):
        try:
            loaded[submodule_name] = __import__(submodule_name)
        except ImportError:
            continue
        _alias_module(f"models.{submodule_name}", loaded[submodule_name])

    # Upstream pickles `vph_imagenet.pt` / `swin_imagenet.pt` were saved from
    # scripts run under __main__; re-point __main__ attributes to vendored
    # classes so the unpickler can resolve them.
    _merge_into_main(loaded.get("swins"))
    _merge_into_main(loaded.get("dtd"))
    _merge_into_main(loaded.get("fph"))


def _merge_into_main(module) -> None:
    if module is None:
        return
    import __main__
    for attr in dir(module):
        if attr.startswith("_"):
            continue
        if not hasattr(__main__, attr):
            setattr(__main__, attr, getattr(module, attr))


def _patch_legacy_submodules(model) -> None:
    """Backfill attributes that newer torch expects on legacy pickled modules.

    Pickles produced with torch < 2.0 lack fields that current activations
    like GELU read unconditionally. Iterating the module tree once and
    defaulting missing attributes keeps inference compatible without
    re-training.
    """
    import torch.nn as nn
    from timm.layers import DropPath

    for module in model.modules():
        if isinstance(module, nn.GELU) and not hasattr(module, "approximate"):
            module.approximate = "none"
        if isinstance(module, DropPath) and not hasattr(module, "scale_by_keep"):
            module.scale_by_keep = True


def _install_timm_legacy_aliases() -> None:
    """Alias pre-1.0 `timm.models.layers.*` paths to their current equivalents.

    The DocTamper checkpoints were pickled with timm 0.4.12, which exposed
    classes like DropPath under `timm.models.layers.drop`. In modern timm
    those live in `timm.layers`; the alias lets unpickling resolve them.
    """
    import timm.layers

    for legacy_path in (
        "timm.models.layers",
        "timm.models.layers.drop",
        "timm.models.layers.weight_init",
        "timm.models.layers.helpers",
    ):
        _alias_module(legacy_path, timm.layers)


def _alias_module(alias: str, module) -> None:
    if alias not in sys.modules:
        sys.modules[alias] = module


def _approximate_quantized_dct(
    gray: np.ndarray, qt_table: np.ndarray, quality: int,
) -> np.ndarray:
    """Approximate JPEG quantized DCT coefficients without `jpegio`.

    JPEG-encodes the grayscale input at `quality`, decodes it back (so pixels
    reflect quantization noise), and then computes 8x8 block-wise DCT-II and
    divides by the standard quantization table. This reproduces the statistics
    `jpegio` would read from the same encoded stream closely enough for the
    downstream FPH branch, even though it is not bit-identical.
    """
    from scipy.fft import dctn

    ok, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("Failed to JPEG-encode image for DCT extraction")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if decoded is None:
        raise RuntimeError("Failed to JPEG-decode image for DCT extraction")

    h, w = decoded.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    decoded = decoded[:h8, :w8]

    shifted = decoded.astype(np.float32) - 128.0
    blocks = shifted.reshape(h8 // 8, 8, w8 // 8, 8).swapaxes(1, 2)
    coefs = dctn(blocks, type=2, norm="ortho", axes=(-2, -1))
    quantized = np.round(coefs / qt_table.astype(np.float32))
    return quantized.swapaxes(1, 2).reshape(h8, w8)


class MVSSNetEngine:
    """Wrapper around MVSS-Net for pixel-level splicing detection.

    The model produces a per-pixel forgery probability map. The image-level
    score exposed to the pipeline is the maximum pixel probability — the
    upstream reference metric — which responds sharply to localized pasted
    regions and stays low on authentic crops.
    """

    name = "mvss-casia"

    def __init__(
        self,
        weights_path: Path = _DEFAULT_MVSS_WEIGHTS,
        device: str = "cpu",
        input_size: int = _MVSS_INPUT_SIZE,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "torch is not installed. Run: pip install -e '.[doctamper]'"
            ) from exc

        if not Path(weights_path).is_file():
            raise RuntimeError(f"MVSS-Net weights not found: {weights_path}")

        self._torch = torch
        self._device = torch.device(device)
        self._input_size = int(input_size)
        self._model = self._load_model(Path(weights_path))

    def _load_model(self, weights_path: Path):
        torch = self._torch
        _register_mvss_module(_VENDORED_ROOT)
        from .models.mvss.mvssnet import get_mvss

        model = get_mvss(
            backbone="resnet50", pretrained_base=False, nclass=1,
            sobel=True, constrain=True, n_input=3,
        )
        checkpoint = torch.load(
            str(weights_path), map_location=self._device, weights_only=True,
        )
        model.load_state_dict(checkpoint, strict=True)
        model.eval().to(self._device)
        return model

    def detect(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        torch = self._torch
        orig_h, orig_w = image_rgb.shape[:2]
        size = self._input_size

        # Upstream ingests BGR frames via cv2.imread and normalizes with the
        # ImageNet stats in that channel order, so we mirror that exactly.
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(bgr, (size, size), interpolation=cv2.INTER_AREA)
        tensor_input = resized.astype(np.float32) / 255.0
        tensor_input = (tensor_input - _IMAGENET_MEAN) / _IMAGENET_STD
        tensor = torch.from_numpy(
            np.ascontiguousarray(tensor_input.transpose(2, 0, 1))
        ).float().unsqueeze(0).to(self._device)

        with torch.no_grad():
            _, seg_logits = self._model(tensor)
        probs = torch.sigmoid(seg_logits)[0, 0]
        heatmap = probs.cpu().numpy().astype(np.float32)

        if heatmap.shape != (orig_h, orig_w):
            heatmap = cv2.resize(
                heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
            )

        # Upstream reports max as the image-level score, but that metric is
        # brittle on ID documents where isolated pixel activations from
        # security features (holograms, microprint) push a clean page to a
        # false HARD_REJECT. Top-1% mean aggregates the strongest activations
        # while ignoring single-pixel outliers — real splicing produces many
        # high-probability pixels, scanner noise produces only a handful.
        score = _top_fraction_mean(heatmap, fraction=0.01)
        return heatmap, score


def _top_fraction_mean(heatmap: np.ndarray, fraction: float) -> float:
    """Return the mean value of the top `fraction` most-activated pixels."""
    if heatmap.size == 0:
        return 0.0
    flat = heatmap.reshape(-1)
    k = max(1, int(round(flat.size * fraction)))
    # np.partition is an O(n) alternative to sort for top-k selection.
    top_k = np.partition(flat, flat.size - k)[flat.size - k:]
    return float(top_k.mean())


def _register_mvss_module(root: Path) -> None:
    """Expose the vendored mvss code under its expected import path."""
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def build_engine(
    model_name: str = "auto",
    device: str = "cpu",
) -> TamperingEngine:
    """Factory. `auto` tries DocTamper first and falls back to Mock."""
    if model_name == "mock":
        return MockEngine()
    if model_name == "doctamper":
        return DocTamperEngine(device=device)
    if model_name == "auto":
        try:
            return DocTamperEngine(device=device)
        except (RuntimeError, ImportError, FileNotFoundError) as exc:
            print(f"[WARN] DocTamper unavailable ({exc.__class__.__name__}): {exc}")
            print("[WARN] Falling back to MockEngine — NO real detection.")
            return MockEngine()
    raise ValueError(f"Unknown model_name: {model_name!r}")


def build_mvssnet_engine(device: str = "cpu") -> Optional["MVSSNetEngine"]:
    """Factory for the MVSS-Net engine. Returns None when weights are missing
    or torch cannot be loaded, so callers can degrade gracefully.
    """
    try:
        return MVSSNetEngine(device=device)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        print(f"[WARN] MVSS-Net unavailable ({exc.__class__.__name__}): {exc}")
        return None
