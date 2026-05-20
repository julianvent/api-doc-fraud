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
_DEFAULT_TRUFOR_WEIGHTS = _WEIGHTS_DIR / "trufor" / "trufor.pth.tar"

# Fixed input size for predictable latency.
_TRUFOR_INPUT_SIZE = 512
# DocTamper was trained on 512x512 crops.
_MODEL_INPUT_SIZE = 512
# JPEG quality for DCT coefficient extraction at inference time.
_JPEG_QUALITY = 85

_IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


class TamperingEngine(Protocol):
    name: str

    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Return (HxW float32 heatmap, score in [0,1])."""
        ...


class MockEngine:
    """Stub engine for exercising the pipeline."""

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

    Upstream uses `jpegio` for DCT extraction; we approximate via JPEG
    round-trip + scipy DCT since `jpegio` lacks Windows wheels.
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
        # Strip the DataParallel `module.` prefix if present.
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

        # Mean tampered probability across the heatmap.
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
    """Expose vendored model code under the names expected by upstream
    DocTamper code and its pickled checkpoints."""
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
    """Backfill attributes missing from pre-2.0 torch pickles."""
    import torch.nn as nn
    from timm.layers import DropPath

    for module in model.modules():
        if isinstance(module, nn.GELU) and not hasattr(module, "approximate"):
            module.approximate = "none"
        if isinstance(module, DropPath) and not hasattr(module, "scale_by_keep"):
            module.scale_by_keep = True


def _install_timm_legacy_aliases() -> None:
    """Alias pre-1.0 `timm.models.layers.*` paths to current `timm.layers`."""
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
    """Approximate JPEG quantized DCT coefficients via encode+decode+blockwise DCT."""
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


class TruForEngine:
    """TruFor (CVPR 2023): SegFormer-B2 CMX + Noiseprint++. Exposes the
    pixel-level heatmap (softmax channel 1) and the image-level `det` score.
    """

    name = "trufor-cmx-mit_b2"

    def __init__(
        self,
        weights_path: Path = _DEFAULT_TRUFOR_WEIGHTS,
        device: str = "cpu",
        input_size: int = _TRUFOR_INPUT_SIZE,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("torch is not installed") from exc

        if not Path(weights_path).is_file():
            raise RuntimeError(f"TruFor weights not found: {weights_path}")

        self._torch = torch
        self._device = torch.device(device)
        self._input_size = int(input_size)
        self._model = self._load_model(Path(weights_path))

    def _load_model(self, weights_path: Path):
        torch = self._torch
        from yacs.config import CfgNode as CN

        # Config inlined from upstream trufor.yaml.
        cfg = CN()
        cfg.MODEL = CN()
        cfg.MODEL.NAME = "detconfcmx"
        cfg.MODEL.PRETRAINED = ""
        cfg.MODEL.MODS = ("RGB", "NP++")
        cfg.MODEL.EXTRA = CN(new_allowed=True)
        cfg.MODEL.EXTRA.BACKBONE = "mit_b2"
        cfg.MODEL.EXTRA.DECODER = "MLPDecoder"
        cfg.MODEL.EXTRA.DECODER_EMBED_DIM = 512
        cfg.MODEL.EXTRA.PREPRC = "imagenet"
        cfg.MODEL.EXTRA.BN_EPS = 0.001
        cfg.MODEL.EXTRA.BN_MOMENTUM = 0.1
        cfg.MODEL.EXTRA.DETECTION = "confpool"
        cfg.MODEL.EXTRA.CONF = True
        cfg.DATASET = CN()
        cfg.DATASET.NUM_CLASSES = 2

        from .models.trufor.cmx.builder_np_conf import myEncoderDecoder
        model = myEncoderDecoder(cfg=cfg)
        checkpoint = torch.load(
            str(weights_path), map_location=self._device, weights_only=False,
        )
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval().to(self._device)
        return model

    def detect(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, float]:
        """Run TruFor on an HxWx3 RGB uint8 image.

        Returns (heatmap, score):
          * heatmap: HxW float32 in [0, 1], resized back to the original
            input dimensions. Each pixel is P(tampered).
          * score: float in [0, 1] from the detection head — directly
            trained for image-level decision, not derived from heatmap.
        """
        torch = self._torch
        orig_h, orig_w = image_rgb.shape[:2]
        size = self._input_size

        # Upstream uses Image.open(...).convert("RGB"), tensor(transpose) / 256.
        # Mirror that division by 256 (not 255) since the model's prepro layer
        # was calibrated against that scaling.
        resized = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(
            np.ascontiguousarray(resized.transpose(2, 0, 1))
        ).float().unsqueeze(0).to(self._device) / 256.0

        with torch.no_grad():
            pred, _conf, det, _npp = self._model(tensor)

        probs = torch.softmax(pred, dim=1)[0, 1]
        heatmap = probs.cpu().numpy().astype(np.float32)

        if heatmap.shape != (orig_h, orig_w):
            heatmap = cv2.resize(
                heatmap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR,
            )

        score = float(torch.sigmoid(det).item())
        return heatmap, score


def build_trufor_engine(device: str = "cpu") -> Optional["TruForEngine"]:
    """Factory. Returns None when weights or required modules are missing."""
    try:
        return TruForEngine(device=device)
    except (RuntimeError, ImportError, FileNotFoundError) as exc:
        print(f"[WARN] TruFor unavailable ({exc.__class__.__name__}): {exc}")
        return None


