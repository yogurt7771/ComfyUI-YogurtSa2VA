"""
VITMatte Post-Processing for Sa2VA
Simple, focused implementation - VITMatte only
"""

import gc
import logging
import math
import os
import threading
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

import folder_paths

from .model_management_adapter import YogurtComfyManagedModel, load_models_gpu

logger = logging.getLogger(__name__)


DEFAULT_VITMATTE_MODEL_NAMES = [
    "hustvl/vitmatte-base-composition-1k",
    "hustvl/vitmatte-small-composition-1k",
]


def _normalize_model_name(path):
    return path.replace(os.sep, "/")


def _dedupe_model_names(model_names):
    deduped = []
    seen = set()
    for model_name in model_names:
        if not model_name or model_name in seen:
            continue
        deduped.append(model_name)
        seen.add(model_name)
    return deduped


def get_local_hf_model_names(folder_name):
    """Return relative model directory names that contain a Hugging Face config.json."""
    local_models = []
    try:
        model_roots = folder_paths.get_folder_paths(folder_name)
    except Exception as e:
        logger.warning(f"Could not list model folder '{folder_name}': {e}")
        return local_models

    for model_root in model_roots:
        if not os.path.isdir(model_root):
            continue

        for dirpath, dirnames, filenames in os.walk(model_root):
            dirnames[:] = [dirname for dirname in dirnames if dirname != ".git"]
            if "config.json" not in filenames:
                continue

            relative_path = os.path.relpath(dirpath, model_root)
            if relative_path == ".":
                continue

            local_models.append(_normalize_model_name(relative_path))

    return sorted(set(local_models))


def get_available_vitmatte_model_names():
    """Return local VITMatte model folders first, followed by known remote defaults."""
    return _dedupe_model_names(
        get_local_hf_model_names("vitmatte") + DEFAULT_VITMATTE_MODEL_NAMES
    )


def get_local_vitmatte_path(model_name):
    """Check if VITMatte model exists in ComfyUI's models directory.

    Args:
        model_name: HuggingFace model identifier (e.g., "hustvl/vitmatte-small-composition-1k")

    Returns:
        Local path if model exists, otherwise original model_name for HuggingFace download
    """
    try:
        # Get ComfyUI models directory
        models_dir = folder_paths.models_dir
        vitmatte_dir = os.path.join(models_dir, "vitmatte")

        # Convert HuggingFace model name to local path
        # "hustvl/vitmatte-small-composition-1k" -> "models/vitmatte/hustvl/vitmatte-small-composition-1k"
        local_path = os.path.join(vitmatte_dir, model_name)

        # Check if model exists locally (look for config.json as indicator)
        config_file = os.path.join(local_path, "config.json")
        if os.path.exists(config_file):
            logger.info(f"Found local VITMatte model at: {local_path}")
            return local_path

        # Check alternative location without organization prefix
        # Some users might download to "models/vitmatte/vitmatte-small-composition-1k" directly
        if "/" in model_name:
            model_name_short = model_name.split("/")[-1]
            local_path_alt = os.path.join(vitmatte_dir, model_name_short)
            config_file_alt = os.path.join(local_path_alt, "config.json")
            if os.path.exists(config_file_alt):
                logger.info(f"Found local VITMatte model at: {local_path_alt}")
                return local_path_alt

        # Model not found locally, will download from HuggingFace
        logger.info(f"Local VITMatte model not found. Will download from HuggingFace: {model_name}")
        return model_name

    except Exception as e:
        logger.warning(f"Error checking local VITMatte path: {e}. Using HuggingFace download.")
        return model_name


class VITMattePostProcessor:
    """VITMatte-based mask refinement for Sa2VA."""

    def __init__(
        self,
        detail_erode: int = 6,
        detail_dilate: int = 6,
        black_point: float = 0.15,
        white_point: float = 0.99,
        max_megapixels: float = 2.0,
        device: str = "cuda",
        model_name: str = "hustvl/vitmatte-small-composition-1k",
        model_bundle=None,
    ):
        """Initialize VITMatte post-processor.

        Args:
            detail_erode: Erosion kernel size for trimap (inner boundary)
            detail_dilate: Dilation kernel size for trimap (outer boundary)
            black_point: Histogram black point (0.01-0.98)
            white_point: Histogram white point (0.02-0.99)
            max_megapixels: Max resolution for processing (0.5-10.0)
            device: "cuda" or "cpu"
        """
        self.detail_erode = detail_erode
        self.detail_dilate = detail_dilate
        self.black_point = black_point
        self.white_point = white_point
        self.max_megapixels = max_megapixels
        self.device = device
        self.model_name = model_name
        self.model_bundle = model_bundle

        self.model = None
        self.processor = None

    def generate_trimap(self, mask_np: np.ndarray) -> Image.Image:
        """Generate trimap from sigmoid mask.

        Args:
            mask_np: Raw sigmoid mask [H, W] in 0-1 range

        Returns:
            PIL Image with trimap (L mode):
                - 0 (black): Definite background
                - 128 (gray): Uncertain region (for VITMatte to decide)
                - 255 (white): Definite foreground

        Algorithm:
            1. Convert sigmoid (0-1) to binary (0/255) with threshold 0.5
            2. Erode to get definite foreground (shrink inward)
            3. Dilate to get possible foreground (expand outward)
            4. Uncertain region = dilated - eroded
        """
        # Convert to binary uint8 for OpenCV
        mask_binary = (mask_np > 0.5).astype(np.uint8) * 255

        # Create morphological kernels
        erode_kernel = np.ones((self.detail_erode, self.detail_erode), np.uint8)
        dilate_kernel = np.ones((self.detail_dilate, self.detail_dilate), np.uint8)

        # Generate trimap zones (5 iterations like LayerStyle)
        eroded = cv2.erode(mask_binary, erode_kernel, iterations=5)
        dilated = cv2.dilate(mask_binary, dilate_kernel, iterations=5)

        # Build trimap
        trimap = np.zeros_like(mask_binary)
        trimap[dilated == 255] = 128  # Uncertain region (gray)
        trimap[eroded == 255] = 255   # Definite foreground (white)
        # Background stays 0 (black)

        return Image.fromarray(trimap).convert('L')

    def process_mask(
        self,
        image_pil: Image.Image,
        mask_np: np.ndarray
    ) -> np.ndarray:
        """Process mask with VITMatte.

        Args:
            image_pil: Original RGB image as PIL Image
            mask_np: Raw sigmoid mask [H, W] in 0-1 range

        Returns:
            Refined alpha matte [H, W] in 0-1 range

        Pipeline:
            1. Generate trimap from sigmoid mask
            2. Resize if image too large (memory management)
            3. Run VITMatte AI inference
            4. Upscale back to original size
            5. Apply histogram remapping
        """
        # Load model (lazy loading)
        self._load_model()

        # Ensure RGB mode
        if image_pil.mode != 'RGB':
            image_pil = image_pil.convert('RGB')

        # Step 1: Generate trimap
        trimap_pil = self.generate_trimap(mask_np)

        # Step 2: Resolution management (from LayerStyle)
        width, height = image_pil.size
        max_pixels = self.max_megapixels * 1_048_576

        if width * height > max_pixels:
            # Downscale for processing
            ratio = width / height
            target_width = int(math.sqrt(ratio * max_pixels))
            target_height = int(target_width / ratio)

            image_resized = image_pil.resize(
                (target_width, target_height),
                Image.BILINEAR
            )
            trimap_resized = trimap_pil.resize(
                (target_width, target_height),
                Image.BILINEAR
            )
        else:
            image_resized = image_pil
            trimap_resized = trimap_pil

        # Step 3: VITMatte inference
        device = torch.device(self.device if torch.cuda.is_available() else "cpu")

        inputs = self.processor(
            images=image_resized,
            trimaps=trimap_resized,
            return_tensors="pt"
        )

        inference_lock = getattr(self.model_bundle, "inference_lock", None)
        if inference_lock is None:
            inference_lock = threading.RLock()

        with inference_lock:
            if self.model_bundle is not None:
                self.model_bundle.ensure_model_on_gpu()
            with torch.no_grad():
                inputs = {k: v.to(device) for k, v in inputs.items()}
                predictions = self.model(**inputs).alphas

        # Explicit cleanup of intermediate tensors
        del inputs

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        # Step 4: Convert to numpy and remove padding
        alpha_np = predictions[0, 0].cpu().numpy()

        # VITMatte works in 32px tiles - crop to actual size
        alpha_np = alpha_np[:image_resized.height, :image_resized.width]

        # Step 5: Upscale if downscaled
        if width * height > max_pixels:
            alpha_pil = Image.fromarray((alpha_np * 255).astype(np.uint8))
            alpha_pil = alpha_pil.resize((width, height), Image.BILINEAR)
            alpha_np = np.array(alpha_pil).astype(np.float32) / 255.0

        # Step 6: Histogram remapping
        alpha_np = self.histogram_remap(alpha_np)

        return alpha_np

    def histogram_remap(self, alpha_np: np.ndarray) -> np.ndarray:
        """Apply histogram remapping for edge cleanup.

        Args:
            alpha_np: Alpha matte [H, W] in 0-1 range

        Returns:
            Remapped alpha with enhanced contrast

        Effect:
            - Values < black_point → 0 (remove gray halos)
            - Values > white_point → 1 (solidify foreground)
            - Values between → linearly stretched

        Formula (from LayerStyle):
            bp = min(black_point, white_point - 0.001)
            scale = 1 / (white_point - bp)
            output = clip((input - bp) * scale, 0, 1)
        """
        bp = min(self.black_point, self.white_point - 0.001)
        scale = 1.0 / (self.white_point - bp)
        remapped = np.clip((alpha_np - bp) * scale, 0.0, 1.0)
        return remapped

    def _load_model(self):
        """Load VITMatte model (lazy loading)."""
        if self.model is not None:
            return

        if self.model_bundle is not None:
            if self.model_bundle.model is None or self.model_bundle.processor is None:
                raise RuntimeError("VITMatte model bundle is not loaded")
            self.model = self.model_bundle.model
            self.processor = self.model_bundle.processor
            self.device = self.model_bundle.device
            return

        from transformers import VitMatteImageProcessor, VitMatteForImageMatting

        # Check for local model first, fallback to HuggingFace download
        model_path = get_local_vitmatte_path(self.model_name)

        self.processor = VitMatteImageProcessor.from_pretrained(
            model_path,
        )

        self.model = VitMatteForImageMatting.from_pretrained(
            model_path,
        )

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        logger.info(f"VITMatte model loaded on {device}")

    def cleanup(self):
        """Unload model and free memory."""
        if self.model_bundle is not None:
            self.model = None
            self.processor = None
            return

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()


class YogurtVITMatteModel:
    """Reusable VITMatte model bundle returned by the loader node."""

    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.model = None
        self.processor = None
        self.inference_lock = threading.RLock()
        self.managed_model = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested for VITMatte but not available. Falling back to CPU.")
            return "cpu"
        return device

    def load_model(self):
        if self.model is not None and self.processor is not None:
            logger.info(f"Reusing loaded VITMatte model: {self.model_name}")
            return

        from transformers import VitMatteImageProcessor, VitMatteForImageMatting

        model_path = get_local_vitmatte_path(self.model_name)
        self.processor = VitMatteImageProcessor.from_pretrained(model_path)
        self.model = VitMatteForImageMatting.from_pretrained(model_path)

        device = torch.device(self.device if torch.cuda.is_available() or self.device == "cpu" else "cpu")
        self.model.to(device)
        self.model.eval()
        self.device = str(device)
        self._attach_managed_model(device)

        logger.info(f"VITMatte model loaded on {device}: {self.model_name}")

    def ensure_model_on_gpu(self):
        if self.managed_model is None:
            return
        load_models_gpu([self.managed_model], force_full_load=True)

    def _attach_managed_model(self, load_device):
        self.managed_model = YogurtComfyManagedModel(
            self.model,
            name=f"VITMatte {self.model_name}",
            load_device=load_device,
        )
        self.ensure_model_on_gpu()

    def unload_model(self):
        if self.managed_model is not None:
            self.managed_model.detach()
            self.managed_model = None

        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        gc.collect()


_VITMATTE_MODEL_CACHE = {}


def _vitmatte_cache_key(model_name, device):
    return (str(model_name), str(device))


def require_loaded_vitmatte(vitmatte_model):
    if not isinstance(vitmatte_model, YogurtVITMatteModel):
        raise TypeError("vitmatte_model must come from Yogurt VITMatte Model Loader")
    if vitmatte_model.model is None or vitmatte_model.processor is None:
        raise RuntimeError("VITMatte model is not loaded. Re-run the Yogurt VITMatte Model Loader node.")
    return vitmatte_model


class YogurtVITMatteModelLoader:
    """Load VITMatte once and pass it to V2 refinement nodes."""

    _NODE_NAME = "Yogurt VITMatte Model Loader"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (
                    get_available_vitmatte_model_names(),
                    {"default": "hustvl/vitmatte-small-composition-1k"},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto"},
                ),
                "force_reload": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Unload and recreate the cached VITMatte model for this model/device configuration.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("YOGURT_VITMATTE_MODEL",)
    RETURN_NAMES = ("vitmatte_model",)
    FUNCTION = "load_model"
    CATEGORY = "YogurtSa2VA"
    DESCRIPTION = "Loads a reusable VITMatte model for Yogurt Sa2VA V2 detail refinement."

    def load_model(self, model_name, device="auto", force_reload=False):
        key = _vitmatte_cache_key(model_name, device)

        if force_reload and key in _VITMATTE_MODEL_CACHE:
            logger.info(f"Force reloading cached VITMatte model: {model_name}")
            _VITMATTE_MODEL_CACHE[key].unload_model()
            del _VITMATTE_MODEL_CACHE[key]

        model_bundle = _VITMATTE_MODEL_CACHE.get(key)
        if model_bundle is None:
            model_bundle = YogurtVITMatteModel(model_name, device)
            model_bundle.load_model()
            _VITMATTE_MODEL_CACHE[key] = model_bundle
        elif model_bundle.model is None or model_bundle.processor is None:
            model_bundle.load_model()
        else:
            logger.info(f"Reusing cached VITMatte model: {model_name}")

        return (model_bundle,)
