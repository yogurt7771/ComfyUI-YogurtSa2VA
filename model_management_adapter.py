"""Small adapter that lets external torch modules participate in ComfyUI VRAM management."""

import gc
import logging
import threading

import torch


logger = logging.getLogger(__name__)


def get_model_management():
    import comfy.model_management as model_management

    return model_management


def load_models_gpu(models, **kwargs):
    get_model_management().load_models_gpu(models, **kwargs)


def get_torch_device():
    return get_model_management().get_torch_device()


def unet_offload_device():
    return get_model_management().unet_offload_device()


def module_size(model):
    return get_model_management().module_size(model)


def _detect_module_device(module, fallback):
    for parameter in module.parameters(recurse=True):
        return parameter.device
    for buffer in module.buffers(recurse=True):
        return buffer.device
    return torch.device(fallback)


def _detect_module_dtype(module):
    for parameter in module.parameters(recurse=True):
        return parameter.dtype
    for buffer in module.buffers(recurse=True):
        return buffer.dtype
    return torch.float32


class YogurtComfyManagedModel:
    """Whole-model offload adapter for ComfyUI's model_management queue.

    ComfyUI's native patchers can partially load weights. Hugging Face models do
    not expose that granularity here, so this adapter moves the whole module
    between load_device and offload_device.
    """

    parent = None

    def __init__(
        self,
        model,
        name="Yogurt external model",
        load_device=None,
        offload_device=None,
        dtype=None,
        unload_strategy="move",
        unload_callback=None,
        reload_callback=None,
        delete_delay_seconds=0.1,
    ):
        if unload_strategy not in {"move", "delete"}:
            raise ValueError("unload_strategy must be 'move' or 'delete'")

        self.model = model
        self.name = name
        self.load_device = torch.device(load_device or get_torch_device())
        self.offload_device = torch.device(
            offload_device or unet_offload_device()
        )
        self._current_device = _detect_module_device(model, self.offload_device)
        self._dtype = dtype or _detect_module_dtype(model)
        self._size = 0
        self.unload_strategy = unload_strategy
        self.unload_callback = unload_callback
        self.reload_callback = reload_callback
        self.delete_delay_seconds = delete_delay_seconds
        self._deferred_delete_model = None
        self._deferred_delete_timer = None

    def model_patches_models(self):
        return []

    def is_dynamic(self):
        return False

    def is_clone(self, other):
        return self.model is not None and getattr(other, "model", None) is self.model

    def model_size(self):
        if self._size <= 0:
            if self.model is None:
                return 0
            self._size = module_size(self.model)
        return self._size

    def loaded_size(self):
        if self.current_loaded_device() == self.load_device:
            return self.model_size()
        return 0

    def current_loaded_device(self):
        return self._current_device

    def model_patches_to(self, target):
        if isinstance(target, torch.dtype):
            self._dtype = target

    def model_dtype(self):
        return self._dtype

    def lowvram_patch_counter(self):
        return 0

    def partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
        device = torch.device(device_to)
        if self.model is None:
            return self._reload_to(device)
        return self._move_to(device)

    def partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
        if self.loaded_size() <= 0:
            return 0
        device = torch.device(device_to)
        if self.unload_strategy == "delete":
            return self._delete_from_memory(device)
        return self._move_to(device)

    def detach(self, unpatch_all=True, unpatch_weights=True):
        if self.unload_strategy == "delete":
            self._delete_from_memory(self.offload_device)
        else:
            self._move_to(self.offload_device)

    def _move_to(self, device):
        if self._current_device == device:
            return 0

        try:
            self.model.to(device)
        except Exception as e:
            logger.warning(f"Could not move {self.name} to {device}: {e}")
            return 0

        self._current_device = device
        if device.type == "cpu" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        return self.model_size()

    def _delete_from_memory(self, device):
        if self.model is None:
            self._current_device = device
            return 0

        freed = self.model_size()
        self._deferred_delete_model = self.model

        try:
            if self.unload_callback is not None:
                self.unload_callback()
        finally:
            self.model = None
            self._current_device = device
            self._schedule_deferred_delete()

        logger.info(f"Deleted {self.name} from memory; it will reload on next use.")
        return freed

    def _schedule_deferred_delete(self):
        if self._deferred_delete_timer is not None:
            self._deferred_delete_timer.cancel()

        self._deferred_delete_timer = threading.Timer(
            self.delete_delay_seconds,
            self._complete_deferred_delete,
        )
        self._deferred_delete_timer.daemon = True
        self._deferred_delete_timer.start()

    def _complete_deferred_delete(self):
        self._deferred_delete_model = None
        self._deferred_delete_timer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def _reload_to(self, device):
        if self.reload_callback is None:
            raise RuntimeError(f"{self.name} was deleted but has no reload callback")

        model = self.reload_callback(device)
        if model is None:
            raise RuntimeError(f"{self.name} reload callback did not return a model")

        self.model = model
        self._dtype = _detect_module_dtype(model)
        self._current_device = device
        if self._size <= 0:
            self._size = module_size(model)

        logger.info(f"Reloaded {self.name} after memory release.")
        return self.model_size()
