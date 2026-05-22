"""Yogurt Sa2VA nodes for ComfyUI."""

import inspect
import os

import folder_paths

print("Loading Yogurt Sa2VA...")

from .sa2va_node import *
from .sa2va_node_v2 import *
from .sa2va_vitmatte import *


models_dir = folder_paths.models_dir

sa2va_dir = os.path.join(models_dir, "sa2va")
os.makedirs(sa2va_dir, exist_ok=True)

vitmatte_dir = os.path.join(models_dir, "vitmatte")
os.makedirs(vitmatte_dir, exist_ok=True)


def _register_model_folder(folder_name, model_dir):
    if folder_name not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[folder_name] = ([model_dir], {"folder"})
        return
    folder_paths.add_model_folder_path(folder_name, model_dir)


def _node_key_for_class(class_name):
    if class_name.startswith("Yogurt"):
        return class_name
    return f"YogurtSa2VA{class_name}"


_register_model_folder("sa2va", sa2va_dir)
_register_model_folder("vitmatte", vitmatte_dir)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

for name, obj in list(globals().items()):
    if inspect.isclass(obj) and hasattr(obj, "_NODE_NAME"):
        node_name = _node_key_for_class(name)
        NODE_CLASS_MAPPINGS[node_name] = obj
        NODE_DISPLAY_NAME_MAPPINGS[node_name] = obj._NODE_NAME

print(f"Yogurt Sa2VA loaded: {tuple(NODE_DISPLAY_NAME_MAPPINGS.values())}")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
