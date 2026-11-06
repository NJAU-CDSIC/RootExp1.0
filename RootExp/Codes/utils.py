import importlib.util
import os


_UNET_UTILS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unet", "utils.py")
)
_spec = importlib.util.spec_from_file_location("rootbin_unet_utils", _UNET_UTILS_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export everything public from UNet utils
for _k, _v in _module.__dict__.items():
    if not _k.startswith("_"):
        globals()[_k] = _v

