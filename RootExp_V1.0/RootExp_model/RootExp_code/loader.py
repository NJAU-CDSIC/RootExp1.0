import importlib.util
import os


_UNET_LOADER_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unet", "loader.py")
)
_spec = importlib.util.spec_from_file_location("rootbin_unet_loader", _UNET_LOADER_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
isic_loader = _module.isic_loader

