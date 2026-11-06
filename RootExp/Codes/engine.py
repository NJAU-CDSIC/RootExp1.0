import importlib.util
import os


_UNET_ENGINE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unet", "engine.py")
)
_spec = importlib.util.spec_from_file_location("rootbin_unet_engine", _UNET_ENGINE_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

# Re-export
train_one_epoch = _module.train_one_epoch
val_one_epoch = _module.val_one_epoch
test_one_epoch = _module.test_one_epoch

