import inspect
from private_llm_analytics.config import load_app_config

cfg = load_app_config("configs")
print("Config fields:", dir(cfg))
print("Privacy fields:", dir(cfg.privacy))
print("Privacy.release fields:", dir(cfg.privacy.release))
