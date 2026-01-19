# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.models.yolo import detect, obb

from .model import YOLO

__all__ = "YOLO", "detect", "obb"
