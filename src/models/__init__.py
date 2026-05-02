# models/__init__.py
from .resnet import ResNet50WithExtras

# If you still have the EfficientNet file and want to keep it available, uncomment below:
# from .efficientnet_v2 import EfficientNetV2MStandard, EfficientNetV2MWithExtras

__all__ = ['ResNet50WithExtras']