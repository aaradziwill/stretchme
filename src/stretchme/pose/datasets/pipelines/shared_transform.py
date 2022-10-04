import warnings
from collections.abc import Sequence

import torch
from mmcv.utils import build_from_cfg
from mmcv.utils import Registry

try:
    import albumentations
except ImportError:
    albumentations = None


PIPELINES = Registry("pipeline")


@PIPELINES.register_module()
class ToTensor:
    """Transform image to Tensor.
    Required key: 'img'. Modifies key: 'img'.
    Args:
        results (dict): contain all information about training.
    """

    def __init__(self, device="cpu"):
        self.device = device

    def _to_tensor(self, x):
        return (
            torch.from_numpy(x.astype("float32"))
            .permute(2, 0, 1)
            .to(self.device)
            .div_(255.0)
        )

    def __call__(self, results):
        if isinstance(results["img"], (list, tuple)):
            results["img"] = [self._to_tensor(img) for img in results["img"]]
        else:
            results["img"] = self._to_tensor(results["img"])

        return results


@PIPELINES.register_module()
class Compose:
    """Compose a data pipeline with a sequence of transforms.
    Args:
        transforms (list[dict | callable]): Either config
          dicts of transforms or transform objects.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    "transform must be callable or a dict, but got"
                    f" {type(transform)}"
                )

    def __call__(self, data):
        """Call function to apply transforms sequentially.
        Args:
            data (dict): A result dict contains the data to transform.
        Returns:
            dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        """Compute the string representation."""
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string
