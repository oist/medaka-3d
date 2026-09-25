"""Resolve the registered single-GPU alias before fixed-width validation.

The first run stopped before begin: a tensor on cuda:0 did not compare equal
to an unspecified torch.device('cuda'). Use the experiment's explicit GPU 0
for runtime tensors, keeping the original serialized config and all equations.
"""

import torch
from fixed_body_width_tracker import FixedBodyWidthTracker as PreviousTracker


def resolved_device(device):
    value = torch.device(device)
    if value.type == "cuda" and value.index is None:
        return torch.device("cuda:0")
    return value


class FixedBodyWidthTracker(PreviousTracker):
    def _width_policy(self):
        self.device = resolved_device(self.device)
        return super()._width_policy()
