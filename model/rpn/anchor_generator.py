import numpy as np


class AnchorGenerator(object):
    def __init__(self, basesize=16, radios=(0.5, 1, 2), scales=(8, 16, 32)):
        super().__init__()
        self.basesize = basesize
        self.radios = radios
        self.scales = scales

    def _generate_base_anchors(self):
        py = self.basesize / 2
        px = self.basesize / 2
        anchor_base = np.zeros((len(self.radios) * len(self.scales), 4), dtype=np.float32)
        for i, radio in enumerate(self.radios):
            for j, scale in enumerate(self.scales):
                h = self.basesize * scale * np.sqrt(radio)
                w = self.basesize * scale * np.sqrt(1. / radio)
                idx = i * len(self.scales) + j
                anchor_base[idx, 0] = py - h / 2
                anchor_base[idx, 1] = px - w / 2
                anchor_base[idx, 2] = py + h / 2
                anchor_base[idx, 3] = px + w / 2
        return anchor_base

    def generate_anchors(self, stride, height, width):
        anchor_base = self._generate_base_anchors()
        shift_x = stride * np.arange(0, width)
        shift_y = stride * np.arange(0, height)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)
        anchors = (shift.reshape(-1, 1, 4) + anchor_base.reshape(1, -1, 4))
        anchors = anchors.reshape(-1, 4).astype(np.float32)
        return anchors