import torch.nn as nn


class FasterRCNN(nn.Module):
    def __init__(self, extractor, rpn, roi_head,
                 delta_normalize_mean=(0., 0., 0., 0.),
                 delta_normalize_std=(0.1, 0.1, 0.2, 0.2)
                 ):
        super().__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.roi_head = roi_head
        self.delta_normalize_mean = delta_normalize_mean
        self.delta_normalize_std = delta_normalize_std
        self.nms_thresh = 0.3
        self.score_thresh = 0.05

    def forward(self):
        pass

    @property
    def n_class(self):
        return self.roi_head.n_class

