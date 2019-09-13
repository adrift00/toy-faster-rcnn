import torch
import torch.nn as nn
from model.roi.roi_module import RoIPooling2D
class ROIHeadVgg16(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super().__init__()
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.classifier = classifier
        self.pred = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)
        # NOTE: use normal_init to match origin paper
        normal_init(self.pred, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        self.roi = RoIPooling2D(self.roi_size, self.roi_size, self.spatial_scale)

    def forward(self, x, rois, roi_indies):
        indices_and_rois = torch.cat((roi_indies.view(-1, 1), rois), dim=1)
        #note: xyxy needed here.
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois = xy_indices_and_rois.contiguous()
        pool = self.roi(x, indices_and_rois)
        pool = pool.view(pool.size(0), -1)

        fc7 = self.classifier(pool)

        roi_scores = self.score(fc7)
        roi_deltas = self.pred(fc7)

        return roi_deltas, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
