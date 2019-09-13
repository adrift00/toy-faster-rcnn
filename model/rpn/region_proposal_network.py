import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model.rpn.anchor_generator import AnchorGenerator
from model.rpn.proposal_creator import ProposalCreator


class RegionProposalNetwork(nn.Module):
    def __init__(self, in_channels=512, mid_channels=512,
                 anchor_ratios=(0.5, 1, 2), anchor_scales=(8, 16, 32),
                 feat_stride=16,
                 nms_thesh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300):
        super().__init__()
        self.anchor_ratios = anchor_ratios
        self.anchor_scales = anchor_scales
        self.feat_stride = feat_stride

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

        score_channels = len(anchor_ratios) * len(anchor_scales) * 2
        self.score = nn.Conv2d(mid_channels, score_channels, 1, 1, 0)

        pred_channels = len(anchor_ratios) * len(anchor_scales) * 4
        self.pred = nn.Conv2d(mid_channels, pred_channels, 1, 1, 0)
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.pred, 0, 0.01)

        self.anchor_generotor = AnchorGenerator(radios=anchor_ratios,
                                                scales=anchor_scales)
        self.proposal_creator = ProposalCreator(nms_thresh=nms_thesh,
                                                n_train_pre_nms=n_train_pre_nms,
                                                n_train_post_nms=n_train_post_nms,
                                                n_test_pre_nms=n_test_pre_nms,
                                                n_test_post_nms=n_test_post_nms)

    def forward(self, x, img_size, scale=1.0):
        N, _, H, W = x.shape

        anchors = self.anchor_generotor.generate_anchors(self.feat_stride, H, W)
        hidden = F.relu(self.conv1(x))
        rpn_deltas = self.pred(hidden)
        rpn_scores = self.score(hidden)

        rpn_deltas = rpn_deltas.permute(0, 2, 3, 1).reshape(N, -1, 4)  # use reshape istead of contiguous().view()
        rpn_scores = rpn_scores.permute(0, 2, 3, 1)
        n_anchor = len(self.anchor_ratios) * len(self.anchor_scales)
        rpn_softmax_scores = F.softmax(rpn_scores.reshape(N, H, W, n_anchor, 2), dim=4)
        rpn_fg_softmax_scores = rpn_softmax_scores[:, :, :, :, 1].reshape(N, -1)
        rpn_scores = rpn_scores.reshape(N, -1, 2)

        rois = []
        roi_indies = []
        for i in range(N):
            roi = self.proposal_creator(anchors=anchors,
                                        scores=rpn_fg_softmax_scores[i].detach().cpu().numpy(),
                                        # use detach() istead of data
                                        preds=rpn_deltas[i].detach().cpu().numpy(),
                                        img_size=img_size,
                                        scale=scale,
                                        is_training=self.training)  #NOTE: distingwish the train and test
            batch_idx = i * np.ones((roi.shape[0], 1), dtype=np.float32)
            rois.append(roi)
            roi_indies.append(batch_idx)

        rois = np.concatenate(rois, axis=0)
        roi_indies = np.concatenate(roi_indies, axis=0)

        return rpn_deltas, rpn_scores, rois, roi_indies, anchors  # Tensor,Tensor,ndarray,ndarray,ndarray


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

