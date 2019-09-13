import numpy as np
from utils.bbox_utils import delta2bbox
import cupy as cp
from model.rpn.nms.non_maximum_suppression import non_maximum_suppression
class ProposalCreator(object):
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        super(ProposalCreator, self).__init__()
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, anchors, scores, preds, img_size, scale,is_training):
        # convert anchors to boxes
        rois = delta2bbox(anchors, preds)

        # clip the boxes to img size
        rois[:, 0:4:2] = np.clip(rois[:, 0:4:2], 0, img_size[0])
        rois[:, 1:4:2] = np.clip(rois[:, 1:4:2], 0, img_size[1])

        min_size = self.min_size * scale
        h = rois[:, 2] - rois[:, 0]
        w = rois[:, 3] - rois[:, 1]
        keep = np.where((w >= min_size) & (h >= min_size))[0]
        rois=rois[keep,:]
        scores=scores[keep]

        if is_training:
            n_pre_nms=self.n_train_pre_nms
            n_post_nms=self.n_train_post_nms
        else:
            n_pre_nms=self.n_test_pre_nms
            n_post_nms=self.n_test_post_nms

        # TODO: the code below is copied from simple_faster_rcnn, understand it soon.

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = scores.ravel().argsort()[::-1]
        if n_pre_nms > 0:
            order = order[:n_pre_nms]
        rois = rois[order, :]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(rois)),
            thresh=self.nms_thresh)
        if n_post_nms > 0:
            keep = keep[:n_post_nms]
        rois = rois[keep]
        return rois
