import numpy as np
from utils.bbox_utils import bbox_iou, bbox2delta


class AnchorTargetCreator(object):
    def __init__(self,
                 n_samples=256,
                 pos_iou_thresh=0.7,
                 neg_iou_thresh=0.3,
                 pos_ratio=0.5
                 ):
        super().__init__()
        self.n_samples = n_samples
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, anchors, bboxes, img_size):
        n_anchors = anchors.shape[0]
        inside_idx = self._calc_inside_idx(anchors, img_size)
        anchors = anchors[inside_idx, :]
        ious = bbox_iou(anchors, bboxes)
        max_iou_idx = np.argmax(ious, axis=1)
        max_iou = np.max(ious, axis=1)
        gt_max_iou=np.max(ious,axis=0)
        gt_max_iou_idx = np.where(ious==gt_max_iou)[0]
        # labels the anchors,
        labels = -1 * np.ones(len(inside_idx),dtype=np.int32)
        # ground true labels 1
        labels[gt_max_iou_idx] = 1
        # max ious > pos_iou_thresh labels 1
        labels[max_iou >= self.pos_iou_thresh] = 1
        # max ious<neg_iou_thresh labels 0
        labels[max_iou < self.neg_iou_thresh] = 0

        n_pos = int(self.n_samples * self.pos_ratio)
        pos_idx = np.where(labels == 1)[0]
        if len(pos_idx) > n_pos:
            sub_num = len(pos_idx) - n_pos
            sub_idx = np.random.choice(pos_idx, sub_num, replace=False)
            labels[sub_idx] = -1
        neg_idx = np.where(labels == 0)[0]
        n_neg = self.n_samples - np.sum(labels == 1)
        if len(neg_idx) > n_neg:
            sub_num = len(neg_idx) - n_neg
            sub_idx = np.random.choice(neg_idx, sub_num, replace=False)
            labels[sub_idx] = -1
        # calc the deltas from anchor to real bbox
        deltas = bbox2delta(anchors, bboxes[max_iou_idx])
        labels = self._fill_data(labels, n_anchors, inside_idx, fill=-1)
        deltas = self._fill_data(deltas, n_anchors, inside_idx, fill=0)

        return deltas, labels

    def _calc_inside_idx(self, anchor, img_size):
        idx = np.where((anchor[:, 0] >= 0)
                       & (anchor[:, 1] >= 0)
                       & (anchor[:, 2] < img_size[0])
                       & (anchor[:, 3] < img_size[1])
                       )[0]
        return idx

    def _fill_data(self, data, ori_len, data_idx, fill=0):
        size = list(data.shape)
        size[0] = ori_len
        ret = np.empty(size,dtype=data.dtype)
        ret.fill(fill)
        ret[data_idx] = data #
        return ret
