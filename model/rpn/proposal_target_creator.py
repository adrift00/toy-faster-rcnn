import numpy as np

from utils.bbox_utils import bbox_iou, bbox2delta


class ProposalTargetCreator(object):
    def __init__(self,
                 n_samples=128,
                 pos_ratio=0.25,
                 pos_iou_threh=0.5,
                 neg_iou_threh_hi=0.5,
                 neg_iou_threh_lo=0
                 ):
        super().__init__()
        self.n_samples = n_samples
        self.pos_ratio = pos_ratio
        self.pos_iou_threh = pos_iou_threh
        self.neg_iou_threh_hi = neg_iou_threh_hi
        self.neg_iou_threh_lo = neg_iou_threh_lo

    def __call__(self, rois, bboxes, labels,
                 delta_normalize_mean=(0., 0., 0., 0.),
                 delta_normalize_std=(0.1, 0.1, 0.2, 0.2)):  # TODO: what is the normalize mean/std?
        rois = np.concatenate((rois, bboxes), axis=0)
        pos_roi_per_img = np.round(self.n_samples * self.pos_ratio)
        iou = bbox_iou(rois, bboxes)
        gt_assignment = np.argmax(iou, axis=1)
        iou_max = np.max(iou, axis=1)
        gt_roi_labels = labels[gt_assignment] + 1

        pos_idx = np.where(iou_max >= self.pos_iou_threh)[0]
        pos_roi_per_this_img = int(min(len(pos_idx), pos_roi_per_img))
        if len(pos_idx) > 0:
            pos_idx = np.random.choice(pos_idx, pos_roi_per_this_img, replace=False)

        neg_idx = np.where((iou_max >= self.neg_iou_threh_lo)
                           & (iou_max < self.neg_iou_threh_hi))[0]
        neg_roi_per_img = self.n_samples-pos_roi_per_this_img
        neg_roi_per_this_img=int(min(len(neg_idx),neg_roi_per_img))
        if len(neg_idx) > 0:
            neg_idx = np.random.choice(neg_idx, neg_roi_per_this_img, replace=False)
        keep_idx = np.append(pos_idx, neg_idx)
        sample_rois = rois[keep_idx]
        gt_roi_labels = gt_roi_labels[keep_idx]
        gt_roi_labels[pos_roi_per_this_img:] = 0

        gt_roi_deltas = bbox2delta(sample_rois, bboxes[gt_assignment[keep_idx]])
        gt_roi_deltas = ((gt_roi_deltas - np.array(delta_normalize_mean, np.float32)
                        ) / np.array(delta_normalize_std, np.float32))
        return sample_rois, gt_roi_deltas, gt_roi_labels
