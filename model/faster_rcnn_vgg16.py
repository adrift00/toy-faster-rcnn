import cupy as cp
import numpy as np
import torch
import torch.nn.functional as F
from collections import namedtuple
from model.faster_rcnn import FasterRCNN
from model.roi.roi_head_vgg16 import ROIHeadVgg16
from model.rpn.anchor_target_creator import AnchorTargetCreator
from model.rpn.nms.non_maximum_suppression import non_maximum_suppression
from model.rpn.proposal_target_creator import ProposalTargetCreator
from model.rpn.region_proposal_network import RegionProposalNetwork
from model.vgg16.vgg16 import decom_vgg16
from utils.bbox_utils import delta2bbox

from config import cfg

LossTuple = namedtuple('LossTuple',
                       ['rpn_reg_loss',
                        'rpn_cls_loss',
                        'roi_reg_loss',
                        'roi_cls_loss',
                        'total_loss'])


def nograd(f):
    def new_f(*args, **kwargs):
        with torch.no_grad():
            return f(*args, **kwargs)

    return new_f


class FasterRCNNVgg16(FasterRCNN):
    def __init__(self):
        extractor, classifier = decom_vgg16()
        rpn = RegionProposalNetwork()
        head = ROIHeadVgg16(n_class=cfg.n_fg_class + 1,
                            roi_size=cfg.roi_size,
                            spatial_scale=(1. / cfg.feat_stride),
                            classifier=classifier)
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()
        super().__init__(extractor, rpn, head)

        # TODO: ADD indicators for training status

    def forward(self, imgs, bboxes, gt_labels, scale=1.0):
        img_size = imgs.shape[2:]
        features = self.extractor(imgs)
        # TODO:batch_size is noly support 1, try to support n
        imgs = imgs[0]
        bboxes = bboxes[0]
        gt_labels = gt_labels[0]
        ######################################
        rpn_deltas, rpn_scores, rois, roi_indies, anchors = self.rpn(features, img_size=img_size, scale=scale)
        # batchsize only support 1
        rpn_delta = rpn_deltas[0]
        rpn_scores = rpn_scores[0]
        gt_rpn_deltas, gt_rpn_labels = self.anchor_target_creator(anchors,
                                                                bboxes=bboxes.detach().cpu().numpy(),
                                                                img_size=img_size)  # note: ndarray is need

        sample_rois, gt_roi_deltas, gt_roi_labels = self.proposal_target_creator(rois=rois,
                                                                               bboxes=bboxes.detach().cpu().numpy(),
                                                                               labels=gt_labels.detach().cpu().numpy())

        # ndarray->Tensor
        sample_rois = torch.from_numpy(sample_rois).to(cfg.device)
        sample_roi_indies = torch.zeros(sample_rois.shape[0]).to(cfg.device)
        roi_deltas, roi_scores = self.roi_head(features, roi_indies=sample_roi_indies, rois=sample_rois)

        # calc rpn loss
        # ndarray->Tensor
        gt_rpn_deltas = torch.from_numpy(gt_rpn_deltas).to(cfg.device)
        gt_rpn_labels = torch.from_numpy(gt_rpn_labels).to(cfg.device).long()  # NOTE: cross_entropy need long
        rpn_reg_loss = self._fast_rcnn_reg_loss(rpn_delta, gt_rpn_deltas, gt_rpn_labels, cfg.rpn_sigma)
        rpn_cls_loss = F.cross_entropy(rpn_scores, gt_rpn_labels, ignore_index=-1)

        # calc roi loss
        # ndarray->Tensor
        gt_roi_deltas = torch.from_numpy(gt_roi_deltas).to(cfg.device)
        gt_roi_labels = torch.from_numpy(gt_roi_labels).to(cfg.device).long()
        n_samples = roi_deltas.shape[0]
        roi_deltas = roi_deltas.reshape(n_samples, -1, 4)
        roi_deltas = roi_deltas[torch.arange(n_samples), gt_roi_labels]
        roi_reg_loss = self._fast_rcnn_reg_loss(roi_deltas, gt_roi_deltas, gt_roi_labels, cfg.roi_sigma)
        roi_cls_loss = F.cross_entropy(roi_scores, gt_roi_labels)  #

        losses = [rpn_reg_loss, rpn_cls_loss, roi_reg_loss, roi_cls_loss]
        total_loss = sum(losses)
        losses = losses + [total_loss]

        return LossTuple(*losses)

    # TODO: to understand it
    def _fast_rcnn_reg_loss(self, pred_loc, gt_loc, gt_label, sigma):
        in_weight = torch.zeros(gt_loc.shape).to(cfg.device)
        # Localization loss is calculated only for positive rois.
        # NOTE:  unlike origin implementation,
        # we don't need inside_weight and outside_weight, they can calculate by gt_label
        in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
        loc_loss = self._smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
        # Normalize by total number of negtive and positive rois.
        loc_loss /= ((gt_label >= 0).sum().float())  # ignore gt_label==-1 for rpn_loss
        return loc_loss

    # TODO: to understand it
    def _smooth_l1_loss(self, x, t, in_weight, sigma):
        sigma2 = sigma ** 2
        diff = in_weight * (x - t)
        abs_diff = diff.abs()
        flag = (abs_diff.data < (1. / sigma2)).float()
        y = (flag * (sigma2 / 2.) * (diff ** 2) +
             (1 - flag) * (abs_diff - 0.5 / sigma2))
        return y.sum()

    @nograd
    def predict(self, imgs, ori_sizes):
        self.eval()
        bboxes = []
        labels = []
        scores = []
        for img, ori_size in zip(imgs, ori_sizes):
            img = img[None]  # to add a new dim
            img = img.to(cfg.device)
            img_size = img.shape[2:]
            scale = img_size[1] / ori_size[1]
            features = self.extractor(img)
            rpn_deltas, rpn_scores, rois, roi_indices, _ = self.rpn(features, img_size=img_size, scale=scale)
            rois = torch.from_numpy(rois).to(cfg.device)
            roi_indices = torch.from_numpy(roi_indices).to(cfg.device).float()
            roi_deltas, roi_scores = self.roi_head(features, roi_indies=roi_indices, rois=rois)
            mean = torch.Tensor(self.delta_normalize_mean).cuda(). \
                repeat(self.n_class)[None]
            std = torch.Tensor(self.delta_normalize_std).cuda(). \
                repeat(self.n_class)[None]

            roi_deltas = (roi_deltas * std + mean)

            rois = rois / scale

            roi_deltas = roi_deltas.reshape(-1, self.n_class, 4)
            rois = rois.reshape(-1, 1, 4).expand_as(roi_deltas)
            bbox = delta2bbox(src_bbox=rois.reshape(-1, 4).detach().cpu().numpy(),
                              delta=roi_deltas.reshape(-1, 4).detach().cpu().numpy())
            # TODO: try not use torch
            bbox = torch.from_numpy(bbox).to(cfg.device)
            bbox = bbox.reshape(-1, self.n_class * 4)

            bbox[:, 0::2] = bbox[:, 0::2].clamp(min=0, max=ori_size[0])
            bbox[:, 1::2] = bbox[:, 1::2].clamp(min=0, max=ori_size[1])

            probs = F.softmax(roi_scores, dim=1).detach().cpu().numpy()

            bbox = bbox.detach().cpu().numpy()

            bbox, label, score = self._suppress(bbox, probs)

            bboxes.append(bbox)
            labels.append(label)
            scores.append(score)
        self.train()
        return bboxes, labels, scores


    # TODO: understand it
    def _suppress(self, raw_cls_bbox, raw_prob):
        bbox = list()
        label = list()
        score = list()
        # skip cls_id = 0 because it is the background class
        for l in range(1, self.n_class):
            cls_bbox_l = raw_cls_bbox.reshape((-1, self.n_class, 4))[:, l, :]
            prob_l = raw_prob[:, l]
            mask = prob_l > self.score_thresh
            cls_bbox_l = cls_bbox_l[mask]
            prob_l = prob_l[mask]
            keep = non_maximum_suppression(
                cp.array(cls_bbox_l), self.nms_thresh, prob_l)
            keep = cp.asnumpy(keep)
            bbox.append(cls_bbox_l[keep])
            # The labels are in [0, self.n_class - 2].
            label.append((l - 1) * np.ones((len(keep),)))
            score.append(prob_l[keep])
        bbox = np.concatenate(bbox, axis=0).astype(np.float32)
        label = np.concatenate(label, axis=0).astype(np.int32)
        score = np.concatenate(score, axis=0).astype(np.float32)
        return bbox, label, score

    def get_parameter(self):
        params = []
        lr = cfg.lr
        for key, value in dict(self.named_parameters()).items():
            if value.requires_grad:
                if 'bias' in key:
                    params += [{'params': [value], 'lr': lr * 2, 'weight_decay': 0}]
                else:
                    params += [{'params': [value], 'lr': lr, 'weight_decay': cfg.weight_decay}]
        return params


