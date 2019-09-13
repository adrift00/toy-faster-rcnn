from torch.utils.data import DataLoader
from data.dataset import TrainDataset, TestDataset
from model.faster_rcnn_trainer import FasterRCNNTrainer
from config import cfg
from utils.eval_utils import eval_detection_voc


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in enumerate(dataloader):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train():
    train_dataset = TrainDataset()
    test_dataset = TestDataset()
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=cfg.num_workers)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=cfg.num_workers,
                                 pin_memory=True)

    trainer = FasterRCNNTrainer()


    if cfg.load_path:
        trainer.load(cfg.load_path)

    best_map = 0
    for ep in range(cfg.epoch_num):
        for it, (imgs, bboxes, labels, scale) in enumerate(train_dataloader):
            imgs, bboxes, labels, scale = \
                imgs.to(cfg.device), bboxes.to(cfg.device), labels.to(cfg.device), scale.item()
            loss = trainer.step(imgs, bboxes, labels, scale)

            if it % cfg.print_every == 0:
                print('epoch: %d, iter: %d,loss: %.4f ' % (ep, it, loss.total_loss.detach().item()))

        eval_result = eval(test_dataloader, trainer.faster_rcnn, test_num=cfg.test_num)
        lr_ = trainer.optimizer.param_groups[0]['lr']
        print('-----------------------------')
        print('ep: %d, lr: %.5f, map: %.6f' % (ep, lr_, eval_result['map']))
        print('ap:', eval_result['ap'])
        print('-----------------------------')
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if ep == 9:
            trainer.load(best_path)
            trainer.scale_lr(cfg.lr_decay)
            lr_ = lr_ * cfg.lr_decay

        if ep == 13:
            break


if __name__ == "__main__":
    train()
