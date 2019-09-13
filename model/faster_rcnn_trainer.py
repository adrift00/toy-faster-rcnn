import torch
from model.faster_rcnn_vgg16 import FasterRCNNVgg16
from config import cfg
import os
import time

class FasterRCNNTrainer(object):
    def __init__(self):
        super().__init__()
        self.faster_rcnn = FasterRCNNVgg16().to(cfg.device)
        self.optimizer=self.get_optimizer()

    def step(self, imgs, bbox, labels, scale):
        self.optimizer.zero_grad()
        loss = self.faster_rcnn(imgs, bbox, labels, scale)
        loss.total_loss.backward()
        self.optimizer.step()
        return loss

    def get_optimizer(self):
        params = self.faster_rcnn.get_parameter()
        if cfg.use_adam:
            optimizer = torch.optim.Adam(params)
        else:
            optimizer = torch.optim.SGD(params, momentum=0.9)
        return optimizer

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.

        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = cfg.state_dict()
        save_dict['other_info'] = kwargs

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(save_dict, save_path)
        return save_path

    def load(self, path, load_optimizer=True ):
        state_dict = torch.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def scale_lr(self, decay=0.1):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= decay
        return self.optimizer