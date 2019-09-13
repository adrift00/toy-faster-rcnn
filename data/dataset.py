import torch
import skimage.transform as skt
import torchvision.transforms as tvt
from torch.utils.data import Dataset
from utils import data_utils
from data.voc_dataset import VOCBboxDataset
from config import cfg


def pytorch_normalize(img):
    normalize = tvt.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
    img = normalize(torch.from_numpy(img))
    return img.numpy()


def preprocess_img(img, min_size=600, max_size=1000):
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img /= 255.
    img = skt.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    img = pytorch_normalize(img)
    return img


class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess_img(img, min_size=self.min_size, max_size=self.max_size)
        _, OH, OW = img.shape
        scale = OH / H
        bbox = data_utils.resize_bbox(bbox, (H, W), (OH, OW))

        # horizontally flip
        img, params = data_utils.random_flip(
            img, x_random=True, return_param=True)
        bbox = data_utils.flip_bbox(
            bbox, (OH, OW), x_flip=params['x_flip'])

        return img, bbox, label, scale


class TrainDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.data = VOCBboxDataset(cfg.voc_data_dir,split='trainval')
        self.trasform = Transform(cfg.min_size, cfg.max_size)

    def __getitem__(self, idx):
        img, bbox, label, difficult = self.data.get_example(idx)
        img, bbox, label, scale = self.trasform((img, bbox, label))
        return img.copy(),bbox,label,scale

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self,split='test',use_difficult=True):
        super().__init__()
        self.data=VOCBboxDataset(cfg.voc_data_dir,split=split,use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img,bbox,label,difficult=self.data.get_example(idx)
        img=preprocess_img(ori_img)
        return img,ori_img.shape[1:],bbox,label,difficult

    def __len__(self):
        return len(self.data)



