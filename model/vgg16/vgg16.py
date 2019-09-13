import torch
import torch.nn as nn
from torchvision.models import vgg16

from config import cfg


def decom_vgg16():
    model = vgg16(pretrained=True)
    features = list(model.features)[:30]  # features是vgg网络前面的卷积层、池化层
    classifier = list(model.classifier)  # classfier是vgg网络后面的几层全连接层
    del classifier[6]
    if not cfg.use_dropout:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    extractor = nn.Sequential(*features)
    return extractor, classifier
