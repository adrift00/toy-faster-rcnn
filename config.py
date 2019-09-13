import torch
class Config(object):

    device=torch.device('cuda')
    #img config
    voc_data_dir='/home/keyan/ZhangXiong/VOC2007'
    min_size=600
    max_size=1000
    n_fg_class=20

    #mocel
    load_path=None
    #train/test
    num_workers=10
    epoch_num=15
    print_every=100
    test_num=1000

    #optimizer
    use_adam=False
    lr= 1e-3
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4

    #vgg16
    use_dropout = False

    #rpn
    anchor_ratios = (0.5, 1, 2)
    anchor_scales = (8, 16, 32)
    feat_stride=16

    #roi
    roi_size=7

    #loss
    rpn_sigma = 3.
    roi_sigma = 1.

    def state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}






cfg=Config()


