# Based on https://github.com/blakechi/ComVEX/blob/master/comvex/coatnet/model.py

import torch.nn as nn
import torch.nn.functional as F
from MBConv import MBConvForRelativeAttention
from Transformer import TransformerWithRelativeAttention, ProjectionHead
from utils import print_num_params

configs = {
    'coatnet-0': {
        'num_blocks': [2, 2, 3, 5, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-1': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [64, 96, 192, 384, 768],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-2': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [128, 128, 256, 512, 1026],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-3': {
        'num_blocks': [2, 2, 6, 14, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-4': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 192, 384, 768, 1536],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 32,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-5': {
        'num_blocks': [2, 2, 12, 28, 2],
        'num_channels': [192, 256, 512, 1280, 2048],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 64,
        'block_types': ['C', 'C', 'T', 'T']
    },
    'coatnet-6': {
        'num_blocks': [2, 2, 4, [8, 42], 2],
        'num_channels': [192, 192, 384, [768, 1536], 2048],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 128,
        'block_types': ['C', 'C', 'C-T', 'T']
    },
    'coatnet-7': {
        'num_blocks': [2, 2, 4, [8, 42], 2],
        'num_channels': [192, 256, 512, [1024, 2048], 3072],
        'expand_ratio': [4, 4, 4, 4, 4],
        'n_head': 128,
        'block_types': ['C', 'C', 'C-T', 'T']
    }
}

blocks = {
    'C': MBConvForRelativeAttention,
    'T': TransformerWithRelativeAttention
}


class CoAtNet(nn.Module):
    def __init__(self, inp_h, inp_w, in_channels, config='coatnet-0', num_classes=None, head_act_fn='mish', head_dropout=0.1):
        super().__init__()
        self.config = configs[config]
        block_types = self.config['block_types']
        self.s0 = self._make_stem(in_channels)
        self.s1 = self._make_block(block_types[0], inp_h >> 2, inp_w >> 2,
                                   self.config['num_channels'][0],
                                   self.config['num_channels'][1],
                                   self.config['num_blocks'][1],
                                   self.config['expand_ratio'][0])
        self.s2 = self._make_block(block_types[1], inp_h >> 3, inp_w >> 3,
                                   self.config['num_channels'][1],
                                   self.config['num_channels'][2],
                                   self.config['num_blocks'][2],
                                   self.config['expand_ratio'][1])
        self.s3 = self._make_block(block_types[2], inp_h >> 4, inp_w >> 4,
                                   self.config['num_channels'][2],
                                   self.config['num_channels'][3],
                                   self.config['num_blocks'][3],
                                   self.config['expand_ratio'][2])
        self.s4 = self._make_block(block_types[3], inp_h >> 5, inp_w >> 5,
                                   self.config['num_channels'][3],
                                   self.config['num_channels'][4],
                                   self.config['num_blocks'][4],
                                   self.config['expand_ratio'][3])
        if num_classes is not None:
            self.head = ProjectionHead(self.config['num_channels'][-1], num_classes, act_fn=head_act_fn, ff_dropout=head_dropout)
        else:
            self.head = None

    def _make_stem(self, in_channels):
        return nn.Sequential(*[
            nn.Conv2d(
                in_channels if i == 0 else self.config['num_channels'][0],
                self.config['num_channels'][0], kernel_size=3, padding=1,
                stride=2 if i == 0 else 1
            ) for i in range(self.config['num_blocks'][0])
        ])

    def _make_block(self, block_type, inp_h, inp_w, in_channels, out_channels, depth, expand_ratio):
        block_list = []
        if not isinstance(in_channels, int):
            in_channels = in_channels[-1]
        if block_type in blocks:
            block_cls = blocks[block_type]
            block_list.extend([
                block_cls(
                    inp_h, inp_w, in_channels if i == 0 else out_channels,
                    n_head=self.config['n_head'], out_channels=out_channels,
                    expand_ratio=expand_ratio, use_downsampling=i == 0
                ) for i in range(depth)
            ])
        else:
            for i, _block_type in enumerate(block_type.split('-')):
                block_cls = blocks[_block_type]
                block_list.extend(
                    block_cls(
                        inp_h, inp_w, in_channels if i == 0 and j == 0 else out_channels[i - 1] if j == 0 else out_channels[i],
                        n_head=self.config['n_head'], out_channels=out_channels[i],
                        expand_ratio=expand_ratio, use_downsampling=j == 0
                    ) for j in range(depth[i])
                )
        return nn.Sequential(*block_list)

    def forward(self, x):
        x = self.s0(x)
        x = self.s1(x)
        x = self.s2(x)
        x = self.s3(x)
        x = self.s4(x)
        x = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        if self.head is not None:
            x = self.head(x)
        return x


if __name__ == '__main__':
    import torch
    from utils import print_num_params

    for i in range(8):
        config=f'coatnet-{i}'
        print(config)
        coatnet = CoAtNet(224, 224, 3, config=config, num_classes=None)
        #print(coatnet)
        print_num_params(coatnet)
        coatnet.eval()
        random_image = torch.randint(0, 256, size=(1, 3, 224, 224)).float() / 255
        with torch.no_grad():
            output = coatnet(random_image)
