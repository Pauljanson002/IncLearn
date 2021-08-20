from models.ViT import ViT
from torch import nn


class IncVit(ViT):
    def __init__(self, in_channels=3, n_classes=100, emb_size=256, depth=7, img_size=32, patch_size=4):
        super(IncVit, self).__init__(in_channels=in_channels, n_classes=n_classes,
                                     emb_size=emb_size,
                                     depth=depth, img_size=img_size,
                                     patch_size=patch_size)

    def incremental_learning(self, num_classes):
        for encoder_block in self.tranformer_encoder.blocks:
            encoder_block.mha.increaseHead()

        old_ch_linear = self.classification_head.linear
        self.classification_head.linear = nn.Linear(old_ch_linear.in_features, num_classes)
        self.classification_head.linear.weight.data[:old_ch_linear.out_features] = old_ch_linear.weight.data
        self.classification_head.linear.bias.data[:old_ch_linear.out_features] = old_ch_linear.bias.data


def inc_vit_b():
    return IncVit(n_classes=10)
