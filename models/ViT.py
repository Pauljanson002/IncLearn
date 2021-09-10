from torch import nn
from layers import PatchEmbedding, TransformerEncoder, ClassificationHead


class ViT(nn.Module):
    def __init__(self, in_channels=3, n_classes=100, emb_size=256, depth=7, img_size=32, patch_size=4,
                 convolution=False):
        super(ViT, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size, img_size, convolution=convolution)
        self.transformer_encoder = TransformerEncoder(depth)
        self.classification_head = ClassificationHead(emb_size, n_classes,convolution=convolution)

    def forward(self, x):
        x = self.classification_head(self.transformer_encoder(self.patch_embedding(x)))
        return x


def vit():
    return ViT()


def cct():
    return ViT(n_classes=10, convolution=True)
