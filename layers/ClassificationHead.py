from torch import nn


class ClassificationHead(nn.Module):
    def __init__(self, emb_size=256, n_classes=100):
        super(ClassificationHead, self).__init__()
        self.layer_norm = nn.LayerNorm(emb_size)
        self.linear = nn.Linear(emb_size, n_classes)

    def forward(self, x):
        x = x[:, 0]
        x = self.linear(self.layer_norm(x))
        return x

