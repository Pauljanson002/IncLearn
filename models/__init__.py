from .ViT import vit
from .IncVit import inc_vit_b


def get_model(name):
    if name == 'vit':
        return vit()
    elif name == 'inc_vit_b':
        return inc_vit_b()
    else:
        raise NotImplementedError()
