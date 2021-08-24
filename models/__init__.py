from .ViT import vit, cct
from .IncVit import inc_vit_b, inc_cct_b


def get_model(name):
    if name == 'vit':
        return vit()
    elif name == 'inc_vit_b':
        return inc_vit_b()
    elif name == 'inc_cct_b':
        return inc_cct_b()
    elif name == 'cct':
        return cct()
    else:
        raise NotImplementedError()
