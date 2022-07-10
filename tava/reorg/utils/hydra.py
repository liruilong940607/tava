import functools

from hydra.utils import get_method


def partial(_partial_, *args, **kwargs):
    return functools.partial(get_method(_partial_), *args, **kwargs)
