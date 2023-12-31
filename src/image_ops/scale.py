from src.image_ops.base import BaseImageOperation
from src.image_ops.utils.transformations import scale


class ScaleOperation(BaseImageOperation):
    """
    Class that implements operation of scaling an image
    """
    def __init__(self, kwidth, kheight, center: (int, int)):
        self._op = lambda X: scale(X, kwidth, kheight, center)
