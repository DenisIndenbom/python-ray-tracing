from numpy import array, ndarray

__all__ = ['Light']


class Light:
    def __init__(self, position: ndarray[float] | list,
                 color: ndarray[float] | list = array([0.95, 0.9, 1.0])):
        assert isinstance(position, ndarray | list)
        assert isinstance(color, ndarray | list)

        self.position = array(position) if isinstance(position, list) else position
        self.color = array(color) if isinstance(color, list) else color

        assert self.position.shape == (3,)
        assert self.color.shape == (3,)
