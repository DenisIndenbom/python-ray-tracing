from numpy import array, ndarray

__all__ = ['Material']


class Material:
    def __init__(self, color: list | ndarray, matt: int | float):
        assert isinstance(color, list) or isinstance(color, ndarray)
        assert isinstance(matt, int | float)

        self.color = array(color) if isinstance(color, list) else color
        self.matt = matt
