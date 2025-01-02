from numpy import array, ndarray

__all__ = ['Material']


class Material:
    def __init__(self, color: list | ndarray, matt: int | float = 0, refl: int | float = 1, bloom: bool = False):
        assert isinstance(color, list) or isinstance(color, ndarray)
        assert isinstance(matt, int | float)
        assert isinstance(refl, int | float)
        assert isinstance(bloom, bool)

        assert 0 <= matt <= 1
        assert 0 <= refl <= 1

        self.color = array(color) if isinstance(color, list) else color
        self.matt = matt
        self.refl = refl

        self.bloom = bloom
