import numpy as np
from abc import ABC, abstractmethod

from .material import Material

from .vector_methods import *

__all__ = ['Object', 'Sphere']


class Object(ABC):
    @abstractmethod
    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        pass

    @abstractmethod
    def get_normal(self, intersection) -> np.ndarray:
        pass

    @abstractmethod
    def get_pos(self) -> tuple:
        pass

    @abstractmethod
    def get_material(self) -> Material:
        pass


class Sphere(Object):
    def __init__(self, center: np.ndarray, radius: int | float, material: Material):
        assert isinstance(center, np.ndarray) and isinstance(radius, int | float) and isinstance(material, Material)

        self.center = center
        self.radius = radius

        self.__material = material

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        b = 2 * np.dot(ray_direction, ray_origin - self.center)
        c = np.linalg.norm(ray_origin - self.center)**2 - self.radius**2
        delta = b**2 - 4 * c
        if delta > 0:
            t1 = (-b + np.sqrt(delta)) / 2
            t2 = (-b - np.sqrt(delta)) / 2
            if t1 > 0 and t2 > 0:
                return min(t1, t2)

        return None

    def get_normal(self, intersection: np.ndarray):
        return normalize(intersection - self.center)

    def get_pos(self):
        return self.center

    def get_material(self):
        return self.__material


class Cube(Object):
    def __init__(self, center: np.ndarray, size: int | float, material: Material):
        assert isinstance(center, np.ndarray) and isinstance(size, int | float) and isinstance(material, Material)

        self.center = center
        self.size = size

        self.__material = material

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        m = 1.0 / ray_direction
        n = m * (ray_origin - self.center)
        k = abs(m) * self.size
        t1 = -n - k
        t2 = -n + k

        tN = max(max(t1[0], t1[1]), t1[2])
        tF = min(min(t2[0], t2[1]), t2[2])

        if tN > tF or tF < 0.0: return None

        return min(tN, tF)

    def get_normal(self, intersection: np.ndarray):
        local_coords = (intersection - self.center) / self.size

        axis = np.argmax(np.abs(local_coords))

        normal = np.zeros(3)
        normal[axis] = np.sign(local_coords[axis])

        return normal

    def get_pos(self):
        return self.center

    def get_material(self):
        return self.__material
