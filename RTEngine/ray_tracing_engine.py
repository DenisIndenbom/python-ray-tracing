import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .objects import *
from .vector_methods import *

__all__ = ['RayTracingEngine']


class RayTracingEngine:
    def __init__(self,
                 width: int,
                 height: int,
                 max_depth: int,
                 camera: np.ndarray,
                 objects: list[Object],
                 lights: list[np.ndarray]):
        self.width = width
        self.height = height
        self.max_depth = max_depth

        ratio = float(width / height)
        self.screen = (-1, 1 / ratio, 1, -1 / ratio)

        self.camera = camera

        self.objects = objects
        self.lights = lights

    def nearest_intersected_object(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        distances = [obj.intersect(ray_origin, ray_direction) for obj in self.objects]

        nearest_object = None
        min_distance = np.inf

        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = self.objects[index]

        return nearest_object, min_distance

    def get_sky(self, rd: np.ndarray):

        result = np.zeros(3)

        for light in self.lights:
            light = normalize(light)

            col = np.array([0.3, 0.6, 1.0])
            sun = np.array([0.95, 0.9, 1.0])

            sun *= max(0.0, np.dot(rd, light)**256.0)
            col *= max(0.0, np.dot(light, np.array([0.0, 0.0, -1.0])))

            result += sun + col * 0.8

        return result / result.max() if result.max() > 1 else result

    def cast_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, color: np.ndarray, death: int):
        if death <= 0:
            return color

        nearest_object, min_distance = self.nearest_intersected_object(ray_origin, ray_direction)

        if nearest_object is None:
            return color * self.get_sky(ray_direction)

        intersection = ray_origin + min_distance * ray_direction
        normal = nearest_object.get_normal(intersection)

        rand = np.random.rand(3)

        origin = intersection + 1e-5 * normal
        direction = reflected(ray_direction, normal) + \
                    normalize(rand * np.dot(rand, normal)) * nearest_object.get_material().matt

        color *= nearest_object.get_material().color

        return self.cast_ray(origin, direction, color, death - 1)

    def trace_ray(self, pos: tuple):
        i, j, x, y = pos

        pixel = np.array([x, y, 0])
        origin = self.camera
        direction = normalize(pixel - origin)

        color = self.cast_ray(origin, direction, np.array([1., 1., 1.]), self.max_depth)

        return i, j, np.clip(color, 0, 1)

    def render(self, sampling=8, processes=None):
        processes = cpu_count() if processes is None else processes

        image = np.zeros((self.height, self.width, 3))
        screen = self.screen

        cords = [(i, j, x, y) for i, y in enumerate(np.linspace(screen[1], screen[3], self.height)) for j, x in
                 enumerate(np.linspace(screen[0], screen[2], self.width))]

        for _ in tqdm(range(sampling)):
            with Pool(processes=processes) as pool:
                result = pool.map(self.trace_ray, cords)
                for row in result:
                    i, j, color = row

                    image[i, j] += color

        image /= sampling

        image **= 0.7

        return image
