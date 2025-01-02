import numpy as np

from multiprocessing import Pool, cpu_count
from tqdm import tqdm

from .objects import Object
from .lights import Light
from .vector_methods import normalize, reflected

__all__ = ['RayTracingEngine']


class RayTracingEngine:
    def __init__(self,
                 width: int,
                 height: int,
                 max_depth: int,
                 camera: np.ndarray[float],
                 objects: list[Object],
                 lights: list[Light],
                 sky_color: np.ndarray[float] = np.array([0.3, 0.6, 1.0])):
        self.width = width
        self.height = height
        self.max_depth = max_depth

        ratio = float(width / height)
        self.screen = (-1, 1 / ratio, 1, -1 / ratio)

        self.camera = camera

        self.objects = objects
        self.lights = lights

        self.sky_color = sky_color

    def nearest_intersected_object(self, ray_origin: np.ndarray, ray_direction: np.ndarray):
        distances = [obj.intersect(ray_origin, ray_direction) for obj in self.objects]

        nearest_object = None
        min_distance = np.inf

        for index, distance in enumerate(distances):
            if distance and distance < min_distance:
                min_distance = distance
                nearest_object = self.objects[index]

        return nearest_object, min_distance

    def get_sky(self, normal: np.ndarray, ray_origin: np.array, ray_direction: np.ndarray):
        result = np.zeros(3)

        for light_obj in self.lights:
            distance = np.linalg.norm(light_obj.position - ray_origin)

            brightness = np.dot(ray_direction, normal) / distance ** 2

            result += light_obj.color * brightness + self.sky_color

        return result / result.max() if result.max() > 1 else result

    def cast_ray(self, ray_origin: np.ndarray, ray_direction: np.ndarray, color: np.ndarray, death: int):
        normal = np.zeros(3)

        while death > 0 and sum(color) >= 0.01:
            nearest_object, min_distance = self.nearest_intersected_object(ray_origin, ray_direction)

            if nearest_object is None:
                return color * self.get_sky(normal, ray_origin, ray_direction)

            if nearest_object.get_material().bloom:
                return color * nearest_object.get_material().color

            intersection = ray_origin + min_distance * ray_direction
            normal = nearest_object.get_normal(intersection)

            reflection = normalize(reflected(ray_direction, normal)) * nearest_object.get_material().refl

            rand = np.random.rand(3)
            diff = normalize(rand * np.dot(rand, normal)) * nearest_object.get_material().matt

            ray_origin = intersection + 1e-5 * normal
            ray_direction = normalize(reflection + diff)

            color *= nearest_object.get_material().color

            death -= 1

        return np.zeros(3)

    def trace_ray(self, pos: tuple):
        i, j, x, y = pos

        pixel = np.array([x, y, 0])
        origin = self.camera
        direction = normalize(pixel - origin)

        color = self.cast_ray(origin, direction, np.array([1., 1., 1.]), self.max_depth)

        return i, j, np.clip(color, 0, 1)

    def render(self, sampling: int = 8, processes: bool = None, gamma: int = 0.7):
        processes = cpu_count() if processes is None else processes

        image = np.zeros((self.height, self.width, 3))
        screen = self.screen

        cords = [(i, j, x, y) for i, y in enumerate(np.linspace(screen[1], screen[3], self.height)) for j, x in
                 enumerate(np.linspace(screen[0], screen[2], self.width))]

        for _ in tqdm(range(sampling)):
            with Pool(processes=processes) as pool:
                result = pool.map(self.trace_ray, cords)
                for i, j, color in result:
                    image[i, j] += color

        image /= sampling

        image **= gamma

        return image
