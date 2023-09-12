import numpy as np
import matplotlib.pyplot as plt

import functools

from multiprocessing import Pool

from tqdm import tqdm

width = 300 * 3
height = 200 * 3
max_depth = 8

ratio = float(width / height)
screen = (-1, 1 / ratio, 1, -1 / ratio)


def normalize(vector):
    return vector / np.linalg.norm(vector)


def reflected(vector, axis):
    return vector - 2 * np.dot(vector, axis) * axis


def sphere_intersect(center, radius, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - center)
    c = np.linalg.norm(ray_origin - center)**2 - radius**2
    delta = b**2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)

    return None


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(obj['center'], obj['radius'], ray_origin, ray_direction) for obj in objects]

    nearest_object = None
    min_distance = np.inf

    for index, distance in enumerate(distances):
        if distance and distance < min_distance:
            min_distance = distance
            nearest_object = objects[index]

    return nearest_object, min_distance


def get_sky(rd):
    result = np.zeros(3)

    for light in lights:
        light = normalize(light)

        col = np.array([0.3, 0.6, 1.0])
        sun = np.array([0.95, 0.9, 1.0])

        sun *= max(0.0, np.dot(rd, light)**256.0)
        col *= max(0.0, np.dot(light, np.array([0.0, 0.0, -1.0])))

        result += sun + col * 0.8

    return result / result.max() if result.max() > 1 else result


def cast_ray(ray_origin, ray_direction, color, death):
    if death <= 0:
        return color

    nearest_object, min_distance = nearest_intersected_object(objects, ray_origin, ray_direction)

    if nearest_object is None:
        return color * get_sky(ray_direction)

    intersection = ray_origin + min_distance * ray_direction
    normal_to_surface = normalize(intersection - nearest_object['center'])

    rand = np.random.rand(3)

    origin = intersection + 1e-5 * normal_to_surface
    direction = reflected(ray_direction, normal_to_surface) + normalize(rand * np.dot(rand, normal_to_surface)) * \
                nearest_object['matt']

    color *= nearest_object['color']

    return cast_ray(origin, direction, color, death - 1)


lights = [
    np.array([0.3, 0.75, -1]),
    # np.array([-0.3, 0.75, -1])
]

objects = [
    {'center':np.array([0.5, 0, -1]), 'radius':0.3, 'color':np.array([0.9, 0.6, 0.6]), 'matt':0,},
    {'center':np.array([-0.5, 0, -1]), 'radius':0.3, 'color':np.array([0.6, 0.6, 0.9]), 'matt':0.7},
    {'center':np.array([0, -1000, -1]), 'radius':1000 - 0.3, 'color':np.array([0.5, 0.5, 0.5]), 'matt':0.3}
]


def tracing(camera, pos):
    i, j, x, y = pos

    pixel = np.array([x, y, 0])
    origin = camera
    direction = normalize(pixel - origin)

    color = cast_ray(origin, direction, np.array([1., 1., 1.]), max_depth)

    return i, j, np.clip(color, 0, 1)


def do_tracing(camera, sampling=128):
    image = np.zeros((height, width, 3))

    cords = [(i, j, x, y) for i, y in enumerate(np.linspace(screen[1], screen[3], height)) for j, x in
             enumerate(np.linspace(screen[0], screen[2], width))]

    for _ in tqdm(range(sampling)):
        with Pool(processes=16) as pool:
            result = pool.map(functools.partial(tracing, camera), cords)
            for row in result:
                i, j, color = row

                image[i, j] += color

    image /= sampling

    image **= 0.7

    return image


if __name__ == "__main__":
    plt.imsave('image.png', do_tracing(np.array([0, 0, 5]), 64))
