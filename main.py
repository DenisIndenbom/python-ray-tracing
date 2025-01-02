import numpy as np
import matplotlib.pyplot as plt

from RTEngine import RayTracingEngine
from RTEngine import Material
from RTEngine.objects import Sphere, Cube

if __name__ == "__main__":
    scale = 2

    camera_pos = np.array([0, 0, 5])

    lights = [
        np.array([1, 1, -3])
    ]

    objects = [
        Cube(np.array([0.5, 0, -1]), 0.3, Material([0.8, 0.6, 0.6], 0.5)),
        Sphere(np.array([-0.5, 0, -1.2]), 0.3, Material([0.6, 0.6, 0.9], 0.05)),
        Sphere(np.array([0, -1000, 0]), 1000 - 0.3, Material([0.5, 0.5, 0.5], 0.05))
    ]

    rte = RayTracingEngine(300 * scale, 200 * scale, 6, camera_pos, objects, lights)

    plt.imsave('image.png', rte.render(sampling=4))
