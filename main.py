import numpy as np
import matplotlib.pyplot as plt

from RTEngine import RayTracingEngine
from RTEngine import Material
from RTEngine.objects import Sphere, Cube
from RTEngine.lights import Light

if __name__ == "__main__":
    scale = 3

    camera_pos = np.array([0, 0, -7])

    lights = [
        Light([0.35, 0.9, -1]),
    ]

    objects = [
        Sphere(np.array([0.5, 0, -1]), 0.3, Material([0.9, 0.6, 0.6], 0)),
        Sphere(np.array([-0.5, 0, -1]), 0.3, Material([0.6, 0.6, 0.9], 0.7)),
        Cube(np.array([0, -1000, 0]), 1000 - 0.3, Material([0.5, 0.5, 0.5], 0.3))
    ]

    rte = RayTracingEngine(300 * scale, 200 * scale, 6, camera_pos, objects, lights)

    plt.imsave('image.png', rte.render(sampling=1, gamma=0.7))
