import numpy as np
import matplotlib.pyplot as plt

from RTEngine import RayTracingEngine
from RTEngine import Material
from RTEngine.objects import Sphere, Cube

if __name__ == "__main__":
    camera_pos = np.array([0, 0, 5])
    lights = [
        np.array([0.3, 0.75, -1])
    ]
    objects = [
        Cube(np.array([0.5, 0, -1]), 0.3, Material([0.9, 0.6, 0.6], 0.05)),
        Sphere(np.array([-0.5, 0, -1.2]), 0.3, Material([0.6, 0.6, 0.9], 0.2)),
        Sphere(np.array([0, -1000, -1]), 1000 - 0.3, Material([0.5, 0.5, 0.5], 0.3))
    ]

    rte = RayTracingEngine(300 * 3, 200 * 3, 8, camera_pos, objects, lights)

    plt.imsave('image.png', rte.render(sampling=1))
