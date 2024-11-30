from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def func(xi, eta):
    return xi+eta+2

points = np.array([-sqrt(3/5),
                    0,
                    sqrt(3/5)], dtype=float)
weights = np.array([5/9,
                    8/9,
                    5/9], dtype=float)

x_gauss, y_gauss = np.meshgrid(points, points)

z_gauss = func(x_gauss, y_gauss)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(111, projection='3d')

for i in range(3):
    for j in range(3):
        dx = weights[j]
        dy = weights[i]
        dz = z_gauss[i, j]
        ax.bar3d(x_gauss[i, j] - dx/2, y_gauss[i, j] - dy/2, 0, dx, dy, dz, color='blue', alpha=0.1)

        ax.plot([x_gauss[i, j], x_gauss[i, j]],
                [y_gauss[i, j], y_gauss[i, j]],
                [0, z_gauss[i, j]], color='k', linewidth=2)

xi, eta = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
ax.plot_surface(xi, eta, func(xi, eta), cmap='plasma', alpha=0.8)
ax.set_xlabel('ξ')
ax.set_ylabel('η')
ax.set_zlabel('f(ξ, η)')

plt.tight_layout()
plt.show()