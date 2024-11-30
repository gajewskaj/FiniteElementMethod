import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_outline(ax: Axes3D):
    xi_outline = [-1, 1, 1, -1, -1]
    eta_outline = [-1, -1, 1, 1, -1]

    ax.plot(xi_outline, eta_outline, N1(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)
    ax.plot(xi_outline, eta_outline, N2(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)
    ax.plot(xi_outline, eta_outline, N3(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)
    ax.plot(xi_outline, eta_outline, N4(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)

N1 = lambda xi, eta: 0.25 * (1-xi) * (1-eta)
N2 = lambda xi, eta: 0.25 * (1+xi) * (1-eta)
N3 = lambda xi, eta: 0.25 * (1+xi) * (1+eta)
N4 = lambda xi, eta: 0.25 * (1-xi) * (1+eta)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(projection='3d')

xi, eta = np.meshgrid(np.linspace(-1, 1, 50), np.linspace(-1, 1, 50))
ax.plot_surface(xi, eta, N3(xi, eta), cmap='plasma', alpha=0.8)
plot_outline(ax)
ax.set_xlabel('ξ')
ax.set_ylabel('η')
ax.set_zlabel('N\u2083(ξ, η)')

plt.tight_layout()
plt.show()