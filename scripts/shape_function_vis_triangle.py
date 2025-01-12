import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_outline(ax: Axes3D):
    xi_outline = [-1, 1, -1, -1]
    eta_outline = [-1, -1, 1, -1]

    # ax.plot(xi_outline, eta_outline, temp(np.array(xi_outline), np.array(eta_outline)), color='r', linewidth=2)
    # ax.plot(xi_outline, eta_outline, temp2(np.array(xi_outline), np.array(eta_outline)), color='r', linewidth=2)
    # ax.plot(xi_outline, eta_outline, temp3(np.array(xi_outline), np.array(eta_outline)), color='r', linewidth=2)
    ax.plot(xi_outline, eta_outline, N1(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)
    ax.plot(xi_outline, eta_outline, N2(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)
    ax.plot(xi_outline, eta_outline, N3(np.array(xi_outline), np.array(eta_outline)), color='k', linewidth=2)

# temp = lambda xi, eta: 0.1127
# temp2 = lambda xi, eta: 0.8873
# temp3 = lambda xi, eta: 0.5
N1 = lambda xi, eta: -0.5 * (xi+eta)
N2 = lambda xi, eta: 0.5 * (1+xi)
N3 = lambda xi, eta: 0.5 * (1+eta)

fig = plt.figure()
ax: Axes3D = fig.add_subplot(projection='3d')

xi, eta = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
mask = (xi + eta < 0)
N1_masked = np.where(mask, N1(xi, eta), np.nan)
N2_masked = np.where(mask, N2(xi, eta), np.nan)
N3_masked = np.where(mask, N3(xi, eta), np.nan)

ax.plot_surface(xi, eta, N3_masked, cmap='plasma', alpha=0.8)
plot_outline(ax)
ax.set_xlabel('ξ')
ax.set_ylabel('η')
ax.set_zlabel('N\u2083(ξ, η)')

plt.tight_layout()
plt.show()