import numpy as np
import matplotlib.pyplot as plt

def plot_outline(ax):
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

xi = np.linspace(-1, 1, 50)
eta = np.linspace(-1, 1, 50)
xi, eta = np.meshgrid(xi, eta)

N1_vals = N1(xi, eta)
N2_vals = N2(xi, eta)
N3_vals = N3(xi, eta)
N4_vals = N4(xi, eta)

fig = plt.figure()

ax = fig.add_subplot(projection='3d')
ax.plot_surface(xi, eta, N3_vals, cmap='plasma', alpha=0.8)
plot_outline(ax)
ax.set_xlabel('ξ')
ax.set_ylabel('η')
ax.set_zlabel('N3(ξ, η)')

plt.tight_layout()
plt.show()