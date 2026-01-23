import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
from numba import njit

@njit(cache=True)
def _compute_element_forces_numba(p0, p1, p2, x_inv, volume, young_modulus, poisson_ratio, use_cauchy):
    """Oblicza siły elementu 2D trójkątnego z poprawną obsługą dużych rotacji."""
    forces = np.zeros((3, 2), dtype=np.float64)
    if volume == 0.0:
        return forces

    # Gradient deformacji F = [p1-p0, p2-p0] * X_inv
    f00 = (p1[0] - p0[0]) * x_inv[0, 0] + (p2[0] - p0[0]) * x_inv[1, 0]
    f01 = (p1[0] - p0[0]) * x_inv[0, 1] + (p2[0] - p0[0]) * x_inv[1, 1]
    f10 = (p1[1] - p0[1]) * x_inv[0, 0] + (p2[1] - p0[1]) * x_inv[1, 0]
    f11 = (p1[1] - p0[1]) * x_inv[0, 1] + (p2[1] - p0[1]) * x_inv[1, 1]

    # Parametry Lame
    mu = young_modulus / (2.0 * (1.0 + poisson_ratio))
    lmbda = (young_modulus * poisson_ratio) / (1.0 - poisson_ratio**2)

    
    # Model nieliniowy
    # Tensor odkształcenia Greena-Lagrange E = 0.5 * (F^T * F - I)
    c00 = f00*f00 + f10*f10
    c01 = f00*f01 + f10*f11
    c11 = f01*f01 + f11*f11
        
    e00 = 0.5 * (c00 - 1.0)
    e11 = 0.5 * (c11 - 1.0)
    e01 = 0.5 * c01
        
    # Drugi tensor naprężenia Pioli-Kirchhoffa S
    trace = e00 + e11
    s00 = lmbda * trace + 2.0 * mu * e00
    s11 = lmbda * trace + 2.0 * mu * e11
    s01 = 2.0 * mu * e01
        
    # Pierwszy tensor naprężenia Pioli-Kirchhoffa P = F * S (Mapowanie sił na świat)
    p_stress00 = f00 * s00 + f01 * s01
    p_stress01 = f00 * s01 + f01 * s11
    p_stress10 = f10 * s00 + f11 * s01
    p_stress11 = f10 * s01 + f11 * s11

    # Gradienty funkcji kształtu
    g1x, g1y = x_inv[0, 0], x_inv[0, 1]
    g2x, g2y = x_inv[1, 0], x_inv[1, 1]
    g0x, g0y = -(g1x + g2x), -(g1y + g2y)

    grads = ((g0x, g0y), (g1x, g1y), (g2x, g2y))
    for i in range(3):
        gx, gy = grads[i]
        # f = -Area * P * gradN
        forces[i, 0] = -volume * (p_stress00 * gx + p_stress01 * gy)
        forces[i, 1] = -volume * (p_stress10 * gx + p_stress11 * gy)

    return forces

@njit(cache=True)
def _assemble_forces_numba(positions, tri_indices, tri_x_inv, tri_volumes, masses, 
                           gravity, young_modulus, poisson_ratio, use_cauchy):
    n = positions.shape[0]
    f_global = np.zeros((n, 2), dtype=np.float64)

    # Grawitacja
    for i in range(n):
        f_global[i, 1] -= masses[i] * gravity

    # Sily wewnetrzne elementow
    for t in range(tri_indices.shape[0]):
        idx = tri_indices[t]
        fe = _compute_element_forces_numba(
            positions[idx[0]], positions[idx[1]], positions[idx[2]],
            tri_x_inv[t], tri_volumes[t], young_modulus, poisson_ratio, use_cauchy
        )
        for i in range(3):
            f_global[idx[i], 0] += fe[i, 0]
            f_global[idx[i], 1] += fe[i, 1]

    return f_global

@njit(cache=True)
def _integrate_particles_numba(positions, velocities, masses, fixed_mask, forces, 
                               dt, damping, ground_y, bounce):
    for i in range(positions.shape[0]):
        if fixed_mask[i]:
            velocities[i, :] = 0.0
            continue

        inv_m = 1.0 / masses[i]
        # Predkosc (Semi-implicit)
        velocities[i, 0] = (velocities[i, 0] + forces[i, 0] * inv_m * dt) * damping
        velocities[i, 1] = (velocities[i, 1] + forces[i, 1] * inv_m * dt) * damping
        
        # Pozycja
        positions[i, 0] += velocities[i, 0] * dt
        positions[i, 1] += velocities[i, 1] * dt

        # Kolizja z podloga
        if positions[i, 1] < ground_y:
            positions[i, 1] = ground_y
            velocities[i, 1] *= -bounce
            velocities[i, 0] *= 0.9  # Tarcie

class FEMSoftBody:
    def __init__(self, young_modulus=10000, poisson_ratio=0.15, gravity=5):
        self.E = young_modulus
        self.nu = poisson_ratio
        self.gravity = gravity
        self.time = 0.0
        self.create_mesh()
        self.precompute_rest_config()

    def create_mesh(self):
        grid_size = 8
        spacing = 0.15
        self.particles = []
        for j in range(grid_size):
            for i in range(grid_size):
                pos = np.array([i * spacing - 0.5, j * spacing + 0.5], dtype=np.float64)
                self.particles.append({'pos': pos, 'vel': np.zeros(2), 'fixed': False})

        self.positions = np.array([p['pos'] for p in self.particles])
        self.velocities = np.zeros_like(self.positions)
        self.masses = np.ones(len(self.positions)) * 1.0
        self.fixed_mask = np.zeros(len(self.positions), dtype=bool)

        tris = []
        for j in range(grid_size - 1):
            for i in range(grid_size - 1):
                n = j * grid_size + i
                tris.append([n, n+1, n+grid_size])
                tris.append([n+1, n+grid_size+1, n+grid_size])
        self.tri_indices = np.array(tris)

    def precompute_rest_config(self):
        self.tri_x_inv = []
        self.tri_vol = []
        for tri in self.tri_indices:
            p0, p1, p2 = self.positions[tri]
            X = np.column_stack([p1 - p0, p2 - p0])
            self.tri_vol.append(0.5 * abs(np.linalg.det(X)))
            self.tri_x_inv.append(np.linalg.inv(X))
        self.tri_x_inv = np.array(self.tri_x_inv)
        self.tri_vol = np.array(self.tri_vol)

    def update(self, dt):
        self.time += dt
        f = _assemble_forces_numba(self.positions, self.tri_indices, self.tri_x_inv, 
                                   self.tri_vol, self.masses, self.gravity, 
                                   self.E, self.nu, False)
        
        # Początkowy impuls rotacyjny
        if self.time < 0.1:
            center = np.mean(self.positions, axis=0)
            for i in range(len(self.positions)):
                r = self.positions[i] - center
                f[i] += np.array([-r[1], r[0]]) * 150.0

        _integrate_particles_numba(self.positions, self.velocities, self.masses, 
                                   self.fixed_mask, f, dt, 0.999, -1.5, 0.5)
        self.last_forces = f

def main():
    sim = FEMSoftBody()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3); ax.set_ylim(-2, 3)
    ax.axhline(-1.5, color='k', lw=2)
    
    patches = [Polygon(sim.positions[t], fc='cyan', ec='blue', alpha=0.6) for t in sim.tri_indices]
    for p in patches: ax.add_patch(p)

    def animate(_):
        # Substepping: 10 małych kroków na klatkę dla stabilności
        for _ in range(10): sim.update(0.002)
        for i, p in enumerate(patches):
            p.set_xy(sim.positions[sim.tri_indices[i]])
        return patches

    ani = FuncAnimation(fig, animate, frames=500, interval=20, blit=True)
    plt.show()

if __name__ == "__main__":
    main()