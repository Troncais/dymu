# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 11:15:31 2024

@author: renaudf
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as sci

# Parametres
Ne = 50
g = 9.81

# Solution theorique
t = np.linspace(0, 0.25, Ne)
x = t
z = - 1/2 * g * t**2 + t
theta = np.zeros(Ne)

# matrice de masse et vecteur de pesanteur
m = 1
Iy = 1
M = np.diag([m, m, Iy])
B = np.array([0, -m*g, 0])
args = (M, B)

# Derivee de l'etat
def fun(t, y, M, B):
    d2q = np.linalg.solve(M, B)
    dy = np.concatenate((y[3:6], d2q))
    return dy

# Integration temporelle avec Runge Kutta
t_span = [0, 0.25]
y0 = np.array([0, 0, 0, 1, 1, 0])
sol = sci.solve_ivp(fun, t_span, y0, t_eval=t, args=args)
t_rk = sol.t
x_rk = sol.y[0]
z_rk = sol.y[1]
theta_rk = sol.y[2]

# Resultats
ax_x = plt.subplot(3, 1, 1)
ax_x.grid()
ax_x.plot(t, x, label='theorie')
ax_x.plot(t_rk, x_rk, 'o', label='solve ivp')

ax_z = plt.subplot(3, 1, 2)
ax_z.grid()
ax_z.plot(t, z, label='theorie')
ax_z.plot(t_rk, z_rk, 'o', label='solve ivp')

ax_theta = plt.subplot(3, 1, 3)
ax_theta.grid()
ax_theta.plot(t, theta, label='theorie')
ax_theta.plot(t_rk, theta_rk, 'o', label='solve ivp')

# =============================================================================
# Avec le code multicorps
# =============================================================================
import Multicorps2D as mc

modele = mc.Model()
modele.add_solid(m=m, iy=Iy)
modele.solids["solid"].dq_t0 = np.array([1, 1, 0])
t0 = 0
tf = 0.25
dt = tf/Ne
sol2 = modele.transient(t0, tf, dt)

ax_x.plot(sol2.t, sol2.y[0], '^', label='transient')
ax_z.plot(sol2.t, sol2.y[1], '^', label='transient')
ax_theta.plot(sol2.t, sol2.y[2], '^', label='transient')
ax_x.legend()
ax_z.legend()
ax_theta.legend()

