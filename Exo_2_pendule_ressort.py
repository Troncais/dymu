# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 12:16:22 2024

@author: renaudf
"""

import numpy as np
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from utils2D import rotation, position, \
    spring_damper_force, virtual_power

# Position des points dans le repère galiléen
OG_0 = np.array([1, 3])
OP_0 = np.array([2, 4])
OA_0 = np.array([3, 2])

# Orientation initial du solide
GP_0 = OP_0 - OG_0
X_S = GP_0 / np.linalg.norm(GP_0)
theta_0 = - np.arctan2(X_S[1], X_S[0])

# Etat initial du solide
q_0 = np.concatenate((OG_0, [theta_0]))

# Position du point P dans le repère du solide
R_0S = rotation(q_0)
GP_S = R_0S.T @ GP_0

# matrice de masse et vecteur de pesanteur
m = 1
Iy = 0.01
g = 9.81
M = np.diag([m, m, Iy])
B_poids = np.array([0, -m*g, 0])

# Loi de comportement du ressort/amortisseur
k = 1e3
c = 1e2
L0 = 0.5

# spring_damper_force() renvoie F_BonA, ici le point A demandé
# par spring_damper_force() correspond au point P du solide S
# et le point B sera attaché au repère Galiléen
GP_a = GP_S
q_b = np.concatenate((OA_0, [0]))
dq_b = np.zeros(3)
GP_b = np.zeros(2)

# Intégration temporelle avec Runge Kutta
# Liste des arguments supplémentaires à passer à la fonction fun
args = (M, B_poids, GP_a, q_b, dq_b, GP_b, k, c, L0)
def fun(t, y, M, B_poids, GP_a, q_b, dq_b, GP_b, k, c, L0):
    q_a = y[0:3]
    dq_a = y[3:]
    F_BonA = spring_damper_force(q_a, dq_a, GP_a,
                                 q_b, dq_b, GP_b,
                                 k, c, L0)
    B_ressort = virtual_power(q_a, GP_a, Fext=F_BonA)
    B = B_poids + B_ressort
    d2q = np.linalg.solve(M, B)
    dy = np.concatenate((y[3:6], d2q))
    return dy

Ne = 1000
tf = 5
t = np.linspace(0, tf, Ne)
y0 = np.concatenate( (q_0, np.zeros(3)) )
sol = sci.solve_ivp(fun, [0, tf], y0, t_eval=t, args=args,
                    max_step=np.inf, rtol=1e-3, atol=1e-6)

# Résultats
t = sol.t
x = sol.y[0]
z = sol.y[1]
theta = sol.y[2]

# Evolution temporelle
ax_x = plt.subplot(3, 1, 1)
ax_x.grid()
ax_x.plot(t, x, label='solve ivp')
ax_x.set_ylabel('Position x [m]')

ax_z = plt.subplot(3, 1, 2)
ax_z.grid()
ax_z.plot(t, z, label='solve ivp')
ax_z.set_ylabel('Altitude z [m]')

ax_theta = plt.subplot(3, 1, 3)
ax_theta.grid()
ax_theta.plot(t, theta, label='solve ivp')
ax_theta.set_xlabel('Temps [s]')
ax_theta.set_ylabel('Angle [rad]')

# Animation
fig, ax_anim = plt.subplots()
ax_anim.axis('equal')
ax_anim.set_xlim(left=OA_0[0]-3, right=OA_0[0]+3)
ax_anim.set_ylim(bottom=OA_0[1]-3, top=OA_0[1]+3)
ax_anim.grid()
titre = ax_anim.set_title("t = {:.3f}s".format(0))
curv1, = ax_anim.plot(np.zeros(2), np.zeros(2),
                      marker='^', label="solid")
curv2, = ax_anim.plot(np.zeros(2), np.zeros(2),
                      marker='+', label="ressort")
ax_anim.legend()

# Creating an animation
def animate_fct(ind):
    titre.set_text("t = {:.3f}s".format(t[ind]))
    OP_t = position(sol.y[0:3, ind], GP_S)
    # Animation du solide : segment GP_0
    x1 = np.array( [ x[ind], OP_t[0] ] )
    z1 = np.array( [ z[ind], OP_t[1] ] )
    curv1.set_data(x1, z1)
    # Animation du ressort : segment AP_0
    x2 = np.array( [ OA_0[0], OP_t[0] ] )
    z2 = np.array( [ OA_0[1], OP_t[1] ] )
    curv2.set_data(x2, z2)
    return curv1, curv2


anim = ani.FuncAnimation(fig=fig, func=animate_fct,
                         frames=len(t)-1, interval=10)

# # To save the animation using Pillow as a gif
# writer = ani.PillowWriter(fps=15,
#                           metadata=dict(artist='Me'),
#                           bitrate=1800)
# anim.save('Animation.gif', writer=writer)


# =============================================================================
# Avec le code multicorps
# =============================================================================
import Multicorps2D as mc

modele = mc.Model()
modele.add_solid(tag_S="S1", label="S1", m=m, iy=Iy,
                 G_t0=OG_0, Xs=OP_0-OG_0,
                 dG_t0=np.zeros(2), dtheta_t0=0,
                 color=[0, 0, 1], marker='+')
modele.add_frame(tag_S="S1", tag_F="P", label="P",
                 P_t0=OP_0, Xs=OP_0-OG_0)
modele.add_interaction("SpringDashpot_1sFix", tag_I="ressort",
                       tag_S="S1", tag_F="P",
                       P_Gal=OA_0, k=k, c=c, L0=L0)
t0 = 0
dt = tf/Ne
sol2 = modele.transient(t0, tf, dt)
t, res_solids, res_inter = modele.transient_post(sol2)

# Figures
ax_x.plot(sol2.t, sol2.y[0], '^', label='transient')
ax_z.plot(sol2.t, sol2.y[1], '^', label='transient')
ax_theta.plot(sol2.t, sol2.y[2], '^', label='transient')
ax_x.legend()
ax_z.legend()
ax_theta.legend()
