# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:09:14 2024

@author: renaudf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import Multicorps2D as mc
from utils2D import rotation

# Initialisation du modèle
modele = mc.Model()

# Coordonnées du point de fixation de la chainette dans le galiléen
Fix_Gal = np.array([2, 5])
# Nombre de maillons
Ns = 25
# Initialement la chainette est tendu dans la direction Xs
Xs = np.array([1, 0])/Ns

# Echelle de couleur
cmap = plt.colormaps["terrain"]
map = cmap(np.linspace(0, 1, Ns))

# Création des solides
for ind in range(0, Ns):
    tagS = "S"+str(ind)
    tagF1 = "P1_"+str(ind)
    tagF2 = "P2_"+str(ind)
    P1 = Fix_Gal + ind * Xs
    G  = Fix_Gal + (ind+0.5) * Xs
    P2 = Fix_Gal + (ind+1) * Xs
    couleur = map[ind, 0:3]
    modele.add_solid(tag_S=tagS, label=tagS, m=1e-2, iy=1e-9,
                     G_t0=G, Xs=Xs, color=couleur, marker='+')
    modele.add_frame(tag_S=tagS, tag_F=tagF1, label=tagF1, P_t0=P1, Xs=Xs)
    modele.add_frame(tag_S=tagS, tag_F=tagF2, label=tagF2, P_t0=P2, Xs=Xs)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Déplacement imposé par un sinus
t_final = 1.25
ts = np.linspace(0, t_final, 100)
Ampl = 0.15
f = 0.5
omega = 2 * np.pi * f
theta = omega * ts
s   =              Ampl * np.sin(theta)
ds  =   omega    * Ampl * np.cos(theta)
d2s = - omega**2 * Ampl * np.sin(theta)
modele.add_interaction("ImposedDispl_Z", tag_I="ImposedDispl_Z", tag_S="S10",
                       t=ts, z=s, dz=ds, d2z=d2s)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# Création des pivots
# Entre les solides successifs
for ind in range(0, Ns-1):
    tagSa = "S"+str(ind)
    tagFa = "P2_"+str(ind)
    tagSb = "S"+str(ind+1)
    tagFb = "P1_"+str(ind+1)
    tag_I = "pivot_" + str(ind) + "_" + str(ind+1)
    modele.add_interaction("Hinge_2s", tag_I=tag_I, label=tag_I,
                           tag_S=[tagSa, tagSb], tag_F=[tagFa, tagFb])

sol = modele.transient(0, t_final, 0.01)
t, res_solids, res_inter = modele.transient_post(sol)
modele.plot_state(sol)

fig, ax = modele.draw_t0()
anim = modele.animate(sol)
# # To save the animation using Pillow as a gif
# writer = ani.PillowWriter(fps=15,
#                           metadata=dict(artist='Me'),
#                           bitrate=1800)
# anim.save('Chainette.gif', writer=writer)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Effort des multiplicateurs de Lagrange du déplacement imposé
tag_I = 'ImposedDispl_Z'
Lambda_ImpDisp = res_inter[tag_I].Lambda

fig2 = plt.figure()
axL = plt.subplot(2, 1, 1)
axL.grid()
axL.set_title('Multiplicateurs de Lagrange de l''interaction '
              + modele.interactions[tag_I].label )
axL.set_xlabel('Temps [s]')
axL.set_ylabel('Effort [N]')
axL.plot(t, Lambda_ImpDisp[0], label='X_galiléen')
axL.plot(t, Lambda_ImpDisp[1], label='Z_galiléen')
axL.legend()

axL = plt.subplot(2, 1, 2)
axL.grid()
axL.set_title('Multiplicateurs de Lagrange de l''interaction '
              + modele.interactions[tag_I].label )
axL.set_xlabel('Temps [s]')
axL.set_ylabel('Couple [N.m]')
axL.plot(t, Lambda_ImpDisp[2], label='theta')
axL.legend()

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Effort des multiplicateurs de Lagrange d'une pivot
tag_I = 'pivot_23_24'
tag_Sa = 'S23'
tag_Sb = 'S24'
Lambda_pivot_gal = res_inter[tag_I].Lambda
Lambda_pivot_sa = np.zeros(np.shape(Lambda_pivot_gal))
Lambda_pivot_sb = np.zeros(np.shape(Lambda_pivot_gal))
q_sa = res_solids[tagSa].q
q_sb = res_solids[tagSb].q
for ind in range(0, len(t)):
    Lambda_pivot_sa[:, ind] = rotation(q_sa[:, ind]).T @ Lambda_pivot_gal[:, ind]
    Lambda_pivot_sb[:, ind] = rotation(q_sb[:, ind]).T @ Lambda_pivot_gal[:, ind]

fig = plt.figure()
ax = plt.subplot(2, 1, 1)
ax.grid()
ax.set_title('Multiplicateurs de Lagrange de l''interaction '
              + modele.interactions[tag_I].label )
ax.set_xlabel('Temps [s]')
ax.set_ylabel('Effort [N]')
ax.plot(t, Lambda_pivot_gal[0], label='X_galiléen')
ax.plot(t, Lambda_pivot_gal[1], label='Z_galiléen')

ax = plt.subplot(2, 1, 2)
ax.grid()
ax.set_title('Multiplicateurs de Lagrange de l''interaction '
              + modele.interactions[tag_I].label )
ax.set_xlabel('Temps [s]')
ax.set_ylabel('Effort [N]')
ax.plot(t, Lambda_pivot_sa[0], label='X_'+tag_Sa)
ax.plot(t, Lambda_pivot_sa[1], label='Z_'+tag_Sa)
ax.plot(t, Lambda_pivot_sb[0], label='X_'+tag_Sb)
ax.plot(t, Lambda_pivot_sb[1], label='Z_'+tag_Sb)
ax.legend()
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
