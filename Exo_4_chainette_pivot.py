# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:09:14 2024

@author: renaudf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import Multicorps2D as mc

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
cmap = plt.colormaps["viridis"]
cmap = plt.colormaps["gist_rainbow"]
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

# Création des pivots
# Entre le premier solide et le galiléen
modele.add_interaction("Hinge_1sFix", tag_I="pivot_0",
                       tag_S="S0", tag_F="P1_0", P_Gal=Fix_Gal)
# Entre les solides successifs
for ind in range(0, Ns-1):
    tagSa = "S"+str(ind)
    tagFa = "P2_"+str(ind)
    tagSb = "S"+str(ind+1)
    tagFb = "P1_"+str(ind+1)
    tag_I = "pivot_" + str(ind) + "_" + str(ind+1)
    modele.add_interaction("Hinge_2s", tag_I=tag_I,
                           tag_S=[tagSa, tagSb], tag_F=[tagFa, tagFb])

sol = modele.transient(0, 1.5, 0.01)
t, res_solids, res_inter = modele.transient_post(sol)
modele.plot_state(sol)

fig, ax = modele.draw_t0()
anim = modele.animate(sol)
# # To save the animation using Pillow as a gif
# writer = ani.PillowWriter(fps=15,
#                           metadata=dict(artist='Me'),
#                           bitrate=1800)
# anim.save('Chainette.gif', writer=writer)
