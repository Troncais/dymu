# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 15:09:14 2024

@author: renaudf
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import Multicorps2D as mc

X_0 = np.array([1, 0])
P = np.array([0, 0])
G = np.array([1, 0])
modele = mc.Model()
modele.add_solid(tag_S="S1", label="S1", m=1, iy=1e-2, G_t0=G, Xs=X_0)
modele.add_frame(tag_S="S1", tag_F="P", label="P", P_t0=P, Xs=X_0)
modele.add_interaction("Hinge_1sFix", tag_I="pivot",
                       tag_S="S1", tag_F="P", P_Gal=P)

sol = modele.transient(0, 5, 0.01)
t, res_solids, res_inter = modele.transient_post(sol)
modele.plot_state(sol)
anim = modele.animate(sol)
# # To save the animation using Pillow as a gif
# writer = ani.PillowWriter(fps=15,
#                           metadata=dict(artist='Me'),
#                           bitrate=1800)
# anim.save('PenduleSimple.gif', writer=writer)
