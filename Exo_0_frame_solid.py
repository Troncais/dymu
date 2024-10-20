# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:14:23 2024

@author: renaudf
"""

from utils2D import *
import numpy as np
import Multicorps2D as mc
import matplotlib.pyplot as plt

# Definition of points in the galilean frame
pts = {}
pts['G1'] = np.array([1, 1])
pts['G2'] = np.array([1, 2])
pts['G3'] = np.array([2, 2])
pts['G4'] = np.array([2, 1])
pts['A'] = np.array([0, 0])
pts['B'] = np.array([2, 3])
pts['C'] = np.array([0, 4])
pts['D'] = np.array([4, 0.5])
pts['E'] = np.array([0, 2])
pts['F'] = np.array([2.5, 0])

# Creation of solid 1
s1 = mc.Solid(label="Carter", G_t0=pts['G1'])
s1.add_frame(tag_F="A", label="Axe pivot 1", P_t0=pts['A'])
s1.add_frame(tag_F="B", label="Axe pivot 2", P_t0=pts['B'])
s1.add_frame(tag_F="C", label="Rotule 1", P_t0=pts['C'])
s1.add_frame(tag_F="D", label="Capteur", P_t0=pts['D'])

# Creation of solid 2
s2 = mc.Solid(label="Bielle", G_t0=pts['G2'],
              color=[0, 0.7, 0], marker='o')
s2.add_frame(tag_F="A", label="A", P_t0=pts['A'])
s2.add_frame(tag_F="C", label="C", P_t0=pts['C'])
s2.add_frame(tag_F="E", label="E", P_t0=pts['E'])
s2.add_frame(tag_F="F", label="F", P_t0=pts['F'])

# Creation of solid 2
s3 = mc.Solid(label="Roue dentée", G_t0=pts['G3'],
              color=[0.75, 0.75, 0], marker='s')
s3.add_frame(tag_F="D", label="D", P_t0=pts['D'])
s3.add_frame(tag_F="G2", label="G2", P_t0=pts['G2'])
s3.add_frame(tag_F="G4", label="G4", P_t0=pts['G4'])

# PLotting the solids
fig, ax = plt.subplots()
ax.grid()
s1.draw_t0(ax)
s2.draw_t0(ax)
s3.draw_t0(ax)
ax.legend()

q = np.array([1, 0, np.pi/2])
dq = np.array([0, 1, 1])
d2q = np.array([1, 0, 1])
GP_Rs = np.array([10, 10])

print(rotation(q))
print(position(q, GP_Rs))

print(rotation_v(q))
print(transformation_v(q, GP_Rs))
print(velocity(q, dq, GP_Rs))

print(rotation_a(q))
print(transformation_a(q, dq, GP_Rs))
print(acceleration(q, dq, d2q, GP_Rs))

print(s1)
print(s1.frames['A'])
