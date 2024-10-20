# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 14:12:27 2024

@author: renaudf
"""

import numpy as np


def rotation(q):
    """
    Parameters
    ----------
    q : numpy array (3, )

    Returns
    -------
    R : numpy array (3, 3)
    """
    theta = q[2]
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, s], [-s, c]])
    return R


def position(q, GP_Rs):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    GP_Rs : numpy array (2, )

    Returns
    -------
    P : numpy array (2, )
    """
    R = rotation(q)
    P = q[0:2] + R @ GP_Rs
    return P


def rotation_v(q):
    """
    Parameters
    ----------
    q : numpy array (3, )

    Returns
    -------
    R_v : numpy array (3, 3)
    """
    theta = q[2]
    c = np.cos(theta)
    ms = - np.sin(theta)
    R_v = np.array([[ms, c], [-c, ms]])
    return R_v


def transformation_v(q, GP_Rs):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    GP_Rs : numpy array (2, )

    Returns
    -------
    T_v : numpy array (2, 3)
    """
    Id2 = np.eye(2)
    temp = (rotation_v(q) @ GP_Rs)[:, np.newaxis]
    T_v = np.concatenate([ Id2, temp ], axis=1 )
    return T_v


def velocity(q, dq, GP_Rs):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    dq    : numpy array (3, )
    GP_Rs : numpy array (2, )

    Returns
    -------
    dP : numpy array (2, )
    """
    dR = dq[2] * rotation_v(q)
    dP = dq[0:2] + dR @ GP_Rs
    return dP


def rotation_a(q):
    """
    Parameters
    ----------
    q : numpy array (3, )

    Returns
    -------
    R_a : numpy array (3, 3)
    """
    R_a = - rotation(q)
    return R_a


def transformation_a(q, dq, GP_Rs):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    dq    : numpy array (3, )
    GP_Rs : numpy array (2, )

    Returns
    -------
    T_a : numpy array (2, 3)
    """
    Id2 = np.zeros((2,2))
    temp = (dq[2] * rotation_a(q) @ GP_Rs)[:, np.newaxis]
    T_a = np.concatenate([ Id2, temp], axis=1 )
    return T_a


def acceleration(q, dq, d2q, GP_Rs):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    dq    : numpy array (3, )
    d2q   : numpy array (3, )
    GP_Rs : numpy array (2, )

    Returns
    -------
    d2P : numpy array (2, )
    """
    d2R = d2q[2] * rotation_v(q) + dq[2]**2 * rotation_a(q)
    d2P = d2q[0:2] + d2R @ GP_Rs
    return d2P


def deflection(q_a, dq_a, GP_a, q_b, dq_b, GP_b):
    # Relative displacement
    AB = position(q_b, GP_b) - position(q_a, GP_a)
    # Relative velocity
    d_AB = velocity(q_b, dq_b, GP_b) - velocity(q_a, dq_a, GP_a)
    return AB, d_AB


def load_direction(AB, d_AB):
    # Direction of the force
    Long_AB = np.linalg.norm(AB)
    # Normal behaviour of non-null spring length
    if Long_AB != 0:
        u = AB / Long_AB
    # Strange behaviour where A and B are coincident
    else:
        d_Long_AB = np.linalg.norm(d_AB)
        # If the velocity is non-null
        if d_Long_AB != 0:
            u = np.array([d_AB[1], - d_AB[0]]) / d_Long_AB
        # Most improbable case
        else:
            u = np.random.rand(2)
            u = u / np.linalg.norm(u)
    return u


def spring_damper_force(q_a, dq_a, GP_a,
                        q_b, dq_b, GP_b,
                        k, c, L0):
    # Deflection of the spring and dahspot
    AB, d_AB = deflection(q_a, dq_a, GP_a, q_b, dq_b, GP_b)
    # Direction of the force
    u = load_direction(AB, d_AB)
    # deflection
    Long_AB = np.linalg.norm(AB)
    delta   = Long_AB - L0
    d_delta = u @ d_AB
    # Force of the spring and the dashpot on point A
    F_BonA = ( k * delta + c * d_delta ) * u
    return F_BonA


def virtual_power(q, GP_Rs, Fext=np.zeros(2), Mext=0):
    """
    Parameters
    ----------
    q     : numpy array (3, )
    GP_Rs : numpy array (2, )
    Fext  : numpy array (2, ), optional
        The default is np.zeros(2).
    Mext : scalar, optional
        The default is 0.

    Returns
    -------
    B : numpy array (3, )
    """
    T_v = transformation_v(q, GP_Rs)
    B = T_v.T @ Fext + np.array([0, 0, Mext])
    return B


def tabular(t, t_tab, f_tab):
    f_ext = np.zeros(2)
    f_ext[0] = np.interp(t, t_tab, f_tab[0])
    f_ext[1] = np.interp(t, t_tab, f_tab[1])
    return f_ext


if __name__ == "__main__":
    Qs = np.array([1, 2, np.pi/2])
    GP_Rs = np.array([1, 2])
    print(rotation(Qs))
    print(position(Qs, GP_Rs))
