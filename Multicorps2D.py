# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:30:42 2024

@author: renaudf
"""

import numpy as np
import scipy.linalg as scl
import scipy.integrate as sci
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from utils2D import rotation, position, velocity, transformation_v, transformation_a, spring_damper_force, virtual_power


class Frame:
    def __init__(self, label="frame",
                 P=np.zeros(2), theta=0):
        self.label = label
        self.position = P
        self.theta = theta

    def __repr__(self):
        s  = "'{}'\n".format(self.label)
        temp = "\tx = {:+.3f} m, \tz = {:+.3f} m, " +\
            "\ttheta = {:+.3f} rad\n"
        s += temp.format(self.position[0], self.position[1],
                         self.theta)
        return s


class Solid:
    def __init__(self, label="solid", m=1.0, iy=0.01,
                 G_t0=np.zeros(2), theta_t0=0,
                 dG_t0=np.zeros(2), dtheta_t0=0,
                 color=[0, 0, 1], marker='+'):
        self.label = label
        self.m = m
        self.iy = iy

        self.q_t0 = np.concatenate((G_t0, [theta_t0]))
        self.dq_t0 = np.concatenate((dG_t0, [dtheta_t0]))
        self.frames = {"G": Frame(label="G")}

        self.color = color
        self.marker = marker

    def add_frame(self, tag_F="frame", label="frame",
                  P_t0=np.zeros(2), Xs=np.array([1., 0.])):
        R = rotation(self.q_t0)
        P_Rs = R.T @ (P_t0 - self.q_t0[0:2])
        theta = np.arctan2(-Xs[1], Xs[0]) - self.q_t0[2]
        self.frames[tag_F] = Frame(label=label, P=P_Rs, theta=theta)

    def draw_t0(self, ax):
        Nb = len(self.frames)
        G = self.q_t0[0:2]
        P = np.zeros((2, 2*Nb))
        ind = 0
        for item, value in self.frames.items():
            P[:, ind] = G
            P[:, ind+1] = position(self.q_t0, value.position)
            ind += 2
        self._curv, = ax.plot(P[0], P[1], label=self.label,
                             color=self.color, marker=self.marker,
                             fillstyle='none', markersize=8)

    def _draw_update(self, q):
        Nb = len(self.frames)
        G = q[0:2]
        P = np.zeros((2, 2*Nb))
        ind = 0
        for item, value in self.frames.items():
            P[:, ind] = G
            P[:, ind+1] = position(q, value.position)
            ind += 2
        return P

    def __repr__(self):
        s  = "'{}' :\n".format(self.label)
        temp = "\tx_0 = {:+.3f} m, \tz_0 = {:+.3f} m, " + \
            "\ttheta_0 = {:+.3f} rad\n"
        s += temp.format(self.q_t0[0], self.q_t0[1], self.q_t0[2])
        s  += "\tm = {:.3f} kg, \tiy = {:.3f} kg.m^2\n" \
            .format(self.m, self.iy)
        for key, value in self.frames.items():
            s += "\t.frames[{}] = ".format(key)
            s += value.__repr__().replace("\n\t", "\n\t\t")
        return s

    def mass(self):
        """
        Returns
        -------
        M : numpy array (3, 3)
        """
        m = self.m
        iy = self.iy
        M = np.diag([m, m, iy])
        return M

    def weight(self, g):
        """
        Parameters
        ----------
        g : scalar, 9.81 m/s^2

        Returns
        -------
        B : numpy array (3, )
        """
        m = self.m
        B = np.array([0, -m*g, 0])
        return B

    def plot_state(self, sol, ax_x, ax_z, ax_t):
        t = sol.t
        x = sol.y[self.ind_q[0]]
        z = sol.y[self.ind_q[1]]
        theta = sol.y[self.ind_q[2]]
        ax_x.plot(t, x, label=self.label, color=self.color,
                  marker=self.marker, fillstyle='none', markersize=8)
        ax_z.plot(t, z, label=self.label, color=self.color,
                  marker=self.marker, fillstyle='none', markersize=8)
        ax_t.plot(t, theta, label=self.label, color=self.color,
                  marker=self.marker, fillstyle='none', markersize=8)


class Param:
    def __init__(self):
        self.g = 9.81
        self.bmgt_xi = 1
        self.bmgt_w0 = 2*np.pi*10


class Matrices:
    def __init__(self):
        pass

    def copy(self):
        new = Matrices()
        for attribute, value in self.__dict__.items():
            setattr(new, attribute, value.copy())
        return new


class Plot_mpl:
    def __init__(self):
        pass


class Model:
    def __init__(self):
        # Default parameters of the model
        self.param = Param()

        # Empty dictionaries of solids
        self.solids = {}

        # Empty dictionaries of interactions
        self.interactions = {}

        # Number of
        self.n_s = 0  # solids
        self.n_q = 0  # dof (degree of freedom)
        self.n_i = 0  # interactions
        self.n_c = 0  # constraints

        # Some matrices for later computation
        self._matrices = Matrices()

        # Plotting pointers
        self._plt = Plot_mpl()

    def add_solid(self, tag_S="solid", label="solid", m=1.0, iy=0.01,
                  G_t0=np.zeros(2), Xs=np.array([1., 0.]),
                  dG_t0=np.zeros(2), dtheta_t0=0,
                  color=[0, 0, 1], marker='+'):
        theta_t0 = np.arctan2(-Xs[1], Xs[0])
        self.solids[tag_S] = Solid(label=label, m=m, iy=iy,
                                   G_t0=G_t0, theta_t0=theta_t0,
                                   dG_t0=dG_t0, dtheta_t0=dtheta_t0,
                                   color=color, marker=marker)

    def add_frame(self, tag_S="solid", tag_F="frame", label="frame",
                  P_t0=np.zeros(2), Xs=np.array([1., 0.])):
        self.solids[tag_S].add_frame(tag_F=tag_F, label=label,
                                     P_t0=P_t0, Xs=Xs)

    def add_interaction(self, kind, tag_I="interaction", **kwargs):
        if kind == "SpringDashpot_2s":
            inter = SpringDashpot_2s(**kwargs)
        elif kind == "SpringDashpot_1sFix":
            inter = SpringDashpot_1sFix(**kwargs)
        elif kind == "Hinge_2s":
            inter = Hinge_2s(**kwargs)
        elif kind == "Hinge_1sFix":
            inter = Hinge_1sFix(**kwargs)
        elif kind == "ImposedDispl_Z":
            inter = ImposedDispl_Z(**kwargs)
        # Creation of the interaction
        self.interactions[tag_I] = inter

    def _odefun(self, t, y):
        mat = self._matrices.copy()
        ind_q = mat.ind_q
        ind_dq = mat.ind_dq

        # Sub-matrices assembly from interactions
        for i in self.interactions.values():
            mat = i.assemble(t, y, mat)
        # Matrices assembly
        matrice = np.block([[mat.A, mat.C.T], [mat.C, mat.ZerC]])
        vecteur = np.concatenate((mat.B, mat.D))

        # d2q = inv(M) * F
        q_total = np.linalg.solve(matrice, vecteur)
        dy = np.concatenate((y[ind_dq], q_total[ind_q]))
        return dy

    def _odefun_post(self, t, y):
        mat = self._matrices.copy()
        ind_q = mat.ind_q

        # Sub-matrices assembly from interactions
        for i in self.interactions.values():
            mat = i.assemble(t, y, mat)
        # Matrices assembly
        matrice = np.block([[mat.A, mat.C.T], [mat.C, mat.ZerC]])
        vecteur = np.concatenate((mat.B, mat.D))

        # d2q = inv(M) * F
        q_total = np.linalg.solve(matrice, vecteur)
        d2q = q_total[ind_q]
        Lambda = q_total[self.n_q:]
        return d2q, Lambda

    def transient(self, t0, tf, dt,
                  method='RK45', max_step=np.inf,
                  rtol=1e-3, atol=1e-6):
        # Updating of the model before time integration
        self.build()

        # Parameters for time integration
        fun = self._odefun
        t_span = [t0, tf]
        y0 = self._matrices.y0
        t_eval = np.arange(t0, tf, dt)

        # Time integration itself
        print("\nStarting solve_ivp ...")
        sol = sci.solve_ivp(fun, t_span, y0,
                            method=method,
                            t_eval=t_eval,  # vectorized=False,
                            max_step=max_step,
                            rtol=rtol, atol=atol)
        print("\t\t\t... Ending solve_ivp")

        return sol

    def transient_post(self, sol):
        print("\nStarting post-processing ...")
        t = sol.t
        y = sol.y
        Nt = len(t)

        ind_q  = self._matrices.ind_q
        ind_dq = self._matrices.ind_dq

        # Initialisation
        q = y[ind_q, :]
        dq = y[ind_dq, :]
        d2q = np.zeros((self.n_q, Nt))
        Lambda = np.zeros((self.n_c, Nt))

        # Calcul des accélérations et des multicplicateurs de Lagrange
        for ind in range(0, Nt):
            d2q[:, ind], Lambda[:, ind] = self._odefun_post(t[ind], y[:, ind])

        # Compilation des résultats par solide
        res_solids = dict()
        for key, s in self.solids.items():
            res_solids[key] = Matrices()
            res_solids[key].q   =   q[s.ind_q, :]
            res_solids[key].dq  =  dq[s.ind_q, :]
            res_solids[key].d2q = d2q[s.ind_q, :]

        # Compilation des résultats par interaction
        res_inter = dict()
        for key, i in self.interactions.items():
            res_inter[key] = Matrices()
            res_inter[key].Lambda = Lambda[i.ind_c, :]
            res_inter[key] = i.assemble_post(t, y, res_inter[key])

        print("\t\t\t... Ending post-processing")
        return t, res_solids, res_inter

    def build(self):
        # Updating of :
        # + the indices of position dofs and velocity dofs of solids,
        # + matrices A, B, C and D
        # + interactions
        # + initial conditions
        self._build_solids()
        self._build_interactions()
        self._build_initial_conditions()

    def _build_solids(self):
        # Number of solids in the model
        self.n_s = len(self.solids)
        self.n_q = 0  # Length of vector q

        # Initialisation of matrices A and B
        A = []
        B = []
        ind_q = []
        for s in self.solids.values():
            # List of mass matrices to be concatenated
            M = s.mass()
            A.append(M)
            # List gravity vectors to be concatenated
            F = s.weight(self.param.g)
            B.append(F)
            # Indices of the position dofs of the current solid
            s.ind_q = self.n_q + np.arange(0, 3)
            ind_q.append(s.ind_q)
            # Updating of the total number of position dofs
            self.n_q += 3

        # Indices of the velocity dofs
        ind_dq = []
        for s in self.solids.values():
            s.ind_dq = self.n_q + s.ind_q
            ind_dq.append(s.ind_dq)

        # Matrices A, B, C and D
        self._matrices.A        = scl.block_diag(*A)
        self._matrices.B        = np.concatenate(B)
        self._matrices.y0       = np.zeros(2*self.n_q)
        self._matrices.y        = np.zeros(2*self.n_q)
        self._matrices.ind_q    = np.concatenate(ind_q)
        self._matrices.ind_dq   = np.concatenate(ind_dq)

    def _build_interactions(self):
        # Number of interactions in the model
        self.n_i = len(self.interactions)
        self.n_c = 0  # Length of vector lambda
        for i in self.interactions.values():
            i._build(self.solids, self.param)
            # Indices of the position constraints in matrix C
            i.ind_c = self.n_c + np.arange(0, i.n_c)
            # Updating of the total number of constraints
            self.n_c += i.n_c

        # Matrices A, B, C and D
        n_c = self.n_c
        n_q = self.n_q
        self._matrices.C        = np.zeros((n_c, n_q))
        self._matrices.D        = np.zeros(n_c)
        self._matrices.ZerC     = np.zeros((n_c, n_c))

    def _build_initial_conditions(self):
        y0 = np.zeros( 2 * self.n_q )
        for s in self.solids.values():
            y0[s.ind_q] = s.q_t0
            y0[s.ind_dq] = s.dq_t0
        self._matrices.y0 = y0

    def draw_t0(self):
        # PLotting the solids
        fig, ax = plt.subplots()
        ax.grid()
        for s in self.solids.values():
            s.draw_t0(ax)
        ax.legend()
        return fig, ax

    def animate(self, sol):
        # Calcul des limites de la figure
        Nt = len(sol.t)
        x_min = sol.y[0, 0]
        x_max = sol.y[0, 0]
        z_min = sol.y[1, 0]
        z_max = sol.y[1, 0]
        for s in self.solids.values():
            q = sol.y[s.ind_q, :]
            for f in s.frames.values():
                GP = f.position
                OP = np.zeros((2, Nt))
                for ind in range(0, Nt):
                    OP[:, ind] = position(q[:, ind], GP)
                x_min = np.min( [x_min, np.min(OP[0, :])] )
                x_max = np.max( [x_max, np.max(OP[0, :])] )
                z_min = np.min( [z_min, np.min(OP[1, :])] )
                z_max = np.max( [z_max, np.max(OP[1, :])] )

        x_moy = (x_max + x_min) / 2
        z_moy = (z_max + z_min) / 2
        dx = (x_max - x_min) / 2
        dz = (z_max - z_min) / 2
        d = np.max( [dx, dz] ) * 1.1

        # Création de la figure
        fig, ax = self.draw_t0()
        ax.axis('equal')
        ax.set_xlim(left=x_moy-d, right=x_moy+d)
        ax.set_ylim(bottom=z_moy-d, top=z_moy+d)

        # Stockage des pointeurs
        self._plt.ax = ax
        self._plt.sol = sol
        self._plt.titre = ax.set_title("t = {:.3f}s".format(0))

        # Animation
        anim = ani.FuncAnimation(fig=fig, func=self._animate_fct,
                                 frames=len(sol.t)-1, interval=10)
        return anim

    def _animate_fct(self, ind):
        t = self._plt.sol.t
        y = self._plt.sol.y
        self._plt.titre.set_text("t = {:.3f}s".format(t[ind]))
        for s in self.solids.values():
            q = y[s.ind_q, ind]
            P = s._draw_update(q)
            s._curv.set_data(P[0], P[1])

    def plot_state(self, sol):
        # x
        ax_x = plt.subplot(3, 1, 1)
        ax_x.grid()
        ax_x.set_ylabel('Position x [m]')
        # z
        ax_z = plt.subplot(3, 1, 2)
        ax_z.grid()
        ax_z.set_ylabel('Altitude z [m]')
        # theta
        ax_t = plt.subplot(3, 1, 3)
        ax_t.grid()
        ax_t.set_xlabel('Time [s]')
        ax_t.set_ylabel('Angle [rad]')

        for s in self.solids.values():
            s.plot_state(sol, ax_x, ax_z, ax_t)

        ax_x.legend()
        ax_z.legend()
        ax_t.legend()


class SpringDashpot_2s:
    def __init__(self, label="SpringDashpot_2s",
                 tag_S=["tag_Sa", "tag_Sb"],
                 tag_F=["tag_Fa", "tag_Fb"],
                 k=1e3, c=1, L0=0.5):
        # Common properties
        self.label  = label  # Label of the interaction
        self.n_c    = 0      # Number of contraints equations
        self.tag_S  = tag_S  # Tags of the solids in interaction
        self.tag_F  = tag_F  # Tags of the frames in interaction
        # Specific properties
        # Constitutive law
        self.k = k
        self.c = c
        self.L0 = L0
        # Parameters of Solid A
        self.ind_q_a = []
        self.ind_dq_a = []
        self.GP_a = []
        # Parameters of Solid B
        self.ind_q_b = []
        self.ind_dq_b = []
        self.GP_b = []

    def _build(self, solids, param):
        # Solid Sa
        s_a = solids[self.tag_S[0]]
        f_a = s_a.frames[self.tag_F[0]]
        self.ind_q_a = s_a.ind_q
        self.ind_dq_a = s_a.ind_dq
        self.GP_a = f_a.position
        # Solid Sb
        s_b = solids[self.tag_S[1]]
        f_b = s_b.frames[self.tag_F[1]]
        self.ind_q_b = s_b.ind_q
        self.ind_dq_b = s_b.ind_dq
        self.GP_b = f_b.position

    def assemble(self, t, y, matrices):
        # Parameters of Solid A
        q_a = y[self.ind_q_a]
        dq_a = y[self.ind_dq_a]
        GP_a = self.GP_a
        # Parameters of Solid B
        q_b = y[self.ind_q_b]
        dq_b = y[self.ind_dq_b]
        GP_b = self.GP_b

        # Force of the spring and the dashpot on point A
        F_BonA = spring_damper_force(q_a, dq_a, GP_a,
                                     q_b, dq_b, GP_b,
                                     self.k, self.c, self.L0)

        # Matrix assembly
        ind_a = np.ix_(self.ind_q_a)
        matrices.B[ind_a] += virtual_power(Fext=F_BonA, Mext=0,
                                           q=q_a, GP_Rs=GP_a)
        ind_b = np.ix_(self.ind_q_b)
        matrices.B[ind_b] += virtual_power(Fext=-F_BonA, Mext=0,
                                           q=q_b, GP_Rs=GP_b)
        return matrices

    def assemble_post(self, t, y, res):
        Nt = len(t)
        # Parameters of Solid A
        GP_a = self.GP_a
        # Parameters of Solid B
        GP_b = self.GP_b

        res.F_BonA = np.zeros((2, Nt))
        for ind in range(0, Nt):
            # Parameters of Solid A
            q_a = y[self.ind_q_a, ind]
            dq_a = y[self.ind_dq_a, ind]
            # Parameters of Solid B
            q_b = y[self.ind_q_b, ind]
            dq_b = y[self.ind_dq_b, ind]

            # Force of the spring and the dashpot on point A
            res.F_BonA[:, ind] = spring_damper_force(q_a, dq_a, GP_a,
                                                     q_b, dq_b, GP_b,
                                                     self.k, self.c, self.L0)
        return res


class SpringDashpot_1sFix:
    def __init__(self, label="SpringDashpot_1sFix",
                 tag_S="tag_S", tag_F="tag_F",
                 P_Gal=np.array([0, 0]),
                 k=1e3, c=1, L0=0.5):
        # Common properties
        self.label  = label  # Label of the interaction
        self.n_c    = 0      # Number of contraints equations
        self.tag_S  = tag_S  # Tags of the solids in interaction
        self.tag_F  = tag_F  # Tags of the frames in interaction
        # Specific properties
        # Constitutive law
        self.k = k
        self.c = c
        self.L0 = L0
        # End point of the spring in the Galilean frame
        self.P_Gal = P_Gal
        # Parameters of Solid A
        self.ind_q_a = []
        self.ind_dq_a = []
        self.GP_a = []

    def _build(self, solids, param):
        # Solid Sa
        s_a = solids[self.tag_S]
        f_a = s_a.frames[self.tag_F]
        self.ind_q_a = s_a.ind_q
        self.ind_dq_a = s_a.ind_dq
        self.GP_a = f_a.position

    def assemble(self, t, y, matrices):
        # Parameters of Solid A
        q_a = y[self.ind_q_a]
        dq_a = y[self.ind_dq_a]
        GP_a = self.GP_a
        # Parameters of Solid B
        q_b = np.concatenate((self.P_Gal, [0]))
        dq_b = np.zeros(3)
        GP_b = np.zeros(2)

        # Force of the spring and the dashpot on point A
        F_BonA = spring_damper_force(q_a, dq_a, GP_a,
                                     q_b, dq_b, GP_b,
                                     self.k, self.c, self.L0)

        # Matrix assembly
        ind_a = np.ix_(self.ind_q_a)
        matrices.B[ind_a] += virtual_power(Fext=F_BonA, Mext=0,
                                           q=q_a, GP_Rs=GP_a)
        return matrices

    def assemble_post(self, t, y, res):
        Nt = len(t)
        # Parameters of Solid A
        GP_a = self.GP_a
        # Parameters of Solid B
        q_b = np.concatenate((self.P_Gal, [0]))
        dq_b = np.zeros(3)
        GP_b = np.zeros(2)

        res.F_BonA = np.zeros((2, Nt))
        for ind in range(0, Nt):
            # Parameters of Solid A
            q_a = y[self.ind_q_a, ind]
            dq_a = y[self.ind_dq_a, ind]

            # Force of the spring and the dashpot on point A
            res.F_BonA[:, ind] = spring_damper_force(q_a, dq_a, GP_a,
                                                     q_b, dq_b, GP_b,
                                                     self.k, self.c, self.L0)

        return res


class Hinge_2s:
    def __init__(self, label="Hinge_2s",
                 tag_S=["tag_Sa", "tag_Sb"],
                 tag_F=["tag_Fa", "tag_Fb"]):
        # Common properties
        self.label  = label  # Label of the interaction
        self.n_c    = 2      # Number of contraints equations
        self.tag_S  = tag_S  # Tags of the solids in interaction
        self.tag_F  = tag_F  # Tags of the frames in interaction
        # Specific properties
        # Parameters of Solid A
        self.ind_q_a = []
        self.ind_dq_a = []
        self.GP_a = []
        # Parameters of Solid B
        self.ind_q_b = []
        self.ind_dq_b = []
        self.GP_b = []

    def _build(self, solids, param):
        # Solid Sa
        s_a = solids[self.tag_S[0]]
        f_a = s_a.frames[self.tag_F[0]]
        self.ind_q_a = s_a.ind_q
        self.ind_dq_a = s_a.ind_dq
        self.GP_a = f_a.position
        # Solid Sb
        s_b = solids[self.tag_S[1]]
        f_b = s_b.frames[self.tag_F[1]]
        self.ind_q_b = s_b.ind_q
        self.ind_dq_b = s_b.ind_dq
        self.GP_b = f_b.position
        # Baumgarte parameters
        self.bmgt_2xiw0 = 2 * param.bmgt_xi * param.bmgt_w0
        self.bmgt_w02 = param.bmgt_w0**2

    def assemble(self, t, y, matrices):
        # Parameters of Solid A
        q_a = y[self.ind_q_a]
        dq_a = y[self.ind_dq_a]
        GP_a = self.GP_a
        # Parameters of Solid B
        q_b = y[self.ind_q_b]
        dq_b = y[self.ind_dq_b]
        GP_b = self.GP_b
        # Acceleration terms
        Ca = - transformation_v(q_a, GP_a)
        Cb = + transformation_v(q_b, GP_b)
        Da = transformation_a(q_a, dq_a, GP_a) @ dq_a
        Db = transformation_a(q_b, dq_b, GP_b) @ dq_b
        D = Da - Db
        # Velocity term d_AB
        d_AB = velocity(q_b, dq_b, GP_b) - velocity(q_a, dq_a, GP_a)
        # Position term AB
        AB = position(q_b, GP_b) - position(q_a, GP_a)
        # Matrix assembly
        matrices.C[np.ix_(self.ind_c, self.ind_q_a)] = Ca
        matrices.C[np.ix_(self.ind_c, self.ind_q_b)] = Cb
        matrices.D[self.ind_c] = D - self.bmgt_2xiw0 * d_AB \
            - self.bmgt_w02 * AB
        return matrices

    def assemble_post(self, t, y, res):
        return res


class Hinge_1sFix:
    def __init__(self, label="Hinge_1sFix",
                 tag_S="tag_S", tag_F="tag_F",
                 P_Gal=np.array([1, 1])):
        # Common properties
        self.label  = label  # Label of the interaction
        self.n_c    = 2      # Number of contraints equations
        self.tag_S  = tag_S  # Tags of the solids in interaction
        self.tag_F  = tag_F  # Tags of the frames in interaction
        # Specific properties
        # Point in the Galilean frame
        self.P_Gal = P_Gal
        # Parameters of Solid A
        self.ind_q_a = []
        self.ind_dq_a = []
        self.GP_a = []

    def _build(self, solids, param):
        # Solid Sa
        s_a = solids[self.tag_S]
        f_a = s_a.frames[self.tag_F]
        self.ind_q_a = s_a.ind_q
        self.ind_dq_a = s_a.ind_dq
        self.GP_a = f_a.position
        # Baumgarte parameters
        self.bmgt_2xiw0 = 2 * param.bmgt_xi * param.bmgt_w0
        self.bmgt_w02 = param.bmgt_w0**2

    def assemble(self, t, y, matrices):
        # Parameters of Solid A
        q_a = y[self.ind_q_a]
        dq_a = y[self.ind_dq_a]
        GP_a = self.GP_a
        # Acceleration terms
        Ca = - transformation_v(q_a, GP_a)
        Da = transformation_a(q_a, dq_a, GP_a) @ dq_a
        D = Da
        # Velocity term d_AB
        d_AB = - velocity(q_a, dq_a, GP_a)
        # Position term AB
        AB = self.P_Gal - position(q_a, GP_a)
        # Matrix assembly
        matrices.C[np.ix_(self.ind_c, self.ind_q_a)] = Ca
        matrices.D[self.ind_c] = D - self.bmgt_2xiw0 * d_AB - self.bmgt_w02 * AB
        return matrices

    def assemble_post(self, t, y, res):
        return res


class ImposedDispl_Z:
    def __init__(self, label="ImposedDispl_Z", tag_S="tag_S",
                 t=np.array([0, 1]), z=np.array([0, 1]),
                 dz=np.array([0, 0]), d2z=np.array([0, 0])):
        # Common properties
        self.label = label  # Label of the interaction
        self.n_c   = 3      # Number of contraints equations
        self.tag_S = tag_S  # Tags of the solids in interaction
        # Parameters of Solid A
        self.ind_q_a = []
        self.ind_dq_a = []
        # Initial state
        self.q_t0 = []
        # Excitation time series
        self.t   = t
        self.z   = z
        self.dz  = dz
        self.d2z = d2z

    def _build(self, solids, param):
        # Solid Sa
        s_a = solids[self.tag_S]
        self.ind_q_a = s_a.ind_q
        self.ind_dq_a = s_a.ind_dq
        # Initial state
        self.q_t0 = s_a.q_t0
        # Baumgarte parameters
        self.bmgt_2xiw0 = 2 * param.bmgt_xi * param.bmgt_w0
        self.bmgt_w02 = param.bmgt_w0**2

    def assemble(self, t, y, matrices):
        # Motion interpolation
        f   = np.array( [0, np.interp(t, self.t,   self.z), 0] )
        df  = np.array( [0, np.interp(t, self.t,  self.dz), 0] )
        d2f = np.array( [0, np.interp(t, self.t, self.d2z), 0] )
        # Parameters of Solid A
        q_a = y[self.ind_q_a]
        dq_a = y[self.ind_dq_a]
        # Acceleration terms
        Ca = - np.eye(3)
        # Velocity term d_AB
        d_P = dq_a - df
        # Position term AB
        P = q_a - (f + self.q_t0)
        # Matrix assembly
        matrices.C[np.ix_(self.ind_c, self.ind_q_a)] = Ca
        matrices.D[self.ind_c] = - d2f + self.bmgt_2xiw0 * d_P + self.bmgt_w02 * P
        return matrices

    def assemble_post(self, t, y, res):
        return res
