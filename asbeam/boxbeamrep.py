import csdl
import numpy as np
import python_csdl_backend



class BoxBeamRep(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']
        E = options['E']
        G = options['G']
        rho = options['rho']

        
        # cs params
        h = self.declare_variable(name+'h',shape=(n)) # beam height
        w = self.declare_variable(name+'w',shape=(n)) # beam width
        t_left = self.declare_variable(name+'t_left',shape=(n),val=0.1)
        t_top = self.declare_variable(name+'t_top',shape=(n),val=0.1)
        t_right = self.declare_variable(name+'t_right',shape=(n),val=0.1)
        t_bot = self.declare_variable(name+'t_bot',shape=(n),val=0.1)

        zero = self.declare_variable('zero',val=0)


        # region Box-beam cross-section 4 parts
        # segments 1:
        A_sect1 = t_top * w
        rho_sect1 = rho
        E_sect1 = E
        cg_sect1_y = 0
        cg_sect1_z = (h - t_top) / 2
        Ixx_sect1 = (w * t_top ** 3) / 12
        Izz_sect1 = (t_top * w ** 3) / 12
        # segments 2:
        A_sect2 = t_left * (h - t_top - t_bot)
        rho_sect2 = rho
        E_sect2 = E
        cg_sect2_y = (t_left - w) / 2
        cg_sect2_z = h / 2 - t_top - (h - t_top - t_bot) / 2
        Ixx_sect2 = (t_left * (h - t_top - t_bot) ** 3) / 12
        Izz_sect2 = ((h - t_top - t_bot) * t_left ** 3) / 12
        # segments 3:
        A_sect3 = t_bot * w
        rho_sect3 = rho
        E_sect3 = E
        cg_sect3_y = 0
        cg_sect3_z = (t_bot - h) / 2
        Ixx_sect3 = (w * t_bot ** 3) / 12
        Izz_sect3 = (t_bot * w ** 3) / 12
        # segments 4:
        A_sect4 = t_right * (h - t_top - t_bot)
        E_sect4 = E
        rho_sect4 = rho
        cg_sect4_y = (w - t_right) / 2
        cg_sect4_z = h / 2 - t_top - (h - t_top - t_bot) / 2
        Ixx_sect4 = (t_right * (h - t_top - t_bot) ** 3) / 12
        Izz_sect4 = ((h - t_top - t_bot) * t_right ** 3) / 12
        # endregion


        # region offsets from the beam axis
        e_cg_x = (cg_sect1_y * A_sect1 * E_sect1 +
                    cg_sect2_y * A_sect2 * E_sect2 +
                    cg_sect3_y * A_sect3 * E_sect3 +
                    cg_sect4_y * A_sect4 * E_sect4) / \
                    (A_sect1 * E_sect1 +
                    A_sect2 * E_sect2 +
                    A_sect3 * E_sect3 +
                    A_sect4 * E_sect4)
        e_cg_z = (cg_sect1_z * A_sect1 * E_sect1 +
                    cg_sect2_z * A_sect2 * E_sect2 +
                    cg_sect3_z * A_sect3 * E_sect3 +
                    cg_sect4_z * A_sect4 * E_sect4) / \
                    (A_sect1 * E_sect1 +
                    A_sect2 * E_sect2 +
                    A_sect3 * E_sect3 +
                    A_sect4 * E_sect4)
        n_ea = e_cg_z
        c_ea = e_cg_x
        n_ta = csdl.expand(zero, (n))
        c_ta = csdl.expand(zero, (n))
        # endregion


        # region bending and torsional stiffness
        # parallel axis theorem
        Ixx = Ixx_sect1 + A_sect1 * (cg_sect1_z - e_cg_z) ** 2 + \
                Ixx_sect2 + A_sect2 * (cg_sect2_z - e_cg_z) ** 2 + \
                Ixx_sect3 + A_sect3 * (cg_sect3_z - e_cg_z) ** 2 + \
                Ixx_sect4 + A_sect4 * (cg_sect4_z - e_cg_z) ** 2
        Izz = Izz_sect1 + A_sect1 * (cg_sect1_y - e_cg_x) ** 2 + \
                Izz_sect2 + A_sect2 * (cg_sect2_y - e_cg_x) ** 2 + \
                Izz_sect3 + A_sect3 * (cg_sect3_y - e_cg_x) ** 2 + \
                Izz_sect4 + A_sect4 * (cg_sect4_y - e_cg_x) ** 2
        Ixz = -(A_sect1 * (cg_sect1_y - e_cg_x) * (cg_sect1_z - e_cg_z) +
                A_sect2 * (cg_sect2_y - e_cg_x) * (cg_sect2_z - e_cg_z) +
                A_sect3 * (cg_sect3_y - e_cg_x) * (cg_sect3_z - e_cg_z) +
                A_sect4 * (cg_sect4_y - e_cg_x) * (cg_sect4_z - e_cg_z))
        J = 2 * (((h - t_top / 2 - t_bot / 2) * (w - t_right / 2 - t_left / 2)) ** 2) / \
                (((w - t_right / 2 - t_left / 2) / (0.5 * t_top + 0.5 * t_bot)) +
                ((h - t_top / 2 - t_bot / 2) / (0.5 * t_right + 0.5 * t_left)))
        EIxx = E * Ixx
        EIzz = E * Izz
        EIxz = E * Ixz
        GJ = G * J
        # endregion


        # region axial and shear stiffness
        GKn = self.create_input(name+'GKn',shape=(n),val=G / 1.2 * np.ones(n))
        GKc = self.create_input(name+'GKc',shape=(n),val=G / 1.2 * np.ones(n))
        EA = E_sect1 * A_sect1 + E_sect2 * A_sect2 + E_sect3 * A_sect3 + E_sect4 * A_sect4
        # endregion

        # region mass properties
        A = A_sect1 + A_sect2 + A_sect3 + A_sect4
        mu = self.create_output(name+'mu',shape=(n-1),val=0)
        for i in range(n - 1):
            A1 = A[i]
            A2 = A[i + 1]
            mu[i] = rho * 1 / 3 * (A1 + A2 + ((A1 * A2)**0.5))
        # endregion

        # region E
        E_inv = self.create_output(name+'E_inv',shape=(3,3,n),val=0)
        for i in range(n):
            denom_i = (EIxx[i]*EIzz[i] - EIxz[i]**2)
            E_inv[0,0,i] = csdl.expand(EIzz[i]/denom_i, (1,1,1))
            E_inv[0,2,i] = csdl.expand(-EIxz[i]/denom_i, (1,1,1))
            E_inv[1,1,i] = csdl.expand(1/(GJ[i]), (1,1,1))
            E_inv[2,2,i] = csdl.expand(EIxx[i]/denom_i, (1,1,1))
            E_inv[2,0,i] = csdl.expand(-EIxz[i]/denom_i, (1,1,1))

        # endregion

        
        # region oneover
        oneover = self.create_output(name+'oneover',shape=(3,3,n),val=0)
        for i in range(n):
            oneover[0,0,i] = csdl.expand(1 / GKc[i], (1,1,1))
            oneover[1,1,i] = csdl.expand(1 / EA[i], (1,1,1))
            oneover[2,2,i] = csdl.expand(1 / GKn[i], (1,1,1))
        # endregion

        # region D
        D = np.zeros((3,3,n))
        D = self.create_output(name+'D',shape=(3,3,n),val=0)
        for i in range(n):
            D[0,1,i] = csdl.expand(-n_ea[i], (1,1,1))
            D[1,0,i] = csdl.expand(n_ta[0], (1,1,1))
            D[1,2,i] = csdl.expand(-c_ta[0], (1,1,1))
            D[2,1,i] = csdl.expand(c_ea[0], (1,1,1))

        # endregion
        





if __name__ == '__main__':

    options = {}
    options['n'] = 2
    options['name'] = 'wing'
    options['free'] = np.array([options['n']-1]) # (tip)
    options['fixed'] = fixed = np.array([0]) # (root)
    options['E'] = 69E9
    options['G'] = 1E20
    options['rho'] = 2700

    sim = python_csdl_backend.Simulator(BoxBeamRep(options=options))
    sim.run()

    print(sim[options['name']+'E_inv'])