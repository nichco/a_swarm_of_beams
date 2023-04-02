import csdl
import python_csdl_backend
import numpy as np

"""
the points must be ordered in the following format
    1 ------------------------------------- 2
      -                y                  -
      -                |                  -
      -                --> x              -
      -                                   -
      -                                   -
    4 ------------------------------------- 3
"""


class Stress(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        name = options['name']
        n = options['n']
        E = options['E']

        num = 4

        E_mat = self.declare_variable(name+'E',shape=(3,3,n),val=0)
        EA = self.declare_variable(name+'EA',shape=(n),val=0)
        Mcsn = self.declare_variable(name+'Mcsn',shape=(3,n),val=0)
        Fcsn = self.declare_variable(name+'Fcsn',shape=(3,n),val=0)

        Mc = Mcsn[0,:]
        Ms = Mcsn[1,:]
        Mn = Mcsn[2,:]
        Fc = Fcsn[0,:]
        Fs = Fcsn[1,:]
        Fn = Fcsn[2,:]


        #symb_stress_points = SX.sym('stress_pts', stress_rec_points.shape[1], stress_rec_points.shape[0]).T










        # region Axial Stress
        # sigma_axial at a time step is
        #   0:n is 1st point (top-left)
        #   n+1:2n is 2nd point (top-right)
        #   2n+1:3n is 3rd point (bottom-right)
        #   3n+1:4n is 4th point (bottom-left)

        #sigma_axial = SX.sym('sigma_a', number_of_stress_points * n, T)
        sigma_axial = self.create_output(name+'sigma_axial',shape=(num*n), val=0)
        for j in range(num):
            for i in range(n):
                x1 = symb_stress_points[2 * j, i]
                x3 = symb_stress_points[2 * j + 1, i]
                EIcc = E_mat[0,0,i]
                EInn = E_mat[2,2,i]
                EA_i = EA[i]
                sigma_axial[j * n + i, :] = E * (Fs[i, :] / EA_i - x3 * Mc[i, :] / (EIcc) + x1 * Mn[i, :] / (EInn))
                pass


        # endregion