import csdl
import python_csdl_backend
import numpy as np

"""
the points must be ordered in the following format:
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
        G = options['G']

        num = 4
        

        E_mat = self.declare_variable(name+'E',shape=(3,3,n))
        EA = self.declare_variable(name+'EA',shape=(n))
        J = self.declare_variable(name+'J',shape=(n))
        Mcsn = self.declare_variable(name+'Mcsn',shape=(3,n))
        Fcsn = self.declare_variable(name+'Fcsn',shape=(3,n))

        h = self.declare_variable(name+'h',shape=(n))
        w = self.declare_variable(name+'w',shape=(n))

        Mc = csdl.reshape(Mcsn[0,:], new_shape=(n))
        Ms = csdl.reshape(Mcsn[1,:], new_shape=(n))
        Mn = csdl.reshape(Mcsn[2,:], new_shape=(n))
        Fc = csdl.reshape(Fcsn[0,:], new_shape=(n))
        Fs = csdl.reshape(Fcsn[1,:], new_shape=(n))
        Fn = csdl.reshape(Fcsn[2,:], new_shape=(n))

        # point coordinates
        x_coord = self.create_output(name+'x_coord',shape=(4,n),val=0)
        y_coord = self.create_output(name+'y_coord',shape=(4,n),val=0)
        # point 1
        x_coord[0,:] = csdl.reshape(-w/2, new_shape=(1,n))
        y_coord[0,:] = csdl.reshape(h/2, new_shape=(1,n))
        # point 2
        x_coord[1,:] = csdl.reshape(w/2, new_shape=(1,n))
        y_coord[1,:] = csdl.reshape(h/2, new_shape=(1,n))
        # point 3
        x_coord[2,:] = csdl.reshape(w/2, new_shape=(1,n))
        y_coord[2,:] = csdl.reshape(-h/2, new_shape=(1,n))
        # point 4
        x_coord[3,:] = csdl.reshape(-w/2, new_shape=(1,n))
        y_coord[3,:] = csdl.reshape(-h/2, new_shape=(1,n))

        
        # compute axial stress
        sigma_axial = self.create_output(name+'sigma_axial',shape=(4,n), val=0)
        for point in range(num): # for every point
            for i in range(n): # for every node
                x = csdl.reshape(x_coord[point,i], new_shape=(1))
                y = csdl.reshape(y_coord[point,i], new_shape=(1))

                EIcc = csdl.reshape(E_mat[0,0,i], new_shape=(1))
                EInn = csdl.reshape(E_mat[2,2,i], new_shape=(1))
                EA_i = EA[i]

                Fs_i = Fs[i]
                Mc_i = Mc[i]
                Mn_i = Mn[i]

                #sigma_axial[point,i] = E*(Fs[i, :] / EA_i - x3 * Mc[i, :] / (EIcc) + x1 * Mn[i, :] / (EInn))
                sigma = (E*Fs_i/EA_i) + (E*y*Mc_i/EIcc) + (E*x*Mn_i/EInn)
                sigma_axial[point,i] = csdl.expand(sigma, (1,1), 'i->ij')

        #self.print_var(sigma_axial[0,:])



        # compute torsional shear
        tau_torsion = self.create_output(name+'tau_torsion',shape=(4,n), val=0)
        for point in range(num):
            for i in range(n):
                x = csdl.reshape(x_coord[point,i], new_shape=(1))
                y = csdl.reshape(y_coord[point,i], new_shape=(1))
                r = (x**2 + y**2)**0.5

                Ms_i = Ms[i]
                J_i = J[i]

                tau = Ms_i*r/J_i

                tau_torsion[point,i] = csdl.expand(tau, (1,1), 'i->ij')

        #self.print_var(tau_torsion[0,:])


        # compute the transverse shear
        tau_shear = self.create_output(name+'tau_shear',shape=(4,n), val=0)
        for point in range(num):
            for i in range(n):
                Fn_i = Fn[i]
                Fc_i = Fc[i]

                F_shear = (Fn_i**2 + Fc_i**2)**0.5

                EA_i = EA[i]

                tau = E*F_shear/EA_i

                tau_shear[point,i] = csdl.expand(tau, (1,1), 'i->ij')


        FOS = 2 # safety factor

        # compute the von-mises stress
        sigma_vm = (sigma_axial**2 + tau_torsion**2 - sigma_axial*tau_torsion + 1e-19)**0.5
        self.register_output(name+'sigma_vm', sigma_vm)

        self.register_output(name+'max_sigma_vm', csdl.max(sigma_vm))
        self.register_output(name+'max_sigma_vm_fos', FOS*csdl.max(sigma_vm))

        #self.print_var(sigma_vm[3,:])
        #self.print_var(sigma_axial[3,:])
