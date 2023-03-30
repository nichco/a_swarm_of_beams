import csdl
import python_csdl_backend
import numpy as np


class CalcNodalT(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']
        beam_type = options['beam_type']

        x = self.declare_variable(name+'x',shape=(12,n),val=0)
        theta = x[3:6,:]

        one = self.declare_variable('one',val=1)

        R_phi = self.create_output(name+'R_phi', shape=(n,3,3),val=0)
        R_theta = self.create_output(name+'R_theta', shape=(n,3,3),val=0)
        R_psi = self.create_output(name+'R_psi', shape=(n,3,3),val=0)

        T = self.create_output(name+'T',shape=(3,3,n),val=0)
        Ta = self.create_output(name+'Ta',shape=(3,3,n-1),val=0)

        for i in range(0, n):
            a1, a2, a3 = theta[0, i], theta[1, i], theta[2, i]

            # rotation tensor for phi (angle a1)
            R_phi[i,0,0] = csdl.expand(one,(1,1,1))
            R_phi[i,1,1] = csdl.expand(csdl.cos(a1),(1,1,1),'ij->ijk')
            R_phi[i,1,2] = csdl.expand(csdl.sin(a1),(1,1,1),'ij->ijk')
            R_phi[i,2,1] = csdl.expand(-csdl.sin(a1),(1,1,1),'ij->ijk')
            R_phi[i,2,2] = csdl.expand(csdl.cos(a1),(1,1,1),'ij->ijk')

            # rotation tensor for theta (angle a2)
            R_theta[i,0,0] = csdl.expand(csdl.cos(a2),(1,1,1),'ij->ijk')
            R_theta[i,0,2] = csdl.expand(-csdl.sin(a2),(1,1,1),'ij->ijk')
            R_theta[i,1,1] = csdl.expand(one,(1,1,1))
            R_theta[i,2,0] = csdl.expand(csdl.sin(a2),(1,1,1),'ij->ijk')
            R_theta[i,2,2] = csdl.expand(csdl.cos(a2),(1,1,1),'ij->ijk')

            # rotation tensor for psi (angle a3)
            R_psi[i,0,0] = csdl.expand(csdl.cos(a3),(1,1,1),'ij->ijk')
            R_psi[i,0,1] = csdl.expand(csdl.sin(a3),(1,1,1),'ij->ijk')
            R_psi[i,1,0] = csdl.expand(-csdl.sin(a3),(1,1,1),'ij->ijk')
            R_psi[i,1,1] = csdl.expand(csdl.cos(a3),(1,1,1),'ij->ijk')
            R_psi[i,2,2] = csdl.expand(one,(1,1,1))

            if beam_type == 'wing':
                T[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_psi[i,:,:], new_shape=(3,3)), csdl.reshape(R_phi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')
            elif beam_type == 'fuse':
                T[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_phi[i,:,:], new_shape=(3,3)), csdl.reshape(R_psi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')



        # compute Ta from T
        for i in range(n-1):
            Ta[:,:,i] = 0.5*(T[:,:,i+1] + T[:,:,i])


