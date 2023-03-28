import csdl
import python_csdl_backend
import numpy as np




class BeamDef(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']
        beam_type = options['beam_type']


        r_0 = self.declare_variable(name+'r_0',shape=(3,n),val=0)
        theta_0 = self.declare_variable(name+'theta_0',shape=(3,n),val=0)



        # compute delta_s_0
        delta_s_0 = self.create_output(name+'delta_s_0',shape=(n-1),val=0)
        delta_theta_0 = self.create_output(name+'delta_theta_0',shape=(3,n-1),val=0)
        for i in range(n-1):
            delta_r_i = r_0[:, i + 1] - r_0[:, i] + 1E-19
            delta_s_0[i] = csdl.reshape((delta_r_i[0,0]**2 + delta_r_i[1,0]**2 + delta_r_i[2,0]**2)**0.5, new_shape=(1))
            delta_theta_0[:,i] = theta_0[:,i+1] - theta_0[:,i]

        

        # compute the initial curvature matrix K_0
        one = self.declare_variable('one',val=1)
        K_0 = self.create_output(name+'K_0',shape=(3,3,n),val=0)
        for i in range(n):
            if beam_type == 'wing':
                K_0[0,0,i] = csdl.reshape(csdl.cos(theta_0[2,i])*csdl.cos(theta_0[1,i]), new_shape=(1,1,1))
                K_0[0,2,i] = csdl.reshape(-csdl.sin(theta_0[1,i]), new_shape=(1,1,1))
                K_0[1,0,i] = csdl.reshape(-csdl.sin(theta_0[2,i]), new_shape=(1,1,1))
                K_0[1,1,i] = csdl.reshape(one, new_shape=(1,1,1))
                K_0[2,0,i] = csdl.reshape(csdl.cos(theta_0[2,1])*csdl.sin(theta_0[1,i]), new_shape=(1,1,1))
                K_0[2,2,i] = csdl.reshape(csdl.cos(theta_0[1,i]), new_shape=(1,1,1))
            if beam_type == 'fuse':
                K_0[0,0,i] = csdl.reshape(csdl.cos(theta_0[1,i]),new_shape=(1,1,1))
                K_0[0,2,i] = csdl.reshape(-csdl.cos(theta_0[0,i])*csdl.sin(theta_0[1,i]),new_shape=(1,1,1))
                K_0[1,1,i] = csdl.reshape(one,new_shape=(1,1,1))
                K_0[1,2,i] = csdl.reshape(csdl.sin(theta_0[0,i]),new_shape=(1,1,1))
                K_0[2,0,i] = csdl.reshape(csdl.sin(theta_0[1,i]),new_shape=(1,1,1))
                K_0[2,2,i] = csdl.reshape(csdl.cos(theta_0[0,i])*csdl.cos(theta_0[1,i]),new_shape=(1,1,1))
                


        # compute the initial transformation matrix T_0
        R_phi = self.create_output(name+'R_phi_0', shape=(n,3,3),val=0)
        R_theta = self.create_output(name+'R_theta_0', shape=(n,3,3),val=0)
        R_psi = self.create_output(name+'R_psi_0', shape=(n,3,3),val=0)
        T_0 = self.create_output(name+'T_0',shape=(3,3,n),val=0)
        for i in range(0, n):
            a1 = theta_0[0, i]
            a2 = theta_0[1, i]
            a3 = theta_0[2, i]

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
                T_0[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_psi[i,:,:], new_shape=(3,3)), csdl.reshape(R_phi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')
            elif beam_type == 'fuse':
                T_0[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_phi[i,:,:], new_shape=(3,3)), csdl.reshape(R_psi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')