import csdl
import python_csdl_backend
import numpy as np


class CalcNodalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']
        beam_type = options['beam_type']

        x = self.declare_variable(name+'x',shape=(12,n))
        theta = x[3:6,:]

        one = self.declare_variable('one',val=1)

        K = self.create_output(name+'K',shape=(3,3,n),val=0)
        Ka = self.create_output(name+'Ka',shape=(3,3,n-1),val=0)

        for i in range(0, n):
            if beam_type == 'wing':
                K[0,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,0,i] = csdl.expand(-csdl.sin(theta[2,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[2,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')

            if beam_type == 'fuse':

                K[0,0,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.cos(theta[0,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[1,2,i] = csdl.expand(csdl.sin(theta[0,i]),(1,1,1),'ij->ijk')
                K[2,0,i] = csdl.expand(csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[0,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')


        # compute Ka from K
        for i in range(n-1):
            Ka[:,:,i] = 0.5*(K[:,:,i+1] + K[:,:,i])