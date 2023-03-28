import csdl
import numpy as np
import python_csdl_backend





class DifVec(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']



        x = self.declare_variable(name+'x',shape=(12,n))
        r = x[0:3,:]
        theta = x[3:6,:]
        F = x[6:9,:]
        M = x[9:12,:]



        delta_r = self.create_output(name+'delta_r',shape=(3,n-1))
        delta_theta = self.create_output(name+'delta_theta',shape=(3,n-1))
        delta_s = self.create_output(name+'delta_s',shape=(n-1))
        delta_F = self.create_output(name+'delta_F',shape=(3,n-1))
        delta_M = self.create_output(name+'delta_M',shape=(3,n-1))
        Fa = self.create_output(name+'Fa',shape=(3,n-1))
        Ma = self.create_output(name+'Ma',shape=(3,n-1))


        for i in range(0,n-1):
            delta_r[:, i] = r[:, i + 1] - r[:, i] + 1E-19
            delta_s[i] = csdl.reshape(((delta_r[0, i])**2 + (delta_r[1, i])**2 + (delta_r[2, i])**2)**0.5, new_shape=(1))

            delta_theta[:,i] = theta[:,i+1] - theta[:,i]

            delta_F[:,i] = F[:,i+1] - F[:,i]
            delta_M[:,i] = M[:,i+1] - M[:,i]

            Fa[:,i] = 0.5*(F[:,i+1] + F[:,i])
            Ma[:,i] = 0.5*(M[:,i+1] + M[:,i])