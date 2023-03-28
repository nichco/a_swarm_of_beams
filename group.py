import csdl
import python_csdl_backend
import numpy as np


class Group(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        num_nodes = sum(beams[beam_name]['n'] for beam_name in beams)
        cols = num_nodes+len(joints)


        # declare the state
        x = self.declare_variable('x',shape=(12,cols),val=0)

        # create the residual
        res = self.create_output('res',shape=(12,cols),val=0)

        # partition the state
        [(self.register_output(beam_name+'x', x[:, i:i + beams[beam_name]['n']]), i := i+beams[beam_name]['n']) for beam_name in beams]

        # partition the joint state
        for i, joint_name in enumerate(joints): self.register_output(joint_name+'x', x[:,num_nodes+i])

        # partition the beam properties inputs
        vars = {'r_0': (3,num_nodes),
                'theta_0': (3,num_nodes),
                'E_inv': (3,3,num_nodes),
                'D': (3,3,num_nodes),
                'oneover': (3,3,num_nodes)}
        
        r_0, theta_0, E_inv, D, oneover = [self.declare_variable(var_name, shape=var_shape) for var_name, var_shape in vars.items()]

        i = 0
        for beam_name in beams:
            n = beams[beam_name]['n']
            for var_name, var in vars.items():
                self.register_output(beam_name + var_name, eval(var_name)[:, i:i+n])
            i += n


        # get the beam residuals
        for beam_name in beams:
            

        



        