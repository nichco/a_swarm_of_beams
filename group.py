import csdl
import python_csdl_backend
import numpy as np
from beamresidual import BeamRes


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
        i = 0
        for beam_name in beams:
            self.register_output(beam_name+'x', x[:, i:i+(n:=beams[beam_name]['n'])])
            i += n

        # partition the joint state
        for i, joint_name in enumerate(joints): self.register_output(joint_name+'x', x[:,num_nodes+i])

        # partition the beam properties inputs
        vars = {'r_0': (3,num_nodes),
                'theta_0': (3,num_nodes),
                'E_inv': (3,3,num_nodes),
                'D': (3,3,num_nodes),
                'oneover': (3,3,num_nodes),
                'fa': (3,num_nodes)}
        
        r_0, theta_0, E_inv, D, oneover, fa = [self.declare_variable(var_name, shape=var_shape) for var_name, var_shape in vars.items()]

        i = 0
        for beam_name in beams:
            n = beams[beam_name]['n']
            self.register_output(beam_name+'r_0', r_0[:,i:i+n])
            self.register_output(beam_name+'theta_0', theta_0[:,i:i+n])
            self.register_output(beam_name+'E_inv', E_inv[:,:,i:i+n])
            self.register_output(beam_name+'D', D[:,:,i:i+n])
            self.register_output(beam_name+'oneover', oneover[:,:,i:i+n])
            self.register_output(beam_name+'fa', fa[:,i:i+n-1])
            i += n


        # get the beam residuals
        i = 0
        for beam_name in beams:
            self.add(BeamRes(options=beams[beam_name], joints=joints), name=beam_name+'BeamRes')
            res[:, i:i+n] = self.declare_variable(beam_name+'res', shape=(12,n), val=0) + 0*x[:,i:i+n]
            i += n


        # get the joint residuals
        for joint_name in joints:
            pass


        



        