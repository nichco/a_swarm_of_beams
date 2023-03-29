import csdl
import python_csdl_backend
import numpy as np


class PointLoads(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('joints')
    def define(self):
        options = self.parameters['options']
        joints = self.parameters['joints']
        beam_name = options['name']
        n = options['n']

        # create the point load/moment outputs
        delta_FP = self.create_output(beam_name+'delta_FP',shape=(3,n-1),val=0)
        delta_MP = self.create_output(beam_name+'delta_MP',shape=(3,n-1),val=0)

        # declare the point mass variables
        delta_F_pmass = self.declare_variable(beam_name+'delta_F_pmass',shape=(3,n-1),val=0)
        delta_M_pmass = self.declare_variable(beam_name+'delta_M_pmass',shape=(3,n-1),val=0)

        # create a parent list and a parent dictionary for any parent nodes in the current beam
        parent_dict = {joints[joint_name]['parent_node']: joint_name for joint_name in joints if joints[joint_name]['parent_name'] == beam_name}
        parent = list(parent_dict.keys())


        # compute the joint loads and moments
        delta_F_joint, delta_M_joint = [self.create_output(beam_name+f'delta_{var}_joint',shape=(3,n-1),val=0) for var in ['F', 'M']]
        zero = self.declare_variable('zero',val=0)
        for i in range(n-1):
            if i in parent:
                joint_name = parent_dict[i]
                jx = self.declare_variable(joint_name+'x',shape=(12,1),val=0)
                Fj, Mj = jx[6:9,0], jx[9:12,0]
                delta_F_joint[:,i], delta_M_joint[:,i] = Fj, Mj
            else:
                delta_F_joint[:,i], delta_M_joint[:,i] = [csdl.expand(zero,(3,1),'i->ij')] * 2


        # point loads/moments are the summation of point-mass loads/moments and joint loads/moments
        delta_FP[:,:] = delta_F_pmass + delta_F_joint
        delta_MP[:,:] = delta_M_pmass + delta_M_joint





