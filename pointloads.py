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

        delta_FP = self.create_output(beam_name+'delta_FP',shape=(3,n-1),val=0)
        delta_MP = self.create_output(beam_name+'delta_MP',shape=(3,n-1),val=0)


        delta_F_pmass = self.declare_variable(beam_name+'delta_F_pmass',shape=(3,n-1),val=0)
        delta_M_pmass = self.declare_variable(beam_name+'delta_M_pmass',shape=(3,n-1),val=0)


        parent, parent_dict = [], {} # a list of any parent nodes in the current beam
        for joint_name in joints:
            if joints[joint_name]['parent_name'] == beam_name:
                parent_node = joints[joint_name]['parent_node']
                parent.append(parent_node)
                parent_dict[parent_node] = joint_name


        delta_F_joint = self.create_output(beam_name+'delta_F_joint',shape=(3,n-1),val=0)
        delta_M_joint = self.create_output(beam_name+'delta_M_joint',shape=(3,n-1),val=0)
        zero = self.declare_variable('zero',val=0)
        for i in range(n-1):
            if i in parent: # if the node is a parent node, assign delta_F_joint[:,i] and delta_M_joint[:,i] a value of Fj or Mj (ASW p27 eq143)
                joint_name = parent_dict[i]
                jx = self.declare_variable(joint_name+'x',shape=(12,1),val=0)
                Fj, Mj = jx[6:9,0], jx[9:12,0]

                delta_F_joint[:,i], delta_M_joint[:,i] = Fj, Mj
            else:
                delta_F_joint[:,i], delta_M_joint[:,i] = csdl.expand(zero,(3,1),'i->ij'), csdl.expand(zero,(3,1),'i->ij')




        delta_FP[:,:] = delta_F_pmass + delta_F_joint
        delta_MP[:,:] = delta_M_pmass + delta_M_joint





