import csdl
import python_csdl_backend
import numpy as np


class JointRes(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joint')
    def define(self):
        beams = self.parameters['beams']
        joint = self.parameters['joint']

        # process the dictionaries
        joint_name = joint['name']
        parent_name, parent_node = joint['parent_name'], joint['parent_node']
        child_name, child_node = joint['child_name'], joint['child_node']
        num_parent_nodes, num_child_nodes = beams[parent_name]['n'], beams[child_name]['n']

        # create joint residual
        res = self.create_output(joint_name+'res', shape=(12,1), val=0)

        # partition the joint state
        jx = self.declare_variable(joint_name+'x',shape=(12,1),val=0)
        r_j = jx[0:3,0]
        theta_j = jx[0:3,0]
        F_j = jx[6:9,0]
        M_j = jx[9:12,0]

        # declare the variables
        parent_x = self.declare_variable(parent_name+'x',shape=(12,num_parent_nodes),val=0)
        child_x = self.declare_variable(child_name+'x',shape=(12,num_child_nodes),val=0)
        parent_r_0 = self.declare_variable(parent_name+'r_0',shape=(3,num_parent_nodes),val=0)
        child_r_0 = self.declare_variable(child_name+'r_0',shape=(3,num_child_nodes),val=0)
        child_theta_0 = self.declare_variable(child_name+'theta_0',shape=(3,num_child_nodes),val=0)
        T_parent = self.declare_variable(parent_name+'T',shape=(3,3,num_parent_nodes),val=0)
        T_child = self.declare_variable(child_name+'T',shape=(3,3,num_child_nodes),val=0)
        T_parent_0 = self.declare_variable(parent_name+'T_0',shape=(3,3,num_parent_nodes),val=0)
        T_child_0 = self.declare_variable(child_name+'T_0',shape=(3,3,num_child_nodes),val=0)

        child_delta_F = self.declare_variable(child_name+'delta_F',shape=(3,num_child_nodes-1),val=0)
        child_delta_M = self.declare_variable(child_name+'delta_M',shape=(3,num_child_nodes-1),val=0)
        


        # partition the parent/child states
        r1 = parent_x[0:3,parent_node]
        r2 = child_x[0:3,child_node]
        r1_0 = parent_r_0[:,parent_node]
        r2_0 = child_r_0[:,child_node]
        T1 = csdl.reshape(T_parent[:,:,parent_node], new_shape=(3,3))
        T2 = csdl.reshape(T_child[:,:,child_node], new_shape=(3,3))
        T1_0 = csdl.reshape(T_parent_0[:,:,parent_node], new_shape=(3,3))
        T2_0 = csdl.reshape(T_child_0[:,:,child_node], new_shape=(3,3))
        dF2 = child_delta_F[:,child_node]
        dM2 = child_delta_M[:,child_node]


        # compute the beam distance residual
        matmat = csdl.matmat(csdl.transpose(T1),T1_0)
        term0 = csdl.matvec(matmat, csdl.reshape(r2_0 - r1_0, new_shape=(3)))
        res[0:3,0] = r2 - r1 - csdl.expand(term0, (3,1), 'i->ij')

        # compute the beam orientation residual
        term1 = csdl.matmat(csdl.transpose(T1), T1_0)
        term2 = csdl.matmat(csdl.transpose(T2), T2_0)
        row3 = csdl.dot(term1[:, 1], term2[:, 2], axis=0) - csdl.dot(term1[:, 2], term2[:, 1], axis=0)
        row4 = csdl.dot(term1[:, 2], term2[:, 0], axis=0) - csdl.dot(term1[:, 0], term2[:, 2], axis=0)
        row5 = csdl.dot(term1[:, 0], term2[:, 1], axis=0) - csdl.dot(term1[:, 1], term2[:, 0], axis=0)

        #res[3,0] = csdl.expand(row3, (1,1), 'i->ij')
        #res[4,0] = csdl.expand(row4, (1,1), 'i->ij')
        #res[5,0] = csdl.expand(row5, (1,1), 'i->ij')

        child_theta = child_x[3:6,child_node]
        child_theta_0_i = child_theta_0[:,child_node]
        
        res[3:6,0] = child_theta - child_theta_0_i


        # compute the force balance residual
        force_residual = dF2 - F_j
        res[6:9,0] = force_residual


        # compute the moment balance residual
        moment_residual = dM2 - M_j + csdl.cross((r2 - r1), F_j, axis=0)
        res[9:12,0] = moment_residual
