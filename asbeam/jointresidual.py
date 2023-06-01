import csdl
import python_csdl_backend
import numpy as np


class JointRes(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('beams')
        self.parameters.declare('joint')

    def define(self):
        joint_name = self.parameters['name']
        beams = self.parameters['beams']
        joint = self.parameters['joint']

        # process the dictionaries
        parent_name, parent_node = joint['parent_name'], joint['parent_node']
        child_name, child_node = joint['child_name'], joint['child_node']
        num_parent_nodes, num_child_nodes = len(beams[parent_name]['nodes']), len(beams[child_name]['nodes'])

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

        child_delta_FP = self.declare_variable(child_name+'delta_FP',shape=(3,num_child_nodes-1),val=0)
        child_delta_MP = self.declare_variable(child_name+'delta_MP',shape=(3,num_child_nodes-1),val=0)
        


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
        delta_FP_2 = child_delta_FP[:,child_node]
        delta_MP_2 = child_delta_MP[:,child_node]


        # compute the beam distance residual
        matmat = csdl.matmat(csdl.transpose(T1),T1_0)
        term0 = csdl.matvec(matmat, csdl.reshape(r2_0 - r1_0, new_shape=(3)))
        res[0:3,0] = r2 - r1 - csdl.expand(term0, (3,1), 'i->ij')

        # compute the beam orientation residual
        child_theta = child_x[3:6,child_node]
        child_theta_0_i = child_theta_0[:,child_node]
        
        res[3:6,0] = child_theta - child_theta_0_i


        # compute the force balance residual
        force_residual = dF2 + delta_FP_2 - F_j
        res[6:9,0] = force_residual


        # compute the moment balance residual
        moment_residual = dM2 + delta_MP_2 - M_j + csdl.cross((r2 - r1), F_j, axis=0)
        res[9:12,0] = moment_residual








        A = csdl.transpose(csdl.matmat(csdl.transpose(T1), T1_0))
        B = csdl.transpose(csdl.matmat(csdl.transpose(T2), T2_0))


        a1 = A[0,:]
        b1 = A[1,:]
        c1 = A[2,:]

        a2 = B[0,:]
        b2 = B[1,:]
        c2 = B[2,:]

        row1 = csdl.dot(b1,c2,axis=1) - csdl.dot(c1,b2,axis=1)
        row2 = csdl.dot(c1,a2,axis=1) - csdl.dot(a1,c2,axis=1)
        row3 = csdl.dot(a1,b2,axis=1) - csdl.dot(b1,a2,axis=1)

        # res[3,0] = csdl.expand(row1, (1,1), 'i->ij')
        # res[4,0] = csdl.expand(row2, (1,1), 'i->ij')
        # res[5,0] = csdl.expand(row3, (1,1), 'i->ij')
        
