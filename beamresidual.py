import csdl
import python_csdl_backend
import numpy as np
from difvec import DifVec
from transform import CalcNodalT
from curvature import CalcNodalK
from beamdef import BeamDef



class BeamRes(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
        self.parameters.declare('joints',default={})
    def define(self):
        options = self.parameters['options']
        joints = self.parameters['joints']
        n = options['n']
        name = options['name']
        free = options['free']
        fixed = options['fixed']


        # process the joints dictionary
        child, r_j_list, theta_j_list = [], [], []
        for joint_name in joints:
            jx = self.declare_variable(joint_name+'x',shape=(12,1),val=0)
            r_ji, theta_ji = jx[0:3,0], jx[3:6,0]
            if joints[joint_name]['child_name'] == name: # if the beam has any child nodes
                child.append(joints[joint_name]['child_node']) # add the child node to the list
                r_j_list.append(r_ji)
                theta_j_list.append(theta_ji)





        # the 12 beam representation variables
        x = self.declare_variable(name+'x',shape=(12,n))
        r = x[0:3,:]
        theta = x[3:6,:]
        Fi = x[6:9,:]
        Mi = x[9:12,:]


        # aircraft state variables
        # R = self.declare_variable('R',shape=(3)) # aircraft position
        # U = self.declare_variable('U',shape=(3)) # aircraft velocity
        # OMEGA = self.declare_variable('OMEGA',shape=(3)) # aircraft rotation
        # THETA = self.declare_variable('THETA',shape=(3)) # aircraft orientation Euler angles

        self.add(BeamDef(options=options), name=name+'BeamDef')
        self.add(CalcNodalK(options=options), name=name+'CalcNodalK')
        self.add(CalcNodalT(options=options), name=name+'CalcNodalT')

        # other beam paramaters
        E_inv = self.declare_variable(name+'E_inv',shape=(3,3,n))
        K_0 = self.declare_variable(name+'K_0',shape=(3,3,n))
        D = self.declare_variable(name+'D',shape=(3,3,n))
        oneover = self.declare_variable(name+'oneover',shape=(3,3,n))
        fa = self.declare_variable(name+'fa',shape=(3,n-1),val=0) # distributed forces
        ma = self.declare_variable(name+'ma',shape=(3,n-1),val=0) # distributed moments
        theta_0 = self.declare_variable(name+'theta_0',shape=(3,n),val=0)
        r_0 = self.declare_variable(name+'r_0',shape=(3,n),val=0)
        T = self.declare_variable(name+'T',shape=(3,3,n)) # transformation tensor from xyz to csn
        Ta = self.declare_variable(name+'Ta',shape=(3,3,n-1)) # average transformation tensor from xyz to csn
        Ka = self.declare_variable(name+'Ka',shape=(3,3,n-1)) # curvature/angle-rate relation tensor

        # difference vectors
        self.add(DifVec(options=options), name=name+'Difvec')

        delta_s_0 = self.declare_variable(name+'delta_s_0',shape=(n-1),val=0)
        delta_theta_0 = self.declare_variable(name+'delta_theta_0',shape=(3,n-1),val=0)
        delta_r = self.declare_variable(name+'delta_r',shape=(3,n-1),val=0)
        delta_theta = self.declare_variable(name+'delta_theta',shape=(3,n-1),val=0)
        delta_s = self.declare_variable(name+'delta_s',shape=(n-1),val=0)
        delta_F = self.declare_variable(name+'delta_F',shape=(3,n-1),val=0)
        delta_M = self.declare_variable(name+'delta_M',shape=(3,n-1),val=0)
        Fa = self.declare_variable(name+'Fa',shape=(3,n-1),val=0)
        Ma = self.declare_variable(name+'Ma',shape=(3,n-1),val=0)
        delta_FP = self.declare_variable(name+'delta_FP',shape=(3,n-1),val=0)
        delta_MP = self.declare_variable(name+'delta_MP',shape=(3,n-1),val=0)


        # region strainscsn
        # compute strains in csn
        M_csn = self.create_output(name+'Mcsn',shape=(3,n),val=0)
        F_csn = self.create_output(name+'Fcsn',shape=(3,n),val=0)
        M_csnp = self.create_output(name+'M_csnp',shape=(3,n),val=0)
        strains_csn = self.create_output(name+'strains_csn',shape=(3,n),val=0)
        for i in range(n):
            # transform Fi and Mi in xyz to csn (ASW p6 eq14);
            T_i = csdl.reshape(T[:,:,i], new_shape=(3,3))
            M_i = csdl.reshape(Mi[:,i],new_shape=(3))
            F_i = csdl.reshape(Fi[:,i],new_shape=(3))
            M_csn[:,i] = csdl.expand(csdl.matvec(T_i, M_i),(3,1),'i->ij')
            F_csn[:,i] = csdl.expand(csdl.matvec(T_i, F_i),(3,1),'i->ij')

            # calculate M_csn_prime (ASW p8 eq18)
            D_i = csdl.reshape(D[:,:,i], new_shape=(3,3))
            mcsnp_t2 = csdl.matvec(csdl.transpose(D_i), csdl.reshape(F_csn[:,i], new_shape=(3)))
            M_csnp[:,i] = M_csn[:,i] + csdl.expand(mcsnp_t2, (3,1), 'i->ij')

            # compute strains in csn (ASW p8 eq19)
            oneover_i = csdl.reshape(oneover[:,:,i], new_shape=(3,3))
            E_inv_i = csdl.reshape(E_inv[:,:,i], new_shape=(3,3))
            M_csnp_i = csdl.reshape(M_csnp[:,i], new_shape=(3))
            F_csn_i = csdl.reshape(F_csn[:,i], new_shape=(3))

            term_1 = csdl.matvec(oneover_i, F_csn_i)
            term_2 = csdl.matvec(D_i, csdl.matvec(E_inv_i, M_csnp_i))

            strains_csn[:, i] = csdl.expand(term_1 + term_2, (3,1),'i->ij')
        # endregion


        # preconditioner values
        prec_sd = self.declare_variable(name+'prec_sdr',shape=(3),val=np.array([0.83036895, 0.81256237, 0.69745858]))
        prec_mc = self.declare_variable(name+'prec_mc',shape=(3),val=np.array([0.62307476, 0.71097352, 0.98816219]))
        prec_fe = self.declare_variable(name+'prec_fe',shape=(3),val=np.array([0.83968732, 1.34850155, 0.04046019]))
        prec_me = self.declare_variable(name+'prec_me',shape=(3),val=np.array([0.67601611, 0.33780926, 1.28021492]))


        # region straindisplacementresidual
        # compute the strain-displacement residual (ASW p12 eq48)
        strain_displacement_residual = self.create_output(name+'strain_displacement_residual', shape=(3,n-1), val=0)
        s_vec = self.declare_variable(name+'s_vec',shape=(3),val=np.array([0,1,0]))
        for i in range(n-1):
            strains_csn_a_i = csdl.reshape(0.5*(strains_csn[:,i+1] + strains_csn[:,i]), new_shape=(3))
            temp = s_vec + strains_csn_a_i
            delta_r_i = csdl.reshape(delta_r[:,i], new_shape=(3))
            Ta_i = csdl.reshape(Ta[:,:,i], new_shape=(3,3))
            delta_s_0_i = delta_s_0[i]

            strain_displacement_residual[:,i] = csdl.expand(prec_sd*(delta_r_i - csdl.matvec(csdl.transpose(Ta_i), temp*csdl.expand(delta_s_0_i,(3)))), (3,1), 'i->ij')
        # endregion


        # region momentcurvatureresidual
        # compute the moment-curvature relationship residual (ASW p13 eq54)
        moment_curvature_residual = self.create_output(name+'moment_curvature_residual', shape=(3,n-1), val=0)
        for i in range(n-1):
            Ka_i = csdl.reshape(Ka[:,:,i], new_shape=(3,3))
            K_0a_i = csdl.reshape(0.5*(K_0[:,:,i+1] + K_0[:,:,i]), new_shape=(3,3))
            delta_theta_i = csdl.reshape(delta_theta[:,i], new_shape=(3))
            delta_s_i = csdl.expand(delta_s[i], (3))
            delta_theta_0_i = csdl.reshape(delta_theta_0[:,i], new_shape=(3))
            Ea_inv_i = csdl.reshape(0.5*(E_inv[:,:,i+1] + E_inv[:,:,i]), new_shape=(3,3))
            Ta_i = csdl.reshape(Ta[:,:,i], new_shape=(3,3))
            Ma_i = csdl.reshape(Ma[:,i], new_shape=(3))
                
            moment_curvature_residual[:,i] = csdl.expand(prec_mc*(csdl.matvec(Ka_i,delta_theta_i) - csdl.matvec(K_0a_i,delta_theta_0_i) - csdl.matvec(Ea_inv_i, csdl.matvec(Ta_i, Ma_i*delta_s_i))), (3,1), 'i->ij')
        # endregion


        # region forceequilibriumresiduals
        # force equilibrium residual (ASW p13 eq56)
        force_equilibrium_residual = self.create_output(name+'force_equilibrium_residual', shape=(3,n-1), val=0)
        for i in range(n-1):
            if i not in fixed and i not in child: # if the node is not a child node and not a fixed node use: (ASW p13 eq56)
                delta_F_i = csdl.reshape(delta_F[:,i], new_shape=(3))
                fa_i = csdl.reshape(fa[:,i], new_shape=(3))
                delta_s_i = csdl.expand(delta_s[i], (3))
                delta_FP_i = csdl.reshape(delta_FP[:,i], new_shape=(3))

                force_equilibrium_residual[:,i] = csdl.expand(prec_fe*(delta_F_i + fa_i*delta_s_i + delta_FP_i), (3,1), 'i->ij')

            elif i in fixed: # if the node is a fixed node apply the fixed constraints instead
                r_i = r[:,i]
                r_0_i = r_0[:,i]
                force_equilibrium_residual[:,i] = r_i - r_0_i

            elif i in child: # if the node is a fixed node apply the fixed constraints instead
                r_j = next((r for c, r in zip(child, r_j_list) if c == i), None)
                r_i = r[:,i]
                r_0_i = r_0[:,i]
                force_equilibrium_residual[:,i] = r_i - r_0_i - r_j



        # endregion


        # region momentequilibriumresidual
        # moment equilibrium residual (ASW p13 eq55)
        moment_equilibrium_residual = self.create_output(name+'moment_equilibrium_residual', shape=(3,n-1), val=0)
        for i in range(n-1):
            if i not in fixed and i not in child:
                delta_M_i = csdl.reshape(delta_M[:,i], new_shape=(3))
                ma_i = csdl.reshape(ma[:,i], new_shape=(3))
                delta_s_i = csdl.expand(delta_s[i], (3))
                delta_MP_i = csdl.reshape(delta_MP[:,i], new_shape=(3))
                delta_r_i = csdl.reshape(delta_r[:,i], new_shape=(3))
                Fa_i = csdl.reshape(Fa[:,i], new_shape=(3))

                moment_equilibrium_residual[:,i] = csdl.expand(prec_me*(delta_M_i + ma_i*delta_s_i + delta_MP_i + csdl.cross(delta_r_i, Fa_i, axis=0)), (3,1), 'i->ij')

            elif i in fixed:
                theta_i = theta[:,i]
                theta_0_i = theta_0[:,i]
                moment_equilibrium_residual[:,i] = theta_i - theta_0_i

            elif i in child:
                theta_j = next((t for c, t in zip(child, theta_j_list) if c == i), None)
                theta_i = theta[:,i]
                theta_0_i = theta_0[:,i]
                moment_equilibrium_residual[:,i] = theta_i - theta_0_i - theta_j


        # endregion


        # region bcresidual
        # supports up to two free nodes
        free_force_residual = self.create_output(name+'free_force_residual', shape=(3,2), val=0)
        free_moment_residual = self.create_output(name+'free_moment_residual', shape=(3,2), val=0)
        for i, free_node in enumerate(free):
            free_force_residual[:,i] = Fi[:,int(free_node)] # nodal force at free node is zero
            free_moment_residual[:,i] = Mi[:,int(free_node)] # nodal moment at free node is zero
        
        # endregion



        # region concatenateresidual
        # concatenate residual (12x12 matrix)
        res = self.create_output(name+'res', shape=(12,n))
        res[0:3,0:n-1] = strain_displacement_residual
        res[3:6,0:n-1] = moment_curvature_residual
        res[6:9,0:n-1] = force_equilibrium_residual
        res[9:12,0:n-1] = moment_equilibrium_residual

        res[0:3,n-1] = free_force_residual[:,0]
        res[3:6,n-1] = free_moment_residual[:,0]
        res[6:9,n-1] = free_force_residual[:,1]
        res[9:12,n-1] = free_moment_residual[:,1]
        # endregion
        






