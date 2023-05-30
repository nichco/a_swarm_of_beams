import csdl
import python_csdl_backend
import numpy as np



class BeamRes(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('options')
        self.parameters.declare('joints',default={})
    def define(self):
        name = self.parameters['name']
        options = self.parameters['options']
        joints = self.parameters['joints']

        n = len(options['nodes'])
        free = options['free']
        fixed = options['fixed']
        beam_type = options['type']


        # process the joints dictionary
        child, r_j_list, theta_j_list = [], [], []
        for joint_name in joints:
            jx = self.declare_variable(joint_name + 'x', shape=(12,1), val=0)
            r_ji, theta_ji = jx[0:3,0], jx[3:6,0]
            if joints[joint_name]['child_name'] == name: # if the beam has any child nodes
                child.append(joints[joint_name]['child_node']) # add the child node to the list
                r_j_list.append(r_ji)
                theta_j_list.append(theta_ji)



        parent_dict = {joints[joint_name]['parent_node']: joint_name for joint_name in joints if joints[joint_name]['parent_name'] == name}
        parent = list(parent_dict.keys())



        # the 12 beam representation variables
        x = self.declare_variable(name + 'x', shape=(12,n))
        r = x[0:3,:]
        #self.print_var(r)
        theta = x[3:6,:]
        F = x[6:9,:]
        M = x[9:12,:]


        # aircraft state variables
        # R = self.declare_variable('R',shape=(3)) # aircraft position
        # U = self.declare_variable('U',shape=(3)) # aircraft velocity
        # OMEGA = self.declare_variable('OMEGA',shape=(3)) # aircraft rotation
        # THETA = self.declare_variable('THETA',shape=(3)) # aircraft orientation Euler angles

        # other beam paramaters
        E_inv = self.declare_variable(name+'E_inv',shape=(3,3,n))
        D = self.declare_variable(name+'D',shape=(3,3,n))
        oneover = self.declare_variable(name+'oneover',shape=(3,3,n))
        f = self.declare_variable(name+'f',shape=(3,n),val=0) # distributed forces
        m = self.declare_variable(name+'m',shape=(3,n),val=0) # distributed moments
        theta_0 = self.declare_variable(name+'theta_0',shape=(3,n),val=0)
        r_0 = self.declare_variable(name+'r_0',shape=(3,n),val=0)
        fp = self.declare_variable(name+'fp',shape=(3,n),val=0)
        mp = self.declare_variable(name+'mp',shape=(3,n),val=0)
        zero = self.declare_variable(name+'zero',val=0)
        one = self.declare_variable(name+'one',val=1)
        mu = self.declare_variable(name+'mu',shape=(n-1),val=0)

        

        # compute T & T_0: the transformation tensor from xyz to csn
        R_phi = self.create_output(name+'R_phi', shape=(n,3,3),val=0)
        R_theta = self.create_output(name+'R_theta', shape=(n,3,3),val=0)
        R_psi = self.create_output(name+'R_psi', shape=(n,3,3),val=0)
        R_phi_0 = self.create_output(name+'R_phi_0', shape=(n,3,3),val=0)
        R_theta_0 = self.create_output(name+'R_theta_0', shape=(n,3,3),val=0)
        R_psi_0 = self.create_output(name+'R_psi_0', shape=(n,3,3),val=0)
        T = self.create_output(name+'T',shape=(3,3,n),val=0)
        T_0 = self.create_output(name+'T_0',shape=(3,3,n),val=0)
        for i in range(0, n):
            a1, a2, a3 = theta[0, i], theta[1, i], theta[2, i]
            a1_0, a2_0, a3_0 = theta_0[0, i], theta_0[1, i], theta_0[2, i]
            # rotation tensor for phi (angle a1)
            R_phi[i,0,0] = csdl.expand(one,(1,1,1))
            R_phi[i,1,1] = csdl.expand(csdl.cos(a1),(1,1,1),'ij->ijk')
            R_phi[i,1,2] = csdl.expand(csdl.sin(a1),(1,1,1),'ij->ijk')
            R_phi[i,2,1] = csdl.expand(-csdl.sin(a1),(1,1,1),'ij->ijk')
            R_phi[i,2,2] = csdl.expand(csdl.cos(a1),(1,1,1),'ij->ijk')

            R_phi_0[i,0,0] = csdl.expand(one,(1,1,1))
            R_phi_0[i,1,1] = csdl.expand(csdl.cos(a1_0),(1,1,1),'ij->ijk')
            R_phi_0[i,1,2] = csdl.expand(csdl.sin(a1_0),(1,1,1),'ij->ijk')
            R_phi_0[i,2,1] = csdl.expand(-csdl.sin(a1_0),(1,1,1),'ij->ijk')
            R_phi_0[i,2,2] = csdl.expand(csdl.cos(a1_0),(1,1,1),'ij->ijk')

            # rotation tensor for theta (angle a2)
            R_theta[i,0,0] = csdl.expand(csdl.cos(a2),(1,1,1),'ij->ijk')
            R_theta[i,0,2] = csdl.expand(-csdl.sin(a2),(1,1,1),'ij->ijk')
            R_theta[i,1,1] = csdl.expand(one,(1,1,1))
            R_theta[i,2,0] = csdl.expand(csdl.sin(a2),(1,1,1),'ij->ijk')
            R_theta[i,2,2] = csdl.expand(csdl.cos(a2),(1,1,1),'ij->ijk')

            R_theta_0[i,0,0] = csdl.expand(csdl.cos(a2_0),(1,1,1),'ij->ijk')
            R_theta_0[i,0,2] = csdl.expand(-csdl.sin(a2_0),(1,1,1),'ij->ijk')
            R_theta_0[i,1,1] = csdl.expand(one,(1,1,1))
            R_theta_0[i,2,0] = csdl.expand(csdl.sin(a2_0),(1,1,1),'ij->ijk')
            R_theta_0[i,2,2] = csdl.expand(csdl.cos(a2_0),(1,1,1),'ij->ijk')

            # rotation tensor for psi (angle a3)
            R_psi[i,0,0] = csdl.expand(csdl.cos(a3),(1,1,1),'ij->ijk')
            R_psi[i,0,1] = csdl.expand(csdl.sin(a3),(1,1,1),'ij->ijk')
            R_psi[i,1,0] = csdl.expand(-csdl.sin(a3),(1,1,1),'ij->ijk')
            R_psi[i,1,1] = csdl.expand(csdl.cos(a3),(1,1,1),'ij->ijk')
            R_psi[i,2,2] = csdl.expand(one,(1,1,1))

            R_psi_0[i,0,0] = csdl.expand(csdl.cos(a3_0),(1,1,1),'ij->ijk')
            R_psi_0[i,0,1] = csdl.expand(csdl.sin(a3_0),(1,1,1),'ij->ijk')
            R_psi_0[i,1,0] = csdl.expand(-csdl.sin(a3_0),(1,1,1),'ij->ijk')
            R_psi_0[i,1,1] = csdl.expand(csdl.cos(a3_0),(1,1,1),'ij->ijk')
            R_psi_0[i,2,2] = csdl.expand(one,(1,1,1))

            if beam_type == 'wing':
                T[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_psi[i,:,:], new_shape=(3,3)), csdl.reshape(R_phi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')
                T_0[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta_0[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_psi_0[i,:,:], new_shape=(3,3)), csdl.reshape(R_phi_0[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')
            elif beam_type == 'fuse':
                T[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_phi[i,:,:], new_shape=(3,3)), csdl.reshape(R_psi[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')
                T_0[:,:,i] = csdl.expand(csdl.matmat(csdl.reshape(R_theta_0[i,:,:], new_shape=(3,3)), csdl.matmat(csdl.reshape(R_phi_0[i,:,:], new_shape=(3,3)), csdl.reshape(R_psi_0[i,:,:], new_shape=(3,3)))), (3,3,1),'ij->ijk')






        # compute K & K_0: the curvature/angle-rate relation tensor
        K = self.create_output(name+'K',shape=(3,3,n),val=0)
        K_0 = self.create_output(name+'K_0',shape=(3,3,n),val=0)
        for i in range(0, n):
            if beam_type == 'wing':
                K[0,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,0,i] = csdl.expand(-csdl.sin(theta[2,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[2,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')

                K_0[0,0,i] = csdl.reshape(csdl.cos(theta_0[2,i])*csdl.cos(theta_0[1,i]), new_shape=(1,1,1))
                K_0[0,2,i] = csdl.reshape(-csdl.sin(theta_0[1,i]), new_shape=(1,1,1))
                K_0[1,0,i] = csdl.reshape(-csdl.sin(theta_0[2,i]), new_shape=(1,1,1))
                K_0[1,1,i] = csdl.reshape(one, new_shape=(1,1,1))
                K_0[2,0,i] = csdl.reshape(csdl.cos(theta_0[2,1])*csdl.sin(theta_0[1,i]), new_shape=(1,1,1))
                K_0[2,2,i] = csdl.reshape(csdl.cos(theta_0[1,i]), new_shape=(1,1,1))

            if beam_type == 'fuse':
                K[0,0,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.cos(theta[0,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[1,2,i] = csdl.expand(csdl.sin(theta[0,i]),(1,1,1),'ij->ijk')
                K[2,0,i] = csdl.expand(csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[0,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')

                K_0[0,0,i] = csdl.reshape(csdl.cos(theta_0[1,i]),new_shape=(1,1,1))
                K_0[0,2,i] = csdl.reshape(-csdl.cos(theta_0[0,i])*csdl.sin(theta_0[1,i]),new_shape=(1,1,1))
                K_0[1,1,i] = csdl.reshape(one,new_shape=(1,1,1))
                K_0[1,2,i] = csdl.reshape(csdl.sin(theta_0[0,i]),new_shape=(1,1,1))
                K_0[2,0,i] = csdl.reshape(csdl.sin(theta_0[1,i]),new_shape=(1,1,1))
                K_0[2,2,i] = csdl.reshape(csdl.cos(theta_0[0,i])*csdl.cos(theta_0[1,i]),new_shape=(1,1,1))






        # compute the difference/average vectors
        delta_r = self.create_output(name+'delta_r',shape=(3,n-1))
        delta_theta = self.create_output(name+'delta_theta',shape=(3,n-1))
        delta_theta_0 = self.create_output(name+'delta_theta_0',shape=(3,n-1),val=0)
        delta_s = self.create_output(name+'delta_s',shape=(n-1))
        delta_s_0 = self.create_output(name+'delta_s_0',shape=(n-1),val=0)
        delta_F = self.create_output(name+'delta_F',shape=(3,n-1))
        delta_M = self.create_output(name+'delta_M',shape=(3,n-1))
        Fa = self.create_output(name+'Fa',shape=(3,n-1))
        Ma = self.create_output(name+'Ma',shape=(3,n-1))
        fa = self.create_output(name+'fa',shape=(3,n-1))
        ma = self.create_output(name+'ma',shape=(3,n-1))
        delta_fp = self.create_output(name+'delta_fp',shape=(3,n-1),val=0)
        delta_mp = self.create_output(name+'delta_mp',shape=(3,n-1),val=0)
        Ta = self.create_output(name+'Ta',shape=(3,3,n-1),val=0)
        Ka = self.create_output(name+'Ka',shape=(3,3,n-1),val=0)

        for i in range(0,n-1):
            delta_r[:, i] = r[:,i+1] - r[:,i] + 1E-19
            delta_r_0 = r_0[:,i+1] - r_0[:,i] + 1E-19
            delta_s_0[i] = csdl.reshape((delta_r_0[0,0]**2 + delta_r_0[1,0]**2 + delta_r_0[2,0]**2)**0.5, new_shape=(1))
            delta_s[i] = csdl.reshape(((delta_r[0, i])**2 + (delta_r[1, i])**2 + (delta_r[2, i])**2)**0.5, new_shape=(1))
            delta_theta[:,i] = theta[:,i+1] - theta[:,i]
            delta_theta_0[:,i] = theta_0[:,i+1] - theta_0[:,i]
            delta_F[:,i] = F[:,i+1] - F[:,i]
            delta_M[:,i] = M[:,i+1] - M[:,i]
            Fa[:,i] = 0.5*(F[:,i+1] + F[:,i])
            Ma[:,i] = 0.5*(M[:,i+1] + M[:,i])
            fa[:,i] = 0.5*(f[:,i+1] + f[:,i])
            ma[:,i] = 0.5*(m[:,i+1] + m[:,i])

            delta_fp[:,i] = fp[:,i+1]
            delta_mp[:,i] = mp[:,i+1]

            Ta[:,:,i] = 0.5*(T[:,:,i+1] + T[:,:,i])
            Ka[:,:,i] = 0.5*(K[:,:,i+1] + K[:,:,i])


        mass = csdl.sum(mu*delta_s_0)
        self.register_output(name+'mass',mass)


        # compute the point loads
        delta_fj = self.create_output(name+'delta_fj',shape=(3,n-1),val=0)
        delta_mj = self.create_output(name+'delta_mj',shape=(3,n-1),val=0)
        delta_FP = self.create_output(name+'delta_FP',shape=(3,n-1),val=0)
        delta_MP = self.create_output(name+'delta_MP',shape=(3,n-1),val=0)
        for i in range(n-1):
            if i in parent:
                joint_name = parent_dict[i]
                jx = self.declare_variable(joint_name+'x',shape=(12,1),val=0)
                Fj, Mj = jx[6:9,0], jx[9:12,0]
                delta_fj[:,i], delta_mj[:,i] = Fj, Mj # (ASW p27 eq143)
            else: delta_fj[:,i], delta_mj[:,i] = [csdl.expand(zero,(3,1),'i->ij')] * 2
        
        delta_FP[:,:] = delta_fp + delta_fj
        delta_MP[:,:] = delta_mp + delta_mj






        # region strainscsn
        # compute strains in csn
        M_csn = self.create_output(name+'Mcsn',shape=(3,n),val=0)
        F_csn = self.create_output(name+'Fcsn',shape=(3,n),val=0)
        M_csnp = self.create_output(name+'M_csnp',shape=(3,n),val=0)
        strains_csn = self.create_output(name+'strains_csn',shape=(3,n),val=0)
        for i in range(n):
            # transform Fi and Mi in xyz to csn (ASW p6 eq14);
            T_i = csdl.reshape(T[:,:,i], new_shape=(3,3))
            M_i = csdl.reshape(M[:,i],new_shape=(3))
            F_i = csdl.reshape(F[:,i],new_shape=(3))
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
            if i not in child: # if the node is not a child node: (ASW p13 eq56)
                delta_F_i = csdl.reshape(delta_F[:,i], new_shape=(3))
                fa_i = csdl.reshape(fa[:,i], new_shape=(3))
                delta_s_i = csdl.expand(delta_s[i], (3))
                delta_FP_i = csdl.reshape(delta_FP[:,i], new_shape=(3))

                force_equilibrium_residual[:,i] = csdl.expand(prec_fe*(delta_F_i + fa_i*delta_s_i + delta_FP_i), (3,1), 'i->ij')

            # elif i in fixed: # if the node is a fixed node apply the fixed constraints instead
            #     r_i = r[:,i]
            #     r_0_i = r_0[:,i]
            #     force_equilibrium_residual[:,i] = r_i - r_0_i

            elif i in child:
                r_j = next((r for c, r in zip(child, r_j_list) if c == i), None)
                r_i = r[:,i]
                r_0_i = r_0[:,i]
                force_equilibrium_residual[:,i] = r_i - r_0_i - r_j



        # endregion


        # region momentequilibriumresidual
        # moment equilibrium residual (ASW p13 eq55)
        moment_equilibrium_residual = self.create_output(name+'moment_equilibrium_residual', shape=(3,n-1), val=0)
        for i in range(n-1):
            if i not in child:
                delta_M_i = csdl.reshape(delta_M[:,i], new_shape=(3))
                ma_i = csdl.reshape(ma[:,i], new_shape=(3))
                delta_s_i = csdl.expand(delta_s[i], (3))
                delta_MP_i = csdl.reshape(delta_MP[:,i], new_shape=(3))
                delta_r_i = csdl.reshape(delta_r[:,i], new_shape=(3))
                Fa_i = csdl.reshape(Fa[:,i], new_shape=(3))

                moment_equilibrium_residual[:,i] = csdl.expand(prec_me*(delta_M_i + ma_i*delta_s_i + delta_MP_i + csdl.cross(delta_r_i, Fa_i, axis=0)), (3,1), 'i->ij')

            # elif i in fixed:
            #     theta_i = theta[:,i]
            #     theta_0_i = theta_0[:,i]
            #     moment_equilibrium_residual[:,i] = theta_i - theta_0_i

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

        fixed_displacement_residual = self.create_output(name+'fixed_displacement_residual', shape=(3,1), val=0)
        fixed_orientation_residual = self.create_output(name+'fixed_orientation_residual', shape=(3,1), val=0)



        # for i, free_node in enumerate(free):
        #     free_force_residual[:,i] = F[:,int(free_node)] # nodal force at free node is zero
        #     free_moment_residual[:,i] = M[:,int(free_node)] # nodal moment at free node is zero
        
        for free_node in free:
            free_force_residual[:,0] = F[:,int(free_node)] # nodal force at free node is zero
            free_moment_residual[:,0] = M[:,int(free_node)] # nodal moment at free node is zero
        
        
        for fixed_node in fixed:
            fixed_displacement_residual[:,0] = r[:,int(fixed_node)] - r_0[:,int(fixed_node)]
            fixed_orientation_residual[:,0] = theta[:,int(fixed_node)] - theta_0[:,int(fixed_node)]
        
        # endregion



        # region concatenateresidual
        # concatenate residual (12x12 matrix)
        res = self.create_output(name+'res', shape=(12,n))
        res[0:3,0:n-1] = strain_displacement_residual
        res[3:6,0:n-1] = moment_curvature_residual
        res[6:9,0:n-1] = force_equilibrium_residual
        res[9:12,0:n-1] = moment_equilibrium_residual



        # res[0:3,n-1] = free_force_residual[:,0]
        # res[3:6,n-1] = free_moment_residual[:,0]
        # res[6:9,n-1] = fixed_displacement_residual
        # res[9:12,n-1] = fixed_orientation_residual
        # endregion

        if options['child'] == False:
            res[0:3,n-1] = free_force_residual[:,0]
            res[3:6,n-1] = free_moment_residual[:,0]
            res[6:9,n-1] = fixed_displacement_residual
            res[9:12,n-1] = fixed_orientation_residual

        elif options['child'] == True:
            pass

        






