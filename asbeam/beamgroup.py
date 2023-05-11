import csdl
import python_csdl_backend
import numpy as np



class BeamGroup(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        # compute the total number of nodes for the entire beam group:
        num_nodes = 0
        for beam_name in beams: num_nodes = num_nodes + beams[beam_name]['n']

        # concatenate the variables as inputs to the implicit operation
        r_0 = self.create_output('r_0',shape=(3,num_nodes))
        theta_0 = self.create_output('theta_0',shape=(3,num_nodes))
        E_inv = self.create_output('E_inv',shape=(3,3,num_nodes))
        D = self.create_output('D',shape=(3,3,num_nodes))
        oneover = self.create_output('oneover',shape=(3,3,num_nodes))
        f = self.create_output('f',shape=(3,num_nodes))
        m = self.create_output('m',shape=(3,num_nodes))
        fp = self.create_output('fp',shape=(3,num_nodes))
        mp = self.create_output('mp',shape=(3,num_nodes))

        i = 0
        for beam_name in beams:
            n = beams[beam_name]['n']
            r_0[:,i:i+n] = self.declare_variable(beam_name+'r_0',shape=(3,n),val=0)
            theta_0[:,i:i+n] = self.declare_variable(beam_name+'theta_0',shape=(3,n),val=0)
            E_inv[:,:,i:i+n] = self.declare_variable(beam_name+'E_inv',shape=(3,3,n),val=0)
            D[:,:,i:i+n] = self.declare_variable(beam_name+'D',shape=(3,3,n),val=0)
            oneover[:,:,i:i+n] = self.declare_variable(beam_name+'oneover',shape=(3,3,n),val=0)
            f[:,i:i+n] = self.declare_variable(beam_name+'f',shape=(3,n),val=0)
            m[:,i:i+n] = self.declare_variable(beam_name+'m',shape=(3,n),val=0)
            fp[:,i:i+n] = self.declare_variable(beam_name+'fp',shape=(3,n),val=0)
            mp[:,i:i+n] = self.declare_variable(beam_name+'mp',shape=(3,n),val=0)
            i += n



        # define the implicit operation
        solve_res = self.create_implicit_operation(Model(beams=beams,joints=joints))
        solve_res.declare_state('x', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()