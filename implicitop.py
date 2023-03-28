import csdl
import numpy as np
from beamresidual import BeamRes




class ImplicitOp(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        name = options['name']
        n = options['n']


        solve_res = self.create_implicit_operation(BeamRes(options=options))
        solve_res.declare_state(name+'x', residual=name+'res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()

        r_0 = self.declare_variable(name+'r_0',shape=(3,n),val=0)
        theta_0 = self.declare_variable(name+'theta_0',shape=(3,n),val=0)
        E_inv = self.declare_variable(name+'E_inv',shape=(3,3,n),val=0)
        D = self.declare_variable(name+'D',shape=(3,3,n),val=0)
        oneover = self.declare_variable(name+'oneover',shape=(3,3,n),val=0)


        solve_res(r_0, theta_0, E_inv, D, oneover)