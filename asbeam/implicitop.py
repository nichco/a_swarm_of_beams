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
        maxiter=100,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()



        vars = {'r_0': (3,n), # original nodal coordinates
                'theta_0': (3,n), # initial beam orientation
                'E_inv': (3,3,n), # inverse stiffness matrix
                'D': (3,3,n),
                'oneover': (3,3,n),
                'f': (3,n), # distributed loads
                'm': (3,n), # distributed moments
                'fp': (3,n), # point loads
                'mp': (3,n)} # point moments
        
        var_list = [self.declare_variable(name+var_name, shape=var_shape, val=0) for var_name, var_shape in vars.items()]


        solve_res(*var_list, expose=[name+'mass'])