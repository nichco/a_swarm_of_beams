import csdl
import numpy as np
from group import Group




class GroupImplicitOp(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']


        solve_res = self.create_implicit_operation(Group(beams=beams,joints=joints))
        solve_res.declare_state('x', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()


        num_nodes = 0
        for beam_name in beams: num_nodes = num_nodes + beams[beam_name]['n']


        vars = {'r_0': (3,num_nodes),
                'theta_0': (3,num_nodes),
                'E_inv': (3,3,num_nodes),
                'D': (3,3,num_nodes),
                'oneover': (3,3,num_nodes),
                'f': (3,num_nodes),
                'm': (3,num_nodes), # distributed moments
                'fp': (3,num_nodes), # point loads
                'mp': (3,num_nodes)} # point moments
        
        var_list = [self.declare_variable(var_name, shape=var_shape, val=0) for var_name, var_shape in vars.items()]


        solve_res(*var_list)