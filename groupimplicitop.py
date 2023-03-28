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


        solve_res = self.create_implicit_operation(Group(beams=beams,jonts=joints))
        solve_res.declare_state('rj', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
        iprint=False,
        )
        solve_res.linear_solver = csdl.ScipyKrylov()