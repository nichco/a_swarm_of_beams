import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt




class AeroSolver(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
    def define(self):
        n = self.parameters['n']

        B = self.declare_variable('B',shape=(n-2,n-2))
        LHS = self.declare_variable('LHS',shape=(n-2))
        A = self.declare_variable('A',shape=(n-2))

        residual = csdl.matvec(B,A) - LHS
        self.register_output('residual', residual)


class Aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
    def define(self):
        n = self.parameters['n']


        y = self.declare_variable('y',shape=(n),val=0)
        tr = self.declare_variable('tr',val=0.6)
        croot = self.declare_variable('croot',val=2)
        alpha = self.declare_variable('alpha',shape=(n-2),val=np.deg2rad(2))
        alpha_0 = self.declare_variable('zero_lift_aoa',val=np.deg2rad(-1.5))
        cla = self.declare_variable('cla',val=6.3)
        b = 2*y[-1]

        theta = self.create_output('theta',shape=(n-2),val=0)
        for i in range(1,n-1):
            theta[i-1] = csdl.arccos(2*(y[i]**2)**0.5/b)
        #self.print_var(theta)
        
        c = self.create_output('c',shape=(n-2),val=0)
        for i in range(1,n-1):
            c[i-1] = croot*(1+(tr-1)*csdl.cos(theta[i-1]))
        #self.print_var(c)
        
        mu = c*csdl.expand(cla/(4*b), (n-2))
        
        LHS = mu*(alpha - csdl.expand(alpha_0, (n-2)))
        self.register_output('LHS',LHS)

        
        B = self.create_output('B',shape=(n-2,n-2))
        for i in range(1,n-1):
            for j in range(1,n-1):
                term1 = csdl.sin((2*(j) - 1)*theta[i-1])
                term2 = 1 + (mu[i-1]*(2*(j) - 1))/csdl.sin(theta[i-1])
                B[i-1,j-1] = csdl.expand(term1*term2, (1,1))

        for i in range(1,n):


        #for i in range(0,N):
        #    for j in range(0,N):
        #        B_mat[i,j] = csdl.sin((2*(j+1) - 1)*theta[i,0])*(1 + (mu[i,0]*(2*(j+1) - 1))/csdl.sin(theta[i,0]))

        #RHS = []
        #for i in range(1, 2 * N + 1, 2):
        #    RHS_iter = np.sin(i * theta) * (
        #        1 + (mu * i) / (np.sin(list(theta)))
        #    )  # .reshape(1,N)
        #    # print(RHS_iter,"RHS_iter shape")
        #    RHS.append(RHS_iter)


        solver = self.create_implicit_operation(AeroSolver(n=n))
        solver.declare_state('A', residual='residual')
        solver.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,)
        solver.linear_solver = csdl.ScipyKrylov()

        A = solver(B,LHS)
        self.print_var(A)

        s = b*((croot*(tr+1))/2)
        AR = b**2/s

        cl = np.pi*AR*A[0]





if __name__ == '__main__':

    n = 8 # num nodes
    b = 10 # span
    y = np.zeros((n))
    for i in range(n):
        y[i] = (-b/2)*(1-((2*i)/(n-1)))

    print(y)


    sim = python_csdl_backend.Simulator(Aero(n=n))
    sim['y'] = y
    sim.run()