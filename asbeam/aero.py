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
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N']


        y = self.declare_variable('y',shape=(N),val=0)
        tr = self.declare_variable('tr',val=0.6)
        croot = self.declare_variable('croot',val=2.1)
        alpha = self.declare_variable('alpha',shape=(N),val=2)
        alpha_0 = self.declare_variable('zero_lift_aoa',val=-1.5)
        cla = self.declare_variable('cla',val=6.3)
        b = self.declare_variable('b',val=10)

        theta = self.create_output('theta',shape=(N),val=0)
        for i in range(N):
            theta[i] = csdl.arccos(2*((y[i]**2)**0.5)/b)
        #self.print_var(theta)
        
        c = self.create_output('c',shape=(N),val=0)
        for i in range(N):
            c[i] = croot*(1+(tr-1)*csdl.cos(theta[i]))
        #self.print_var(c)
        
        mu = c*csdl.expand(cla/(4*b), (N))
        self.print_var(mu)
        
        LHS = mu*(alpha - csdl.expand(alpha_0, (N)))/57.3
        self.register_output('LHS',LHS)

        self.print_var(LHS)

        
        B = self.create_output('B',shape=(N,N))
        #for i in range(1,n-1):
        #    for j in range(1,n-1):
        #        term1 = csdl.sin((2*(j) - 1)*theta[i-1])
        #        term2 = 1 + (mu[i-1]*(2*(j) - 1))/csdl.sin(theta[i-1])
        #        B[i-1,j-1] = csdl.expand(term1*term2, (1,1))
        


        for i in range(0,N):
            for j in range(0,N):
                term1 = csdl.sin((2*(j+1) - 1)*theta[i])
                term2 = (1 + (mu[i]*(2*(j+1)-1)))/csdl.sin(theta[i])
                B[i,j] = csdl.expand(term1*term2, (1,1), 'i->ij')
        

        #for i=1:N
        #    for j=1:N
        #        B(i,j) = sin((2*j-1) * theta(i)) * (1 + (mu(i) * (2*j-1)) / sin(theta(i)));
        #    end
        #end


        solver = self.create_implicit_operation(AeroSolver(n=n))
        solver.declare_state('A', residual='residual')
        solver.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,)
        solver.linear_solver = csdl.ScipyKrylov()

        A = solver(B,LHS)
        self.print_var(A)
        #self.print_var(LHS)

        s = b*((croot*(tr+1))/2)
        AR = b**2/s

        cl = np.pi*AR*A[0]
        self.print_var(cl)
        




if __name__ == '__main__':

    n = 21 # num nodes
    N = n - 2
    b = 14 # span
    y = np.zeros((n))
    y_in = y[1:-1]
    for i in range(n):
        y[i] = (-b/2)*(1-((2*i)/(n-1)))

    #print(y_in)


    sim = python_csdl_backend.Simulator(Aero(N=N))
    sim['y'] = y_in
    sim['b'] = b
    sim.run()