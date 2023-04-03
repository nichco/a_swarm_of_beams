import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt




class AeroSolver(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
    def define(self):
        n = self.parameters['n']
        B = self.declare_variable('B',shape=(n,n))
        LHS = self.declare_variable('LHS',shape=(n))
        A = self.declare_variable('A',shape=(n))
        residual = csdl.matvec(B,A) - LHS
        self.register_output('residual', residual)


class Aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('N')
    def define(self):
        N = self.parameters['N']

        croot = self.declare_variable('croot',val=2.1)
        ctip = self.declare_variable('ctip',val=1)
        tr = ctip/croot
        b = self.declare_variable('b',val=10)
        cla = self.declare_variable('cla',val=6.3) # (rad^-1)
        alpha_0 = self.declare_variable('zero_lift_aoa',val=-1.5) # (deg)
        alpha = self.declare_variable('alpha',shape=(N),val=2)

        S = b*((ctip+croot)/2)
        AR = (b**2)/S

        theta = self.create_input('theta',val=np.linspace(np.pi/(2*N),np.pi/2,N),shape=(N)) # angular position of each segment (rad)

        #theta = self.create_output('theta',shape=(N),val=0)
        #for i in range(N):
        #    theta[i] = csdl.arccos(2*((y[i]**2)**0.5)/b)
        
        c = self.create_output('c',shape=(N),val=0)
        for i in range(N): c[i] = croot*(1+(tr-1)*csdl.cos(theta[i]))
        
        mu = c*csdl.expand(cla/(4*b), (N))
        
        LHS = mu*(alpha - csdl.expand(alpha_0, (N)))/57.3
        self.register_output('LHS',LHS)
     
        B = self.create_output('B',shape=(N,N))
        for i in range(0,N):
            for j in range(0,N):
                term1 = csdl.sin((2*(j+1) - 1)*theta[i])
                term2 = (1 + (mu[i]*(2*(j+1)-1)))/csdl.sin(theta[i])
                B[i,j] = csdl.expand(term1*term2, (1,1), 'i->ij')


        solver = self.create_implicit_operation(AeroSolver(n=N))
        solver.declare_state('A', residual='residual')
        solver.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=100,
        iprint=False,)
        solver.linear_solver = csdl.ScipyKrylov()

        A = solver(B,LHS)

        cl = np.pi*AR*A[0]
        self.register_output('cl',cl)
        self.print_var(cl)

        delta_vec = self.create_output('delta',shape=(N),val=0)
        for i in range(2,N):
            delta_vec[i] = (2*i - 1)*((A[i-1]/A[0])**2)

        delta = csdl.sum(delta_vec)
        e = 1/(1+delta)
        cdi = (cl**2)/(np.pi*AR*e)

        self.print_var(cdi)
        self.register_output('cdi',cdi)
        




if __name__ == '__main__':


    sim = python_csdl_backend.Simulator(Aero(N=9))
    sim.run()