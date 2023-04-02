import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt




class Aero(csdl.Model):
    def initialize(self):
        self.parameters.declare('n')
    def define(self):
        n = self.parameters['n']


        y = self.declare_variable('y',shape=(n),val=0)
        tr = self.declare_variable('tr',val=0.5)
        croot = self.declare_variable('croot',val=0.5)
        alpha = self.declare_variable('alpha',shape=(n-2),val=np.deg2rad(2))
        alpha_0 = self.declare_variable('zero_lift_aoa',val=np.deg2rad(-1.5))
        cla = self.declare_variable('cla',val=6.3)
        b = 2*y[-1]

        theta = self.create_output('theta',shape=(n-2),val=0)
        for i in range(1,n-1):
            theta[i-1] = csdl.arccos(2*(y[i]**2)**0.5/b)
        self.print_var(theta)
        
        c = self.create_output('c',shape=(n-2),val=0)
        for i in range(1,n-1):
            c[i-1] = croot*(1+(tr-1)*csdl.cos(theta[i-1]))
        self.print_var(c)
        
        mu = c*csdl.expand(cla/(4*b), (n-2))
        
        LHS = mu*(alpha - csdl.expand(alpha_0, (n-2)))
        self.register_output('LHS',LHS)

        """
        B = self.create_output('B',shape=(n,n))
        for i in range(1,n-1):
            for j in range(1,n-1):
                term1 = csdl.sin((2*j - 1)*theta[i])
                term2 = 1 + (mu[i]*(2*j - 1))/csdl.sin(theta[i])
                B[i,j] = csdl.expand(term1*term2, (1,1))

                #self.print_var(term1)
        """
        #RHS = []
        #for i in range(1,2*n+1,2):
        #    RHS_iter = np.sin(i * theta) * (1 + (mu * i) / (np.sin(list(theta))))
        #    RHS.append(RHS_iter)





if __name__ == '__main__':

    n = 6 # num nodes
    b = 10 # span
    y = np.zeros((n))
    for i in range(n):
        y[i] = (-b/2)*(1-((2*i)/(n-1)))

    print(y)


    sim = python_csdl_backend.Simulator(Aero(n=n))
    sim['y'] = y
    sim.run()