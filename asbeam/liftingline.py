import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt


class linear_solver(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_segments')
    def define(self):
        N = self.parameters['num_segments']

        B_mat = self.declare_variable('B_mat',shape=(N,N))
        LHS = self.declare_variable('LHS',shape=(N,1))
        A = self.declare_variable('A',shape=(N,1))

        residual = csdl.matmat(B_mat,A) - LHS
        self.register_output('residual', residual)


# modified from Sadraey aircraft design 2012
class lifting_line(csdl.Model):
    def initialize(self):
        self.parameters.declare('num_segments')
    def define(self):
        N = self.parameters['num_segments']

        S = self.declare_variable('wing_area',val=25) # (m^2)
        AR = self.declare_variable('aspect_ratio',val=8)
        TR = self.declare_variable('taper_ratio',val=0.6)
        alpha_twist = self.declare_variable('alpha_twist',val=-1) # (deg)
        i_w = self.declare_variable('i_w',val=2) # wing set angle (deg)
        a_2d = self.declare_variable('lift_slope',val=6.3) # (rad^-1)
        alpha_0 = self.declare_variable('zero_lift_aoa',val=-1.5) # (deg)

        b = (AR*S)**0.5 # wing span (m)
        self.register_output('b',b)
        MAC = S/b # mean aerodynamic chord (m)
        Croot = (1.5*(1+TR)*MAC)/(1+TR+TR**2) # root chord (m)
        theta_1d = self.create_input('theta',val=np.linspace(np.pi/(2*N),np.pi/2,N),shape=(N,)) # angular position of each segment (rad)
        theta = csdl.expand(theta_1d,(N,1),'ij->iajb')

        #self.print_var(theta)

        c = csdl.expand(Croot,(N,1)) - csdl.expand(Croot - Croot*TR,(N,1))*csdl.cos(theta)

        mu = c*csdl.expand(a_2d,(N,1))/csdl.expand((4.0*b),(N,1))

        #self.print_var(mu)

        alpha = self.create_output('seg_alpha',shape=(N,1))
        for i in range(N):
            alpha[i,0] = csdl.expand((i_w+alpha_twist) - i*alpha_twist/(N-1),(1,1))

        const = self.create_input('const',val=57.3) # convert deg to rad in the following equation
        LHS = mu*(alpha - csdl.expand(alpha_0,(N,1)))/csdl.expand(const,(N,1))
        self.register_output('LHS',LHS)
        
        # solving N equations to find coefficients A(i):
        B_mat = self.create_output('B_mat',shape=(N,N))
        for i in range(0,N):
            for j in range(0,N):
                B_mat[i,j] = csdl.sin((2*(j+1) - 1)*theta[i,0])*(1 + (mu[i,0]*(2*(j+1) - 1))/csdl.sin(theta[i,0]))

        # solve linear system with an implicit operation
        solver = self.create_implicit_operation(linear_solver(num_segments=N))
        solver.declare_state('A', residual='residual')
        solver.nonlinear_solver = csdl.NewtonSolver(
        solve_subsystems=False,
        maxiter=200,
        iprint=False,
        )
        solver.linear_solver = csdl.ScipyKrylov()

        A = solver(B_mat,LHS)
        # end linear system

        # compute total wing cl
        cl_wing_2d = csdl.expand(np.pi*AR,(1,1))*A[0,0]
        cl_wing = csdl.reshape(cl_wing_2d,new_shape=(1,))
        self.register_output('cl',cl_wing)
        # compute induced drag coefficient
        cd_i_2d = (cl_wing_2d**2)/csdl.expand((np.pi*AR),(1,1))
        cd_i = csdl.reshape(cd_i_2d,new_shape=(1,))
        self.register_output('cd_i',cd_i)


        
        # calculate lift coefficient distribution
        k = self.declare_variable('k', val=(np.arange(1,2*N,2)).reshape((N,1)), shape=(N,1)) # np.arange(1,2*N-1,2)
        kphi = csdl.transpose(csdl.matmat(k, csdl.transpose(theta)))
        sin_kphi = csdl.sin(kphi)
        one = self.declare_variable('one', val=1, shape=(N,1))
        oa = csdl.matmat(one, csdl.transpose(A))
        oap = oa*sin_kphi # element wise haddemard product
        row_sum = csdl.matmat(csdl.transpose(one), csdl.transpose(oap))
        sum_cl = csdl.transpose(row_sum)
        coef = csdl.expand(4*b, (N,1))
        cl_dist = (coef*sum_cl)/c
        self.register_output('cl_dist', cl_dist)
        




n = 9

sim = python_csdl_backend.Simulator(lifting_line(num_segments=n))
sim.run()
print(sim['cl'])
print(sim['cd_i'])




# for plotting lift coefficient vs half-span
A = sim['A']
cl_dist = sim['cl_dist']
b = sim['b']
zero = np.zeros((1,1))
Cl2 = np.concatenate((zero, cl_dist))
dy = (b/2)/n
ys = np.arange(0, (b/2), dy).reshape((n,1))
ys = np.concatenate((ys, [ys[-1,0] + dy]))
plt.plot(np.flip(ys),Cl2)
plt.title('Lift Coefficient Distribution vs Span')
plt.xlabel('span (m)')
plt.ylabel('Cl')
plt.show()
