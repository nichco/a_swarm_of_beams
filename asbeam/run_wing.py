import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt
from implicitop import ImplicitOp
from boxbeamrep import BoxBeamRep
from stress import Stress
from modopt.scipy_library import SLSQP
from modopt.csdl_library import CSDLProblem




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        beam_name = options['name']
        n = options['n']

        self.create_input(beam_name+'h',shape=(n))
        self.create_input(beam_name+'t_left',shape=(n))
        self.create_input(beam_name+'t_right',shape=(n))
        self.create_input(beam_name+'t_top',shape=(n))
        self.create_input(beam_name+'t_bot',shape=(n))


        self.add(BoxBeamRep(options=options), name=beam_name+'BoxBeamRep') # get beam properties
        self.add(ImplicitOp(options=options), name=beam_name+'ImplicitOp') # solve the beam
        self.add(Stress(options=options), name=beam_name+'Stress') # stress recovery

        x = self.declare_variable(beam_name+'x',shape=(12,options['n']),val=0)
        r = x[0:3,:]
        theta = x[3:6,:]
        self.register_output(beam_name+'max_z',csdl.max(r[2,:]))

        mass = self.declare_variable(beam_name+'mass',val=0)
        self.print_var(mass)

        #self.add_design_variable(beam_name+'h',lower=0.1)
        self.add_design_variable(beam_name+'t_left',lower=0.002)
        self.add_design_variable(beam_name+'t_right',lower=0.002)
        self.add_design_variable(beam_name+'t_top',lower=0.002)
        self.add_design_variable(beam_name+'t_bot',lower=0.002)
        #self.add_constraint(beam_name+'max_z',upper=0.8)
        self.add_constraint(beam_name+'max_sigma_vm_fos',upper=options['SY'],scaler=1E-8)
        self.add_objective(beam_name+'mass',scaler=1E-3)




if __name__ == '__main__':

    beams = {}
    name = 'wing'
    beams[name] = {}
    beams[name]['n'] = 59
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'wing'
    beams[name]['free'] = np.array([0,beams[name]['n']-1])
    beams[name]['fixed'] = np.array([29])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['SY'] = 450E6 # yield stress (MPa)
    beams[name]['rho'] = 2700
    beams[name]['dir'] = 1

    span = 20
    r_0 = np.zeros((3,beams[name]['n']))
    r_0[1,:] = np.linspace(-span/2,span/2,beams[name]['n'])
    theta_0 = np.zeros((3,beams[name]['n']))
    f = np.zeros((3,beams[name]['n']))
    f[2,:] = 20000

    fp = np.zeros((3,beams[name]['n']))
    fp[2,beams[name]['n']-1] = 20000

    sim = python_csdl_backend.Simulator(Run(options=beams[name]))

    sim[name+'h'] = 0.5
    sim[name+'w'] = 0.5
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = r_0
    sim[name+'theta_0'] = theta_0
    #sim[name+'f'] = f
    sim[name+'fp'] = fp

    sim.run()

    #prob = CSDLProblem(problem_name='run_wing', simulator=sim)
    #optimizer = SLSQP(prob, maxiter=1000, ftol=1E-6)
    #optimizer.solve()
    #optimizer.print_results()

    F = 20000
    L = 10
    E = 69E9
    I = sim[name+'Ixx'][0]
    dmax = F*L**3/(3*E*I)
    print('dmax: ',dmax)



    x = sim[name+'x'][0,:]
    y = sim[name+'x'][1,:]
    z = sim[name+'x'][2,:]
    print(z[-1])

    plt.scatter(y,z)
    plt.plot(y,z,color='k')
    plt.ylim(-0.25,0.25)
    plt.scatter(0,0,color='k')
    plt.show()
    """
    t_left = sim[name+'t_left']
    t_right = sim[name+'t_right']
    t_top = sim[name+'t_top']
    t_bot = sim[name+'t_bot']

    plt.plot(y,t_left)
    plt.plot(y,t_right)
    plt.plot(y,t_top)
    plt.plot(y,t_bot)
    plt.legend(['t_left','t_right','t_top','t_bot'])
    plt.show()

    sigma_vm = sim[name+'sigma_vm']
    plt.plot(sigma_vm[0,:])
    plt.plot(sigma_vm[1,:])
    plt.plot(sigma_vm[2,:])
    plt.plot(sigma_vm[3,:])
    plt.show()
    """