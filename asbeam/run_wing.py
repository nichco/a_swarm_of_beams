import csdl
import python_csdl_backend
import numpy as np
import matplotlib.pyplot as plt
from implicitop import ImplicitOp
from boxbeamrep import BoxBeamRep




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
    def define(self):
        beams = self.parameters['beams']


        for beam_name in beams:
            self.add(BoxBeamRep(options=beams[beam_name]), name=beam_name+'BoxBeamRep') # get beam properties
            self.add(ImplicitOp(options=beams[beam_name]), name=beam_name+'ImplicitOp') # solve the beam






beams = {}
name = 'wing'
beams[name] = {}
beams[name]['n'] = 15
beams[name]['name'] = name
beams[name]['beam_type'] = 'wing'
beams[name]['free'] = np.array([0,beams[name]['n']-1])
beams[name]['fixed'] = np.array([8])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['dir'] = 1

span = 10
r_0 = np.zeros((3,beams[name]['n']))
r_0[1,:] = np.linspace(-span/2,span/2,beams[name]['n'])
theta_0 = np.zeros((3,beams[name]['n']))
f = np.zeros((3,beams[name]['n']))
f[2,:] = 10000


sim = python_csdl_backend.Simulator(Run(beams=beams))

sim[name+'h'] = 0.25
sim[name+'w'] = 1
sim[name+'t_left'] = 0.05
sim[name+'t_top'] = 0.05
sim[name+'t_right'] = 0.05
sim[name+'t_bot'] = 0.05

sim[name+'r_0'] = r_0
sim[name+'theta_0'] = theta_0
sim[name+'f'] = f

sim.run()



x = sim[name+'x'][0,:]
y = sim[name+'x'][1,:]
z = sim[name+'x'][2,:]

print(x)
print(y)
print(z)

plt.scatter(y,z)
plt.show()