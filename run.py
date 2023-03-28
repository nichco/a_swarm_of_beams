import csdl
import numpy as np
import python_csdl_backend
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


if __name__ == '__main__':

    beams = {}

    # wing beam
    name = 'wing'
    beams[name] = {}
    beams[name]['n'] = 8
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'wing'
    beams[name]['free'] = np.array([0,7])
    beams[name]['fixed'] = np.array([3])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['rho'] = 2700
    beams[name]['dir'] = 1
    
    # fuselage beam
    name = 'fuse'
    beams[name] = {}
    beams[name]['n'] = 8
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'fuse'
    beams[name]['free'] = np.array([0,7])
    beams[name]['fixed'] = np.array([2])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['rho'] = 2700
    beams[name]['dir'] = 1


    sim = python_csdl_backend.Simulator(Run(beams=beams))

    
    # cs params
    name = 'wing'
    sim[name+'h'] = 0.25
    sim[name+'w'] = 1
    sim[name+'t_left'] = 0.05
    sim[name+'t_top'] = 0.05
    sim[name+'t_right'] = 0.05
    sim[name+'t_bot'] = 0.05

    theta_0 = np.zeros((3,beams['wing']['n']))
    sim[name+'theta_0'] = theta_0
    r_0 = np.zeros((3,beams['wing']['n']))
    r_0[1,:] = np.array([-3,-2,-1,0,0,1,2,3])
    sim[name+'r_0'] = r_0
    
    
    name = 'fuse'
    sim[name+'h'] = 0.25
    sim[name+'w'] = 1
    sim[name+'t_left'] = 0.05
    sim[name+'t_top'] = 0.05
    sim[name+'t_right'] = 0.05
    sim[name+'t_bot'] = 0.05

    theta_0 = np.zeros((3,beams['fuse']['n']))
    theta_0[2,:] = np.ones(beams['fuse']['n'])*-np.pi/2
    sim[name+'theta_0'] = theta_0
    r_0 = np.zeros((3,beams['fuse']['n']))
    r_0[0,:] = np.array([-2,-1,0,0,1,2,3,4])
    sim[name+'r_0'] = r_0
    



    sim.run()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for beam_name in beams:
        x = sim[beam_name+'x'][0,:]
        y = sim[beam_name+'x'][1,:]
        z = sim[beam_name+'x'][2,:]

        ax.scatter(x,y,z)

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-3,3)
    ax.set_ylim(-3,3)
    ax.set_zlim(-0.5,0.5)
    plt.show()