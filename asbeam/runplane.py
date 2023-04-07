import csdl
import numpy as np
import python_csdl_backend
import matplotlib.pyplot as plt
from groupimplicitop import GroupImplicitOp
from inputs import *




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        self.add(GroupImplicitOp(beams=beams, joints=joints), name='GroupImplicitOp') # solve the beam-joint system





if __name__ == '__main__':

    sim = python_csdl_backend.Simulator(Run(beams=beams, joints=joints))

    
    # cs params
    name = 'wing'
    sim[name+'h'] = 0.25
    sim[name+'w'] = 1
    sim[name+'t_left'] = 0.03
    sim[name+'t_top'] = 0.03
    sim[name+'t_right'] = 0.03
    sim[name+'t_bot'] = 0.03

    sim[name+'r_0'] = wing_r_0
    sim[name+'theta_0'] = wing_theta_0
    sim[name+'f'] = wing_fa
    
    
    name = 'fuse'
    sim[name+'h'] = 0.5
    sim[name+'w'] = 0.5
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = fuse_r_0
    sim[name+'theta_0'] = fuse_theta_0
    sim[name+'f'] = fuse_fa


    name = 'lboom'
    sim[name+'h'] = 0.2
    sim[name+'w'] = 0.2
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = lboom_r_0
    sim[name+'theta_0'] = lboom_theta_0
    sim[name+'f'] = lboom_fa


    name = 'rboom'
    sim[name+'h'] = 0.2
    sim[name+'w'] = 0.2
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = rboom_r_0
    sim[name+'theta_0'] = rboom_theta_0
    sim[name+'f'] = rboom_fa


    name = 'tail'
    sim[name+'h'] = 0.2
    sim[name+'w'] = 0.2
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = tail_r_0
    sim[name+'theta_0'] = tail_theta_0
    



    sim.run()




    # plot the beams
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    i = 0
    for beam_name in beams:
        n = beams[beam_name]['n']
        beam_state = sim['x'][:,i:i+n]
        x = beam_state[0,:]
        y = beam_state[1,:]
        z = beam_state[2,:]

        ax.scatter(x,y,z)
        ax.plot(x,y,z,color='k',linewidth=1)

        i += n
 
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-20,20)
    ax.set_ylim(-20,20)
    ax.set_zlim(-0.5,1)
    plt.show()