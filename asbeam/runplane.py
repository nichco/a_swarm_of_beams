import csdl
import numpy as np
import python_csdl_backend
import matplotlib.pyplot as plt
from groupimplicitop import GroupImplicitOp
from boxbeamrep import BoxBeamRep
from tubebeamrep import TubeBeamRep
from inputs import *




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
    def define(self):
        beams = self.parameters['beams']

        
        for beam_name in beams:
            if beams[beam_name]['shape'] == 'box':
                self.add(BoxBeamRep(options=beams[beam_name]), name=beam_name+'BoxBeamRep') # get beam properties for box beams
            elif beams[beam_name]['shape'] == 'tube':
                self.add(TubeBeamRep(options=beams[beam_name]), name=beam_name+'TubeBeamRep') # get beam properties tubular beams
        


        num_nodes = 0
        for beam_name in beams: num_nodes = num_nodes + beams[beam_name]['n']

        # concatenate variables
        r_0 = self.create_output('r_0',shape=(3,num_nodes))
        theta_0 = self.create_output('theta_0',shape=(3,num_nodes))
        E_inv = self.create_output('E_inv',shape=(3,3,num_nodes))
        D = self.create_output('D',shape=(3,3,num_nodes))
        oneover = self.create_output('oneover',shape=(3,3,num_nodes))
        fa = self.create_output('fa',shape=(3,num_nodes))

        i = 0
        for beam_name in beams:
            n = beams[beam_name]['n']
            r_0[:,i:i+n] = self.declare_variable(beam_name+'r_0',shape=(3,n),val=0)
            theta_0[:,i:i+n] = self.declare_variable(beam_name+'theta_0',shape=(3,n),val=0)
            E_inv[:,:,i:i+n] = self.declare_variable(beam_name+'E_inv',shape=(3,3,n),val=0)
            D[:,:,i:i+n] = self.declare_variable(beam_name+'D',shape=(3,3,n),val=0)
            oneover[:,:,i:i+n] = self.declare_variable(beam_name+'oneover',shape=(3,3,n),val=0)
            fa[:,i:i+n] = self.declare_variable(beam_name+'fa',shape=(3,n),val=0)
            i += n
        
        self.add(GroupImplicitOp(beams=beams, joints=joints), name='GroupImplicitOp') # solve the beam-joint system


if __name__ == '__main__':
    

    sim = python_csdl_backend.Simulator(Run(beams=beams))

    
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
    sim[name+'fa'] = wing_fa
    
    
    name = 'fuse'
    sim[name+'h'] = 0.5
    sim[name+'w'] = 0.5
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = fuse_r_0
    sim[name+'theta_0'] = fuse_theta_0
    sim[name+'fa'] = fuse_fa


    name = 'lboom'
    sim[name+'h'] = 0.2
    sim[name+'w'] = 0.2
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = lboom_r_0
    sim[name+'theta_0'] = lboom_theta_0
    sim[name+'fa'] = lboom_fa


    name = 'rboom'
    sim[name+'h'] = 0.2
    sim[name+'w'] = 0.2
    sim[name+'t_left'] = 0.01
    sim[name+'t_top'] = 0.01
    sim[name+'t_right'] = 0.01
    sim[name+'t_bot'] = 0.01

    sim[name+'r_0'] = rboom_r_0
    sim[name+'theta_0'] = rboom_theta_0
    sim[name+'fa'] = rboom_fa


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