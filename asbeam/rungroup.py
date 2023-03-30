import csdl
import numpy as np
import python_csdl_backend
import matplotlib.pyplot as plt
from groupimplicitop import GroupImplicitOp
from boxbeamrep import BoxBeamRep
from tubebeamrep import TubeBeamRep




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
    def define(self):
        beams = self.parameters['beams']

        num_nodes = 0
        for beam_name in beams: num_nodes = num_nodes + beams[beam_name]['n']

        # concatenate variables
        r_0 = self.create_output('r_0',shape=(3,num_nodes))
        theta_0 = self.create_output('theta_0',shape=(3,num_nodes))
        E_inv = self.create_output('E_inv',shape=(3,3,num_nodes))
        D = self.create_output('D',shape=(3,3,num_nodes))
        oneover = self.create_output('oneover',shape=(3,3,num_nodes))
        f = self.create_output('f',shape=(3,num_nodes))

        i = 0
        for beam_name in beams:
            n = beams[beam_name]['n']
            r_0[:,i:i+n] = self.declare_variable(beam_name+'r_0',shape=(3,n),val=0)
            theta_0[:,i:i+n] = self.declare_variable(beam_name+'theta_0',shape=(3,n),val=0)
            E_inv[:,:,i:i+n] = self.declare_variable(beam_name+'E_inv',shape=(3,3,n),val=0)
            D[:,:,i:i+n] = self.declare_variable(beam_name+'D',shape=(3,3,n),val=0)
            oneover[:,:,i:i+n] = self.declare_variable(beam_name+'oneover',shape=(3,3,n),val=0)
            f[:,i:i+n] = self.declare_variable(beam_name+'f',shape=(3,n),val=0)
            i += n


        for beam_name in beams:
            if beams[beam_name]['shape'] == 'box':
                self.add(BoxBeamRep(options=beams[beam_name]), name=beam_name+'BoxBeamRep') # get beam properties for box beams
            elif beams[beam_name]['shape'] == 'tube':
                self.add(TubeBeamRep(options=beams[beam_name]), name=beam_name+'TubeBeamRep') # get beam properties tubular beams
        
        
        self.add(GroupImplicitOp(beams=beams, joints=joints), name='GroupImplicitOp') # solve the beam-joint system


if __name__ == '__main__':

    beams, joints = {}, {}

    # wing beam
    name = 'wing'
    beams[name] = {}
    beams[name]['n'] = 8
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'wing'
    beams[name]['free'] = np.array([0,7])
    beams[name]['fixed'] = np.array([4])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['rho'] = 2700
    beams[name]['shape'] = 'box'
    
    # fuselage beam
    name = 'fuse'
    beams[name] = {}
    beams[name]['n'] = 8
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'fuse'
    beams[name]['free'] = np.array([0,7])
    beams[name]['fixed'] = np.array([])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['rho'] = 2700
    beams[name]['shape'] = 'tube'

    
    # joint
    name = 'wingfuse'
    joints[name] = {}
    joints[name]['name'] = name
    joints[name]['parent_name'] = 'wing'
    joints[name]['parent_node'] = 2
    joints[name]['child_name'] = 'fuse'
    joints[name]['child_node'] = 2
    

    sim = python_csdl_backend.Simulator(Run(beams=beams))

    
    # cs params
    name = 'wing'
    sim[name+'h'] = 0.25
    sim[name+'w'] = 1
    sim[name+'t_left'] = 0.03
    sim[name+'t_top'] = 0.03
    sim[name+'t_right'] = 0.03
    sim[name+'t_bot'] = 0.03

    theta_0 = np.zeros((3,beams['wing']['n']))
    sim[name+'theta_0'] = theta_0
    r_0 = np.zeros((3,beams['wing']['n']))
    r_0[1,:] = np.array([-3,-2,-1,-1,0,1,2,3])
    sim[name+'r_0'] = r_0


    f = np.zeros((3,beams['wing']['n']))
    f[2,:] = 0
    sim[name+'f'] = f
    
    
    name = 'fuse'
    #sim[name+'h'] = 0.25
    #sim[name+'w'] = 1
    #sim[name+'t_left'] = 0.03
    #sim[name+'t_top'] = 0.03
    #sim[name+'t_right'] = 0.03
    #sim[name+'t_bot'] = 0.03
    sim[name+'OD'] = 0.5
    sim[name+'t'] = 0.03

    theta_0 = np.zeros((3,beams['fuse']['n']))
    theta_0[2,:] = np.ones(beams['fuse']['n'])*-np.pi/2
    sim[name+'theta_0'] = theta_0
    r_0 = np.zeros((3,beams['fuse']['n']))
    r_0[0,:] = np.array([-2,-1,0,0,1,2,3,4]) # x coord
    r_0[1,:] = np.array([-1,-1,-1,-1,-1,-1,-1,-1]) # y coord
    sim[name+'r_0'] = r_0

    f = np.zeros((3,beams['wing']['n']))
    f[2,:] = -100000
    sim[name+'f'] = f
    



    sim.run()


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = sim['x'][0,:]
    y = sim['x'][1,:]
    z = sim['x'][2,:]
    phi = sim['x'][3,:]
    theta = sim['x'][4,:]
    psi = sim['x'][5,:]

    np.set_printoptions(linewidth=200)
    print('x: ',np.round(x,4))
    print('y: ',np.round(y,4))
    print('z: ',np.round(z,4))
    print('phi: ',np.round(phi,4))
    print('theta: ',np.round(theta,4))
    print('psi: ',np.round(psi,4))

    ax.scatter(x,y,z)

    
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(-3,4)
    ax.set_ylim(-3,3)
    ax.set_zlim(-0.05,0.05)
    plt.show()