import numpy as np
import csdl
from asbeam.asbeam import Asbeam
import python_csdl_backend

# create a mesh:
n = 11
theta_0 = np.zeros((3,n))
r_0 = np.zeros((3,n))
r_0[1,:] = np.linspace(0,10,n)

class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')

    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        self.create_input('wingtheta_0', shape=(3,n), val=theta_0)
        self.create_input('wingr_0', shape=(3,n), val=r_0)
        self.create_input('wingOD', shape=(n), val=0.5)
        self.create_input('wingt', shape=(n), val=0.01)

        self.add(Asbeam(beams=beams, joints=joints), name='Asbeam')



if __name__ == '__main__':

    joints, beams = {}, {}
    beams['wing'] = {'E': 69E9,
                     'G': 26E9,
                     'rho': 2700,
                     'cs': 'tube',
                     'nodes': list(range(n)),
                     'type': 'wing',
                     'free': [10],
                     'fixed': [0]}
    
    sim = python_csdl_backend.Simulator(Run(beams=beams, joints=joints))
    sim.run()




    # plotting:
    import matplotlib.pyplot as plt
    plt.rcParams.update(plt.rcParamsDefault)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x = sim['x'][0,:]
    y = sim['x'][1,:]
    z = sim['x'][2,:]

    ax.scatter(x,y,z)

    ax.set_xlim(-1,1)
    ax.set_ylim(0,10)
    ax.set_zlim(-1,1)

    plt.show()