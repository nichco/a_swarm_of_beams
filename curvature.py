import csdl
import python_csdl_backend
import numpy as np


class CalcNodalK(csdl.Model):
    def initialize(self):
        self.parameters.declare('options')
    def define(self):
        options = self.parameters['options']
        n = options['n']
        name = options['name']
        beam_type = options['beam_type']

        x = self.declare_variable(name+'x',shape=(12,n))
        theta = x[3:6,:]

        one = self.declare_variable('one',val=1)

        K = self.create_output(name+'K',shape=(3,3,n),val=0)
        Ka = self.create_output(name+'Ka',shape=(3,3,n-1),val=0)

        for i in range(0, n):
            if beam_type == 'wing':
                K[0,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,0,i] = csdl.expand(-csdl.sin(theta[2,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[2,0,i] = csdl.expand(csdl.cos(theta[2,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')

            if beam_type == 'fuse':

                K[0,0,i] = csdl.expand(csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')
                K[0,2,i] = csdl.expand(-csdl.cos(theta[0,i])*csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[1,1,i] = csdl.expand(one,(1,1,1))
                K[1,2,i] = csdl.expand(csdl.sin(theta[0,i]),(1,1,1),'ij->ijk')
                K[2,0,i] = csdl.expand(csdl.sin(theta[1,i]),(1,1,1),'ij->ijk')
                K[2,2,i] = csdl.expand(csdl.cos(theta[0,i])*csdl.cos(theta[1,i]),(1,1,1),'ij->ijk')


        # compute Ka from K
        for i in range(n-1):
            Ka[:,:,i] = 0.5*(K[:,:,i+1] + K[:,:,i])


if __name__ == '__main__':

    beams = {}

    # wing beam
    name = 'wing'
    beams[name] = {}
    beams[name]['n'] = 6
    beams[name]['name'] = name
    beams[name]['beam_type'] = 'fuse'
    beams[name]['free'] = np.array([0,5])
    beams[name]['fixed'] = np.array([2])
    beams[name]['E'] = 69E9
    beams[name]['G'] = 1E20
    beams[name]['rho'] = 2700
    beams[name]['dir'] = 1


    x = np.zeros((12,beams[name]['n']))
    x[5,:] = np.ones(beams[name]['n'])*-np.pi/2

    sim = python_csdl_backend.Simulator(CalcNodalK(options=beams[name]))

    sim[name+'x'] = x


    sim.run()

    print(np.round(sim[name+'K'],2))

    #sim.check_partials(compact_print=True)