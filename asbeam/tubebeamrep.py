import csdl
import numpy as np
import python_csdl_backend



class TubeBeamRep(csdl.Model):
    def initialize(self):
        self.parameters.declare('name')
        self.parameters.declare('options')

    def define(self):
        name = self.parameters['name']
        options = self.parameters['options']

        n = len(options['nodes'])
        E = options['E']
        G = options['G']
        rho = options['rho']

        OD = self.declare_variable(name+'OD', shape=(n), val=0.5) # outer diameter
        t = self.declare_variable(name+'t', shape=(n), val=0.01) # thickness

        id = OD - 2*t # inner diameter

        Ixx = Izz = np.pi*(OD**4 - id**4)/64

        Ixz = np.pi*(OD**4 - id**4)/90
        
        J = np.pi*(OD**4 - id**4)/32
        area = np.pi*(OD**2 - id**2)/4 # area

        EIxx, EIxz, EIzz, GJ, EA = E*Ixx, E*Ixz, E*Izz, G*J, E*area

        E_inv = self.create_output(name+'E_inv',shape=(3,3,n),val=0)
        for i in range(n):
            denom_i = (EIxx[i]*EIzz[i] - EIxz[i]**2)
            E_inv[0,0,i] = csdl.expand(EIzz[i]/denom_i, (1,1,1))
            E_inv[0,2,i] = csdl.expand(-EIxz[i]/denom_i, (1,1,1))
            E_inv[1,1,i] = csdl.expand(1/(GJ[i]), (1,1,1))
            E_inv[2,2,i] = csdl.expand(EIxx[i]/denom_i, (1,1,1))
            E_inv[2,0,i] = csdl.expand(-EIxz[i]/denom_i, (1,1,1))


        GKn = self.create_input(name+'GKn',shape=(n),val=G / 1.2 * np.ones(n))
        GKc = self.create_input(name+'GKc',shape=(n),val=G / 1.2 * np.ones(n))

        oneover = self.create_output(name+'oneover',shape=(3,3,n),val=0)
        for i in range(n):
            oneover[0,0,i] = csdl.expand(1 / GKc[i], (1,1,1))
            oneover[1,1,i] = csdl.expand(1 / EA[i], (1,1,1))
            oneover[2,2,i] = csdl.expand(1 / GKn[i], (1,1,1))


        zero = self.declare_variable('zero',val=0)
        n_ea = csdl.expand(zero, (n))
        c_ea = csdl.expand(zero, (n))
        n_ta = csdl.expand(zero, (n))
        c_ta = csdl.expand(zero, (n))

        D = np.zeros((3,3,n))
        D = self.create_output(name+'D',shape=(3,3,n),val=0)
        for i in range(n):
            D[0,1,i] = csdl.expand(-n_ea[i], (1,1,1))
            D[1,0,i] = csdl.expand(n_ta[0], (1,1,1))
            D[1,2,i] = csdl.expand(-c_ta[0], (1,1,1))
            D[2,1,i] = csdl.expand(c_ea[0], (1,1,1))



if __name__ == '__main__':

    options = {}
    options['n'] = 2
    options['name'] = 'wing'
    options['free'] = np.array([options['n']-1]) # (tip)
    options['fixed'] = fixed = np.array([0]) # (root)
    options['E'] = 69E9
    options['G'] = 1E20
    options['rho'] = 2700

    sim = python_csdl_backend.Simulator(TubeBeamRep(options=options))
    sim.run()

    print(sim[options['name']+'E_inv'])