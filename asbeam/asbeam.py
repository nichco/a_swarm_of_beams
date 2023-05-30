import numpy as np
import csdl
from asbeam.beamresidual import BeamRes
from asbeam.boxbeamrep import BoxBeamRep
from asbeam.tubebeamrep import TubeBeamRep
from asbeam.jointresidual import JointRes


class Asbeam(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams', default={})
        self.parameters.declare('joints', default={})

    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        # get the beam cross-sectional properties at each node:
        for beam_name in beams:
            if beams[beam_name]['cs'] == 'box':
                self.add(BoxBeamRep(name=beam_name, options=beams[beam_name]), name=beam_name + 'BoxBeamRep')
            elif beams[beam_name]['cs'] == 'tube':
                self.add(TubeBeamRep(name=beam_name, options=beams[beam_name]), name=beam_name + 'TubeBeamRep')
        

        # compute the total number of nodes for the entire beam group:
        num_nodes = sum(len(beams[beam_name]['nodes']) for beam_name in beams)


        # concatenate the variables as inputs to the implicit operation
        r_0 = self.create_output('r_0',shape=(3,num_nodes))
        theta_0 = self.create_output('theta_0',shape=(3,num_nodes))
        E_inv = self.create_output('E_inv',shape=(3,3,num_nodes))
        D = self.create_output('D',shape=(3,3,num_nodes))
        oneover = self.create_output('oneover',shape=(3,3,num_nodes))
        f = self.create_output('f',shape=(3,num_nodes))
        m = self.create_output('m',shape=(3,num_nodes))
        fp = self.create_output('fp',shape=(3,num_nodes))
        mp = self.create_output('mp',shape=(3,num_nodes))

        i = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            r_0[:,i:i+n] = self.declare_variable(beam_name+'r_0',shape=(3,n),val=0)
            theta_0[:,i:i+n] = self.declare_variable(beam_name+'theta_0',shape=(3,n),val=0)
            E_inv[:,:,i:i+n] = self.declare_variable(beam_name+'E_inv',shape=(3,3,n),val=0)
            D[:,:,i:i+n] = self.declare_variable(beam_name+'D',shape=(3,3,n),val=0)
            oneover[:,:,i:i+n] = self.declare_variable(beam_name+'oneover',shape=(3,3,n),val=0)
            f[:,i:i+n] = self.declare_variable(beam_name+'f',shape=(3,n),val=0)
            m[:,i:i+n] = self.declare_variable(beam_name+'m',shape=(3,n),val=0)
            fp[:,i:i+n] = self.declare_variable(beam_name+'fp',shape=(3,n),val=0)
            mp[:,i:i+n] = self.declare_variable(beam_name+'mp',shape=(3,n),val=0)
            i += n


        # define the implicit operation
        solve_res = self.create_implicit_operation(BeamGroup(beams=beams, joints=joints))
        solve_res.declare_state('x', residual='res')
        solve_res.nonlinear_solver = csdl.NewtonSolver(solve_subsystems=False,maxiter=200,iprint=False,atol=1E-7)
        solve_res.linear_solver = csdl.ScipyKrylov()

        # define the variables to pass to the implicit operation:
        vars = {'r_0': (3,num_nodes),
                'theta_0': (3,num_nodes),
                'E_inv': (3,3,num_nodes),
                'D': (3,3,num_nodes),
                'oneover': (3,3,num_nodes),
                'f': (3,num_nodes), # distributed forces
                'm': (3,num_nodes), # distributed moments
                'fp': (3,num_nodes), # point loads
                'mp': (3,num_nodes)} # point moments
        
        var_list = [self.declare_variable(var_name, shape=var_shape, val=0) for var_name, var_shape in vars.items()]

        solve_res(*var_list)





class BeamGroup(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')

    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']

        # compute the total number of nodes for the entire beam group:
        num_nodes = sum(len(beams[beam_name]['nodes']) for beam_name in beams)
        cols = num_nodes + len(joints)

        # declare the state and the residual:
        x = self.declare_variable('x', shape=(12,cols), val=0)
        res = self.create_output('res', shape=(12,cols), val=0)
        # x = self.declare_variable('x', shape=(12,num_nodes), val=0)
        # res = self.create_output('res', shape=(12,num_nodes), val=0)

        # partition the state
        i = 0
        for beam_name in beams:
            self.register_output(beam_name+'x', x[:, i:i+(n:=len(beams[beam_name]['nodes']))])
            i += n

        # partition the joint state
        for i, joint_name in enumerate(joints): 
            self.register_output(joint_name + 'x', x[:,num_nodes+i])


        # partition the beam properties inputs
        vars = {'r_0': (3,num_nodes),
                'theta_0': (3,num_nodes),
                'E_inv': (3,3,num_nodes),
                'D': (3,3,num_nodes),
                'oneover': (3,3,num_nodes),
                'f': (3,num_nodes),
                'm': (3,num_nodes),
                'fp': (3,num_nodes),
                'mp': (3,num_nodes)}
        

        r_0, theta_0, E_inv, D, oneover, f, m, fp, mp = [self.declare_variable(var_name, shape=var_shape) for var_name, var_shape in vars.items()]

        i = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            self.register_output(beam_name+'r_0', r_0[:,i:i+n])
            self.register_output(beam_name+'theta_0', theta_0[:,i:i+n])
            self.register_output(beam_name+'E_inv', E_inv[:,:,i:i+n])
            self.register_output(beam_name+'D', D[:,:,i:i+n])
            self.register_output(beam_name+'oneover', oneover[:,:,i:i+n])
            self.register_output(beam_name+'f', f[:,i:i+n])
            self.register_output(beam_name+'m', m[:,i:i+n])
            self.register_output(beam_name+'fp', fp[:,i:i+n])
            self.register_output(beam_name+'mp', mp[:,i:i+n])
            i += n


        # get the beam residuals:
        i = 0
        for beam_name in beams:
            n = len(beams[beam_name]['nodes'])
            self.add(BeamRes(name=beam_name, options=beams[beam_name], joints=joints), name=beam_name+'BeamRes')
            res[:, i:i+n] = self.declare_variable(beam_name+'res', shape=(12,n), val=0) + 0*x[:,i:i+n]
            i += n


        # get the joint residuals:
        for i, joint_name in enumerate(joints):
           self.add(JointRes(name=joint_name, beams=beams, joint=joints[joint_name]), name=joint_name+'JointRes')
           res[:, num_nodes+i] = csdl.expand(self.declare_variable(joint_name+'res', shape=(12)), (12,1), 'i->ij')

