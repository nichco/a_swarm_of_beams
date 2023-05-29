import numpy as np
import csdl
from asbeam.asbeam import Asbeam




class Run(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('bounds')
        self.parameters.declare('joints')

    def define(self):
        beams = self.parameters['beams']
        bounds = self.parameters['bounds']
        joints = self.parameters['joints']



        self.add(Asbeam(beams=beams, bounds=bounds, joints=joints), name='Asbeam')























if __name__ == '__main__':

    n = 11

    joints, bounds, beams = {}, {}, {}
    beams['wing'] = {'E': 69E9,'G': 26E9,'rho': 2700,'cs': 'tube','nodes': list(range(n))}
    bounds['root'] = {'beam': 'wing','node': 5,'fdim': [1,1,1,1,1,1]}