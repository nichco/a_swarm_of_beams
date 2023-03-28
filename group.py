import csdl
import python_csdl_backend
import numpy as np
from implicitop import ImplicitOp


class Group(csdl.Model):
    def initialize(self):
        self.parameters.declare('beams')
        self.parameters.declare('joints')
    def define(self):
        beams = self.parameters['beams']
        joints = self.parameters['joints']



        for joint_name in joints:
            pass



        for beam_name in beams:
            self.add(ImplicitOp(options=beams[beam_name]), name=beam_name+'ImplicitOp')