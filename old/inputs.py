import numpy as np

"""
a sample inputs file for an entire aircraft structural model
"""



# beam definitions
beams, joints = {}, {}

# wing beam
name = 'wing'
beams[name] = {}
beams[name]['n'] = 20
beams[name]['name'] = name
beams[name]['beam_type'] = 'wing'
beams[name]['free'] = np.array([0,19]) # wing tips are free
beams[name]['fixed'] = np.array([10])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['shape'] = 'box'

wing_r_0 = np.zeros((3,beams[name]['n']))
wing_r_0[1,:] = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,0,0,1,2,3,4,5,6,7,8,9])*2
wing_theta_0 = np.zeros((3,beams[name]['n']))

wing_fa = np.zeros((3,beams[name]['n']))
wing_fa[2,:] = 1000


# fuselage beam
name = 'fuse'
beams[name] = {}
beams[name]['n'] = 7
beams[name]['name'] = name
beams[name]['beam_type'] = 'fuse'
beams[name]['free'] = np.array([0,6])
beams[name]['fixed'] = np.array([])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['shape'] = 'box'

fuse_r_0 = np.zeros((3,beams[name]['n']))
fuse_r_0[0,:] = np.array([-3,-2,-1,0,0,1,2])
fuse_theta_0 = np.zeros((3,beams[name]['n']))
fuse_theta_0[2,:] = -np.pi/2

fuse_fa = np.zeros((3,beams[name]['n']))
fuse_fa[2,:] = -1000


# left boom
name = 'lboom'
beams[name] = {}
beams[name]['n'] = 8
beams[name]['name'] = name
beams[name]['beam_type'] = 'fuse'
beams[name]['free'] = np.array([0,7])
beams[name]['fixed'] = np.array([])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['shape'] = 'box'

lboom_r_0 = np.zeros((3,beams[name]['n']))
lboom_r_0[0,:] = np.array([-1,0,0,1,2,3,4,5])
lboom_r_0[1,:] = -4
lboom_theta_0 = np.zeros((3,beams[name]['n']))
lboom_theta_0[2,:] = -np.pi/2

lboom_fa = np.zeros((3,beams[name]['n']))
lboom_fa[2,:] = -2000


# right boom
name = 'rboom'
beams[name] = {}
beams[name]['n'] = 8
beams[name]['name'] = name
beams[name]['beam_type'] = 'fuse'
beams[name]['free'] = np.array([0,7])
beams[name]['fixed'] = np.array([])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['shape'] = 'box'

rboom_r_0 = np.zeros((3,beams[name]['n']))
rboom_r_0[0,:] = np.array([-1,0,0,1,2,3,4,5])
rboom_r_0[1,:] = 4
rboom_theta_0 = np.zeros((3,beams[name]['n']))
rboom_theta_0[2,:] = -np.pi/2

rboom_fa = np.zeros((3,beams[name]['n']))
rboom_fa[2,:] = -2000


# tail
name = 'tail'
beams[name] = {}
beams[name]['n'] = 7
beams[name]['name'] = name
beams[name]['beam_type'] = 'wing'
beams[name]['free'] = np.array([0,6])
beams[name]['fixed'] = np.array([])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['shape'] = 'box'

tail_r_0 = np.zeros((3,beams[name]['n']))
tail_r_0[0,:] = 5
tail_r_0[1,:] = np.array([-4,-3.5,-2,0,2,3.5,4])
tail_r_0[2,:] = np.array([0,0.5,0.5,0.5,0.5,0.5,0])
tail_theta_0 = np.zeros((3,beams[name]['n']))
tail_theta_0[0,:] = np.array([np.pi/4,0,0,0,0,0,-np.pi/4])




# joint definitions

name = 'wingfuse'
joints[name] = {}
joints[name]['name'] = name
joints[name]['parent_name'] = 'wing'
joints[name]['parent_node'] = 9
joints[name]['child_name'] = 'fuse'
joints[name]['child_node'] = 3


name = 'winglboom'
joints[name] = {}
joints[name]['name'] = name
joints[name]['parent_name'] = 'wing'
joints[name]['parent_node'] = 7
joints[name]['child_name'] = 'lboom'
joints[name]['child_node'] = 1


name = 'wingrboom'
joints[name] = {}
joints[name]['name'] = name
joints[name]['parent_name'] = 'wing'
joints[name]['parent_node'] = 12
joints[name]['child_name'] = 'rboom'
joints[name]['child_node'] = 1


name = 'lboomtail'
joints[name] = {}
joints[name]['name'] = name
joints[name]['parent_name'] = 'lboom'
joints[name]['parent_node'] = 7
joints[name]['child_name'] = 'tail'
joints[name]['child_node'] = 0