import numpy as np



def test():


    name = 'fuse'


    child, rj = [], []
    for joint_name in joints:
        jx = np.ones((12,1))
        r_j, theta_j = jx[0:3,0], jx[3:6,0]
        if joints[joint_name]['child_name'] == name: # if the beam has any child nodes
            child_node = joints[joint_name]['child_node']
            child.append(child_node) # add the child node to the list
            rj.append(r_j)

    for i in range(0,10):
        if i in child:
            for c, r in zip(child, rj):
                if c == i: r_j = r
            
            print(r_j)

    return child



beams, joints = {}, {}

# wing beam
name = 'wing'
beams[name] = {}
beams[name]['n'] = 8
beams[name]['name'] = name
beams[name]['beam_type'] = 'wing'
beams[name]['free'] = np.array([0,7])
beams[name]['fixed'] = np.array([3])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['dir'] = 1

# fuselage beam
name = 'fuse'
beams[name] = {}
beams[name]['n'] = 8
beams[name]['name'] = name
beams[name]['beam_type'] = 'fuse'
beams[name]['free'] = np.array([0,7])
beams[name]['fixed'] = np.array([2])
beams[name]['E'] = 69E9
beams[name]['G'] = 1E20
beams[name]['rho'] = 2700
beams[name]['dir'] = 1

    
# joint
name = 'wingfuse'
joints[name] = {}
joints[name]['name'] = name
joints[name]['parent_name'] = 'wing'
joints[name]['parent_node'] = 3
joints[name]['child_name'] = 'fuse'
joints[name]['child_node'] = 2


child = test()
print(child)