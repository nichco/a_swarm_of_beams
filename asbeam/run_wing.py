import csdl
import python_csdl_backend
import numpy as np





















beams = {}
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