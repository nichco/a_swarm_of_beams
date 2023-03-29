import math
import numpy as np

def angle_of_curve(coords):
    # Step 1: Compute direction vectors
    direction_vectors = []
    for i in range(len(coords)-1):
        diff = np.array(coords[i+1]) - np.array(coords[i])
        direction_vectors.append(diff)
    
    # Step 2: Compute normal vectors
    normal_vectors = []
    for i in range(len(direction_vectors)-1):
        u = direction_vectors[i]
        v = direction_vectors[i+1]
        normal = np.cross(u, v)
        normal_vectors.append(normal)
    
    # Step 3: Compute angles between normal vectors
    angles = []
    for i in range(len(normal_vectors)-1):
        u = normal_vectors[i]
        v = normal_vectors[i+1]
        dot_product = np.dot(u, v)
        norm_product = np.linalg.norm(u) * np.linalg.norm(v)
        angle = math.acos(dot_product / norm_product)
        angles.append(angle)
    
    # Step 4: Initialize empty angle vector
    angle_vector = np.empty((3, len(coords)))
    
    # Step 5: Convert normal vectors to (phi,theta,psi) format and append to angle vector
    for i in range(len(normal_vectors)):
        phi = math.atan2(normal_vectors[i][1], normal_vectors[i][0])
        theta = math.acos(normal_vectors[i][2] / np.linalg.norm(normal_vectors[i]))
        psi = math.atan2(normal_vectors[i+1][1], normal_vectors[i+1][0])
        angle_vector[:,i+1] = [phi, theta, psi]
    
    return angle_vector





coords = [(0,0,0), (1,1,1), (2,1,0), (3,0,-1), (4,-1,0), (5,-1,1), (6,0,0)]
angle_vector = angle_of_curve(coords)
print(angle_vector)
