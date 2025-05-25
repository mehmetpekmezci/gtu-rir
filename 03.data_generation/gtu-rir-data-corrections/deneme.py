import trimesh
import numpy as np

m=trimesh.load_mesh('mesh.obj')
Y=np.max(m.vertices[:,2])/2
X=np.max(m.vertices[:,0])/2
m.apply_transform(trimesh.transformations.translation_matrix([-X,0,-Y]))
m.export('mesh1.obj')
