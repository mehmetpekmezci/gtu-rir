import trimesh

m=trimesh.load_mesh('mesh.obj')
Y=11.27/2
X=5.42/2
m.apply_transform(trimesh.transformations.translation_matrix([-X,0,-Y]))
m.export('mesh1.obj')
