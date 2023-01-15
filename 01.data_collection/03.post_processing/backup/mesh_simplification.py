import pymeshlab
import sys
#import pymesh2
import mayavi

#6810
#13311


obj_file_name=sys.argv[1]
print("obj_file_name="+obj_file_name)
TARGET_Faces=2000
         
        
meshSet = pymeshlab.MeshSet()
meshSet.load_new_mesh(obj_file_name)
pymeshlabMesh = meshSet.current_mesh()
print('input mesh has ', pymeshlabMesh.vertex_number(), ' vertex and ', pymeshlabMesh.face_number(), ' faces')

if pymeshlabMesh.face_number() > TARGET_Faces :
   meshSet.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=TARGET_Faces, preservenormal=True)
   print("Decimated to ", TARGET_Faces, " faces mesh has ", meshSet.current_mesh().vertex_number(), " vertex")
   pymeshlabMesh= meshSet.current_mesh()
   print('diminished output mesh has ', pymeshlabMesh.vertex_number(), ' vertex and ', pymeshlabMesh.face_number(), ' faces')
   meshSet.save_current_mesh(obj_file_name+'.normalized_2000_to_faces.obj')
   verticeMatrix=pymeshlabMesh.vertex_matrix()
else :
   mesh=pymesh2.meshio.load_mesh(obj_file_name, drop_zero_dim=False)
   mesh=pymesh2.subdivide(mesh, order=2, method="loop")
   print('augmented output mesh has ', mesh.num_vertices(), ' vertex and ', mesh.num_faces(), ' faces')
   pymesh2.save_mesh(obj_file_name+'.normalized_2000_to_faces.obj', mesh, ascii=True);
   verticeMatrix=mesh.vertices()

x=verticeMatrix[:,0]
y=verticeMatrix[:,1]
z=verticeMatrix[:,2]
mayavi.mlab.mesh(x, y, z)
   
