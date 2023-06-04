import pymeshlab as ml
import os
import sys
import trimesh
#6810
#13311

#path ="Meshes"
#move_path = "Simplified_Meshes"

#os.mkdir(move_path)

# path ="/scratch/anton/template"
# move_path = "/scratch/anton/template_60000"

#for subdir,dir,files in os.walk(path):
#    for file in files:

if len(sys.argv) < 2 :
   print("Usage : python3 mesh_simplification.py <SOURCE_OBJ_FILE> <TARGET_OBJ_FILE>")
   exit(1)
else:
#        if(file.endswith(".obj")):
            
            f_path=sys.argv[1]
            m_path=sys.argv[2]
            TARGET_Faces=2000
         
        
            ms = ml.MeshSet()
            ms.load_new_mesh(f_path)
            m = ms.current_mesh()
            print('input mesh has ', m.vertex_number(), ' vertex and ', m.face_number(), ' faces')


            if m.face_number() < TARGET_Faces-100 :
                 #trimesh.remesh.subdivide_to_size
                mesh = trimesh.load(f_path)
                #mesh.vertices, mesh.faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces,TARGET_Faces)
                while mesh.faces.shape[0] < TARGET_Faces:
                      mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
                print('--output mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')
                mesh.export(f_path)

                # reload mesh file from pymeshlab
                ms = ml.MeshSet()
                ms.load_new_mesh(f_path)
                m = ms.current_mesh()         

            ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=TARGET_Faces, preservenormal=True)
            print("Decimated to ", TARGET_Faces, " faces mesh has ", ms.current_mesh().vertex_number(), " vertex")
           
            m = ms.current_mesh()
            print('output mesh has ', m.vertex_number(), ' vertex and ', m.face_number(), ' faces')
            ms.save_current_mesh(m_path)
