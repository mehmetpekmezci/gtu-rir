import pymeshlab as ml
import os
import trimesh
import sys
import numpy as np
#6810
#13311


main_dir=str(sys.argv[1]).strip()
metadata_dirname=str(sys.argv[2]).strip()
metadata_dir = main_dir+"/"+metadata_dirname

path = metadata_dir+"/Meshes"
move_path = metadata_dir+"/Simplified_Meshes"

if not os.path.exists(move_path):
   os.mkdir(move_path)

# path ="/scratch/anton/template"
# move_path = "/scratch/anton/template_60000"

for subdir,dir,files in os.walk(path):
    for file in files:

        if(file.endswith(".obj")):
            
            f_path=os.path.join(subdir,file)
            m_path=os.path.join(move_path,file)
            TARGET_Faces=2000
         
        
            ms = ml.MeshSet()
            ms.load_new_mesh(f_path)
            m = ms.current_mesh()
            print('input mesh has ', m.vertex_number(), ' vertex and ', m.face_number(), ' faces')


            ## MP :  AUGMENT NUMBER OF Faces to 'TARGET_Faces' value which is 2000.
            if m.face_number() < TARGET_Faces-100 :
                 #trimesh.remesh.subdivide_to_size
                mesh = trimesh.load(f_path)
                #mesh.vertices, mesh.faces = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces,TARGET_Faces)
                while mesh.faces.shape[0] < TARGET_Faces:
                      mesh.vertices, mesh.faces = trimesh.remesh.subdivide(mesh.vertices, mesh.faces)
                print('--output mesh has ', mesh.vertices.shape[0], ' vertex and ', mesh.faces.shape[0], ' faces')
                #if np.min(mesh.vertices[:,2]) >= 0 :
                print(f"f_path={f_path}, GTURIR is in f_path = {'GTURIR' in f_path}")
                if "GTURIR" in f_path:
                   print("in GTURIR case")
                   Y=np.max(mesh.vertices[:,2])/2
                   print(f"YYYY={Y}")
                   X=np.max(mesh.vertices[:,0])/2
                   #X=0
                   print(f"XXXX={X}")
                   mesh.apply_transform(trimesh.transformations.translation_matrix([-X,0,-Y]))

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
