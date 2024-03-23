import pymeshlab as ml
import os
import sys

if len(sys.argv) < 2 :
   print("Usage : python3 room_dimension_extractor.py <SOURCE_OBJ_FILE> <TARGET_ROOM_DIM_FILE>")
   exit(1)
else:
            f_path=sys.argv[1]
            room_dim_path=sys.argv[2]
            ms = ml.MeshSet()
            ms.load_new_mesh(f_path)
            boundingbox =  ms.current_mesh().bounding_box()
            DIM_X=boundingbox.dim_x()
            DIM_Y=boundingbox.dim_y()
            DIM_Z=boundingbox.dim_z()
            with open(room_dim_path, "w") as file1:
                #file1.write(f"DIM_X={DIM_X}\nDIM_Y={DIM_Y}\nDIM_Z={DIM_Z}")
                file1.write(f"DIM_X={DIM_X}\nDIM_Y={DIM_Z}\nDIM_Z={DIM_Y}")
