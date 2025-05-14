import bpy
import bmesh
from mathutils import Vector
import numpy as np

def clear_scene():
 for obj in bpy.context.scene.objects:
  if obj.type == 'MESH':
     obj.select_set(True)
  else:
     obj.select_set(False)
 bpy.ops.object.delete()
 for block in bpy.data.meshes:
    if block.users == 0:
        bpy.data.meshes.remove(block)

 for block in bpy.data.materials:
    if block.users == 0:
        bpy.data.materials.remove(block)

 for block in bpy.data.textures:
    if block.users == 0:
        bpy.data.textures.remove(block)

 for block in bpy.data.images:
    if block.users == 0:
        bpy.data.images.remove(block)



def ray_cast(obj_file_path,origin,directions,position_name,max_distance_in_a_room):
 clear_scene()

 context = bpy.context
 #vl = context.view_layer
 scene = context.scene
 context_object = context.object
 depsgraph=context.evaluated_depsgraph_get()
# me = context_object.data
 #bmesh_object=bpy.ops.import_scene.obj(filepath=obj_file_path)
 bmesh_object=bpy.ops.wm.obj_import(filepath=obj_file_path)
 #print(obj_file_path)
 #print(bmesh_object)
 #bpy.ops.wm.obj_export(filepath=obj_file_path+'_'+origin[0]+'_'+origin[1]+'_'+origin[2]+'_same.obj')
 #print(obj_file_path+'_'+origin[0]+'_'+origin[1]+'_'+origin[2]+'_same.obj')
 #clear_scene()
 #bpy.ops.wm.obj_export(filepath=obj_file_path+'_'+origin[0]+'_'+origin[1]+'_'+origin[2]+'_clean.obj')
 #print(obj_file_path+'_'+origin[0]+'_'+origin[1]+'_'+origin[2]+'_clean.obj')

 ##hit, loc, norm, idx, obj, mw = scene.ray_cast(vl, o, d)

 origin=list(np.array(origin).astype(np.float32))

 ray_casting_image=np.zeros((len(directions.keys()),len(directions[0].keys())))

 for alfa in directions:
  for beta in directions[alfa]:
    direction=directions[alfa][beta]
    hit, loc, norm, idx, obj, mw = scene.ray_cast(depsgraph,origin, direction)
  
    if hit:
        distance=np.linalg.norm(np.array(origin)-np.array(loc))
        gray_scale_color=min(int(63+192*distance/max_distance_in_a_room),255)
        ray_casting_image[alfa][beta]=gray_scale_color
        #print(f"HIT: location={np.array(loc).shape} normal_vector_of_hit_point={np.array(norm).shape} idx={np.array(idx).shape} obj={np.array(obj).shape} mw={np.array(mw).shape}")
#        print(f"{alfa},{beta}={gray_scale_color}")
    else:
        ray_casting_image[alfa][beta]=0
#        print(f"{alfa},{beta}=0")

 return ray_casting_image


# if hit:
#    result = {}
#    loc = mw.inverted() @ loc
#    bmesh_object.from_mesh(me)
#    bmesh_object.faces.ensure_lookup_table()
#    f = bmesh_object.faces[idx]
#    result[f] = (loc - f.calc_center_median()).length
#    for e in f.edges:
#        mp = (e.verts[1].co + e.verts[0].co) / 2
#        result[e] = (loc - mp).length
#    for v in f.verts:
#        result[v] = (v.co - loc).length
#    print("-" * 33)
#    for e, d in sorted(result.items(), key=lambda e: e[1]):
#        print(type(e).__name__, e.index, d)
#        if hasattr(e, "link_faces"):
#            print([f.index for f in e.link_faces])
