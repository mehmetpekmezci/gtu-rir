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



def ray_cast_(bmesh_object,scene,depsgraph,origin,directions,position_name,max_distance_in_a_room):

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
        print("alfa:",alfa,"beta:",beta,"loc:",loc," norm:",norm)
        distance=np.linalg.norm(np.array(origin)-np.array(loc))
        print("distance:",distance)
        gray_scale_color=min(int(63+192*distance/max_distance_in_a_room),255)
        print("gray_scale_color:",gray_scale_color)
        ray_casting_image[alfa][beta]=gray_scale_color
        #print(f"HIT: location={np.array(loc).shape} normal_vector_of_hit_point={np.array(norm).shape} idx={np.array(idx).shape} obj={np.array(obj).shape} mw={np.array(mw).shape}")
#        print(f"{alfa},{beta}={gray_scale_color}")
    else:
        ray_casting_image[alfa][beta]=0
#        print(f"{alfa},{beta}=0")

 return ray_casting_image


def ray_cast(bmesh_object,scene,depsgraph,origin,ray_directions,position_name,max_distance_in_a_room):
       origin=list(np.array(origin).astype(np.float32))
       print("origin:",origin,"max_distance_in_a_room:",max_distance_in_a_room)
       ray_casting_image=np.zeros((len(ray_directions.keys()),len(ray_directions[0].keys())))
       for alfa in ray_directions:
        for beta in ray_directions[alfa]:
          direction=ray_directions[alfa][beta]
          hit, loc, normal, idx, obj, mw = scene.ray_cast(depsgraph,origin, direction)
          if hit:
              print("alfa:",alfa,"beta:",beta,"loc:",loc," normal:",normal)
              distance=np.linalg.norm(np.array(origin)-np.array(loc))
              print("distance:",distance)
              gray_scale_color=min(int(63+192*distance/max_distance_in_a_room),255)
              print("gray_scale_color:",gray_scale_color)
              ray_casting_image[alfa][beta]=gray_scale_color
          else:
              ray_casting_image[alfa][beta]=0
       #ray_casting_image=ray_casting_image.reshape(cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION,1).repeat(3,axis=2)
       #ray_casting_image=Image.fromarray(np.uint8(ray_casting_image), mode="RGB")
       #ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32)

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
