import bpy
import bmesh
from mathutils import Vector

context = bpy.context
vl = context.view_layer

scene = context.scene

ob = context.object
me = ob.data
bm = bmesh.new()

o = (10, 10, 10) # ray origin
d = (-1, -1, -1) # ray direction

hit, loc, norm, idx, obj, mw = scene.ray_cast(vl, o, d)

if hit:
    result = {}
    loc = mw.inverted() @ loc
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    f = bm.faces[idx]
    result[f] = (loc - f.calc_center_median()).length
    for e in f.edges:
        mp = (e.verts[1].co + e.verts[0].co) / 2
        result[e] = (loc - mp).length
    for v in f.verts:
        result[v] = (v.co - loc).length
    print("-" * 33)
    for e, d in sorted(result.items(), key=lambda e: e[1]):
        print(type(e).__name__, e.index, d)
        if hasattr(e, "link_faces"):
            print([f.index for f in e.link_faces])

