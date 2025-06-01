import time
import torch.utils.data as data
import os
import numpy as np
import torch
import librosa
import sys
import bpy
import bmesh
from miscc.config import cfg
from PIL import Image
from contextlib import redirect_stdout
import io
import pickle

import io
from contextlib import contextmanager
import trimesh


class RIRDataset(data.Dataset):
    def __init__(self,data_dir,embeddings,split='train',rirsize=4096): 
        self.rirsize = rirsize
        self.data = []
        self.data_dir = data_dir       
        self.embeddings = embeddings
        self.ray_directions=self.generate_ray_directions() 
        self.bpy_context = bpy.context
        self.bpy_scene = self.bpy_context.scene
        self.bpy_depsgraph=self.bpy_context.evaluated_depsgraph_get()
        self.stdout = io.StringIO()
        bpy.context.scene.cycles.device = 'GPU'
    def get_RIR(self, full_RIR_path):

        picklePath=full_RIR_path.replace(".wav",".pickle")
        if os.path.exists(picklePath):
            with open(picklePath, "rb") as f:
                  x = pickle.load(f)
            return x

        wav,fs = librosa.load(full_RIR_path)
 
        # wav_resample = librosa.resample(wav,16000,fs)
        wav_resample = librosa.resample(wav,orig_sr=fs,target_sr=16000)

        length = wav_resample.size

        crop_length = 3968 #int(16384)
        if(length<crop_length):
            zeros = np.zeros(crop_length-length)
            std_value = np.std(wav_resample) * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new = np.concatenate([wav_resample,zeros])/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])
        else:
            wav_resample_new = wav_resample[0:crop_length]
            std_value = np.std(wav_resample_new)  * 10
            std_array = np.repeat(std_value,128)
            wav_resample_new =wav_resample_new/std_value
            RIR_original = np.concatenate([wav_resample_new,std_array])

        resample_length = int(self.rirsize)
        
        RIR = RIR_original

        RIR = np.array([RIR]).astype('float32')

        with open(picklePath, 'wb') as f:
            pickle.dump(RIR, f, protocol=2)

        return RIR

    def __getitem__(self, index):

        graph_path,RIR_path,source_location,receiver_location= self.embeddings[index]
#        s=(np.array(source_location).astype(np.float32))
#        r=(np.array(receiver_location).astype(np.float32))
#
#        s12=[] ## sadece bu olabilir, hepsinde calisan durum.
#        s12.append(s[0])
#        s12.append(-s[2])
#        s12.append(s[1])
#
#        r12=[]
#        r12.append(r[0])
#        r12.append(-r[2])
#        r12.append(r[1])
#
#        source_location=s12
#        receiver_location=r12

        data = {}
        data["RIR"] =  self.get_RIR(os.path.join(self.data_dir,RIR_path))
        data["source_and_receiver"] =  np.concatenate((np.array(source_location).astype('float32'),np.array(receiver_location).astype('float32')))
        data["mesh_embeddings_source_image"],data["mesh_embeddings_receiver_image"] = self.mesh_embeddings(os.path.join(self.data_dir,graph_path),source_location,receiver_location)
        return data
        
    def __len__(self):
        return len(self.embeddings)

    @contextmanager
    def stdout_redirected(to=os.devnull):
      to='/tmp/uselessfile'
      ##import os
      ##with stdout_redirected(to=filename):
      ##  print("from Python")
      ##  os.system("echo non-Python applications are also supported")
      fd = sys.stdout.fileno()

       ##### assert that Python and C stdio write using the same file descriptor
       ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

      def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

      with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

    def mesh_embeddings(self,full_graph_path,source,receiver):
        self.clear_scene()
        #with redirect_stdout(self.stdout), redirect_stderr(self.stdout):
        try:
          with self.stdout_redirected():
             bpy.ops.wm.obj_import(filepath=full_graph_path)
        except:
             bpy.ops.wm.obj_import(filepath=full_graph_path)

        if "GTURIR" in full_graph_path :



            mesh = trimesh.load_mesh(full_graph_path, process=False)
            Y=np.max(mesh.vertices[:,2])/2
            X=np.max(mesh.vertices[:,0])/2

            bpy.ops.object.select_all(action='SELECT') # DESELECT
            bpy.ops.transform.translate(value=(-X, 0, -Y))

        #self.printMesh()
        #bpy.ops.wm.obj_import(filepath=full_graph_path,forward_axis='Z',up_axis='Y')

#        s=(np.array(source).astype(np.float32))
#        r=(np.array(receiver).astype(np.float32))
#
#        if permute_axis: 
#           s12=[] ## sadece bu olabilir, hepsinde calisan durum.
#           s12.append(s[0])
#           s12.append(-s[2])
#           s12.append(s[1])
#
#           r12=[]
#           r12.append(r[0])
#           r12.append(-r[2])
#           r12.append(r[1])
#        else:
#           s12=s
#           r12=r

        ray_cast_image_source=self.ray_cast(source)
        ray_cast_image_receiver=self.ray_cast(receiver)

        #ray_cast_image_source=self.ray_cast(source)
        #ray_cast_image_receiver=self.ray_cast(receiver)
        

#        s=(np.array(source).astype(np.float32))
#        s1=[] ## bu olmaz
#        s1.append(s[0])
#        s1.append(s[2])
#        s1.append(s[1])
#
#        r=(np.array(receiver).astype(np.float32))
#        r1=[]
#        r1.append(r[0])
#        r1.append(r[2])
#        r1.append(r[1])
#
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 0.2, location = s1)
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 0.8, location = r1)
#
#
#        s11=[] ## bu olmaz
#        s11.append(-s[0])
#        s11.append(s[2])
#        s11.append(s[1])
#
#        r11=[]
#        r11.append(-r[0])
#        r11.append(r[2])
#        r11.append(r[1])
#
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 1.2, location = s11)
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 1.8, location = r11)
#
#
#        s12=[] ## sadece bu olabilir, hepsinde calisan durum.
#        s12.append(s[0])
#        s12.append(-s[2])
#        s12.append(s[1])
#
#        r12=[]
#        r12.append(r[0])
#        r12.append(-r[2])
#        r12.append(r[1])
#
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 2.2, location = s12)
#        bpy.ops.mesh.primitive_uv_sphere_add(radius = 2.8, location = r12)
#
#        s13=[] ## bu olmaz
#        s13.append(-s[0])
#        s13.append(-s[2])
#        s13.append(s[1])
#
#        r13=[]
#        r13.append(-r[0])
#        r13.append(-r[2])
#        r13.append(r[1])
#
#        bpy.ops.mesh.primitive_cube_add(size = 2.2, location = s13)
#        bpy.ops.mesh.primitive_cube_add(size = 2.8, location = r13)
#
#
#
#        s2=[] ## bu olabilir , bu olmaza gibi
#        s2.append(s[2])
#        s2.append(s[0])
#        s2.append(s[1])
#
#        r2=[]
#        r2.append(r[2])
#        r2.append(r[0])
#        r2.append(r[1])
#
#        bpy.ops.mesh.primitive_cube_add(size = 0.2, location = s2)
#        bpy.ops.mesh.primitive_cube_add(size = 0.8, location = r2)
#
#
#        s3=[] ## bu olamaz
#        s3.append(s[2])
#        s3.append(-s[0])
#        s3.append(s[1])
#
#        r3=[]
#        r3.append(r[2])
#        r3.append(-r[0])
#        r3.append(r[1])
#
#        bpy.ops.mesh.primitive_cylinder_add(radius = 0.2, location = s3)
#        bpy.ops.mesh.primitive_cylinder_add(radius = 0.8, location = r3)
#
#
#        s4=[] ## bu olmaz.
#        s4.append(-s[2])
#        s4.append(-s[0])
#        s4.append(s[1])
#
#        r4=[]
#        r4.append(-r[2])
#        r4.append(-r[0])
#        r4.append(r[1])
#
#        bpy.ops.mesh.primitive_cylinder_add(radius = 1.2, location = s4)
#        bpy.ops.mesh.primitive_cylinder_add(radius = 1.8, location = r4)
#
#        s5=[] # bu olmaz
#        s5.append(-s[2])
#        s5.append(s[0])
#        s5.append(s[1])
#
#        r5=[]
#        r5.append(-r[2])
#        r5.append(r[0])
#        r5.append(r[1])
#
#        bpy.ops.mesh.primitive_cube_add(size = 1.2, location = s5)
#        bpy.ops.mesh.primitive_cube_add(size = 1.8, location = r5)
#
#
#        #bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.10.obj',forward_axis='Z',up_axis='Y')
#
#        #bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.10.obj',forward_axis='Z',up_axis='Y')
#        bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.11.obj')
#        print(full_graph_path+'.source.and.receiver.added.10.obj')

### SEKILLERE BAKRAK *.5.obj nin en iyi oldugunu gordum.
#### DOLAYISIYLA BUNU EMBEDDING_GENERTOR.PY dosyasina yansitacagim :)

#        temp=source[1]
#        source[1]=source[2]
#        source[2]=temp
#        ray_cast_image_source=self.ray_cast(source)
#        temp=receiver[1]
#        receiver[1]=receiver[2]
#        receiver[2]=temp
#        ray_cast_image_receiver=self.ray_cast(receiver)
#        bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.4.obj')

#        self.clear_scene()
#        bpy.ops.wm.obj_import(filepath=full_graph_path)
#        source[1]=str(-float(source[1]))
#        ray_cast_image_source=self.ray_cast(source)
#        receiver[1]=str(-float(receiver[1]))
#        ray_cast_image_receiver=self.ray_cast(receiver)
#        bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.5.obj')

#        self.clear_scene()
#        bpy.ops.wm.obj_import(filepath=full_graph_path)
#        source[0]=str(-float(source[0]))
#        ray_cast_image_source=self.ray_cast(source)
#        receiver[0]=str(-float(receiver[0]))
#        ray_cast_image_receiver=self.ray_cast(receiver)
#        bpy.ops.wm.obj_export(filepath=full_graph_path+f'.source.and.receiver.added.6.obj')

#        print(full_graph_path+'.source.and.receiver.added.obj')
        return ray_cast_image_source,ray_cast_image_receiver

    def clear_scene(self):
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

    def printMesh(self):


        ## MP: HEM origin coord. hem de mesh vertice coords : XZY
        ## onun icin bir conversion yapmiyoruz.

        lx = [] # list of objects x locations
        ly = [] # list of objects y locations
        lz = [] # list of objects z locations

        for obj in bpy.data.objects:
          print(obj.name)
          print(obj.location)
          print(obj.dimensions)
          try:
           if obj.data.vertices:
            for v in obj.data.vertices:
                lx.append(v.co[0])
                ly.append(v.co[1])
                lz.append(v.co[2])
            print(f"max(lx)={max(lx)} min(lx)={min(lx)} max(ly)={max(ly)} min(ly)={min(ly)} max(lz)={max(lz)} min(lz)={min(lz)} ")
          # if obj.data.faces:
          #  print(obj.data.faces)
          except:
            print("--")

        #print(f"bpy.context.scene.objects.active.location={bpy.context.scene.objects.active.location}")
        #print(f"bpy.context.scene.objects.active.dimensions={bpy.context.scene.objects.active.dimensions}")
        #current_obj = bpy.context.active_object
        #print( current_obj.data.vertices.values())
        #print(current_obj.matrix_world)

#        verts_local = [v.co for v in current_obj.data.vertices.values()]
#        verts_world = [current_obj.matrix_world * v_local for v_local in verts_local]
#
#        print("="*40) # printing marker
#
#        for i, vert in enumerate(verts_world):
#          print("vertices[{i}].SetVertex ({v[0]}, {v[1]}, {v[2]});".format(i=i, v=vert))
#
#        for i, face in enumerate(current_obj.data.polygons):
#          verts_indices = face.vertices[:]
#          print("mesh[{i}].SetTriangle {v_i};".format(i=i, v_i=verts_indices))
#

    def ray_cast(self,origin):
       origin=list(np.array(origin).astype(np.float32))

       #print(f"origin={origin}")
       #self.printMesh()
       ray_casting_image=np.zeros((len(self.ray_directions.keys()),len(self.ray_directions[0].keys()),3))

       for alfa in self.ray_directions:
        for beta in self.ray_directions[alfa]:
          direction=self.ray_directions[alfa][beta]
          hit, loc, normal, idx, obj, mw = self.bpy_scene.ray_cast(self.bpy_depsgraph,origin, direction)
          if hit:
              distance=np.linalg.norm(np.array(origin)-np.array(loc))
              gray_scale_color=min(int(25+220*distance/cfg.MAX_RAY_CASTING_DISTANCE),255)
              #gray_scale_color=min(int(254*distance/cfg.MAX_RAY_CASTING_DISTANCE),255)
              ray_casting_image[alfa][beta][0]=gray_scale_color
              #ray_casting_image[alfa][beta][1]=min(int(254*((100*normal[0]+10000*normal[1]+1000000*normal[2])+(1000000*2))/(1000000*4)),255)
              hit1, loc1, normal1, idx1, obj1, mw1 = self.bpy_scene.ray_cast(self.bpy_depsgraph,loc, normal)
              distance1=np.linalg.norm(np.array(loc)-np.array(loc1))
              gray_scale_color1=min(int(25+220*distance1/cfg.MAX_RAY_CASTING_DISTANCE),255)
              #gray_scale_color1=min(int(254*distance1/cfg.MAX_RAY_CASTING_DISTANCE),255)
              ray_casting_image[alfa][beta][1]=gray_scale_color1

              hit2, loc2, normal2, idx2, obj2, mw2 = self.bpy_scene.ray_cast(self.bpy_depsgraph,loc1, normal1)
              distance2=np.linalg.norm(np.array(loc1)-np.array(loc2))
              gray_scale_color2=min(int(25+220*distance2/cfg.MAX_RAY_CASTING_DISTANCE),255)
              #gray_scale_color1=min(int(254*distance1/cfg.MAX_RAY_CASTING_DISTANCE),255)
              ray_casting_image[alfa][beta][2]=gray_scale_color2
              #print(f"normal={normal}")
              #ray_casting_image[alfa][beta][1]=
              #ray_casting_image[alfa][beta][2]=gray_scale_color
#### NOT : MP : CALISMAZSA BIR DE NORMALI DE ISIN ICINE KATMAYI DENEYEBILIRIZ              
              #print(f"HIT: location={np.array(loc).shape} normal_vector_of_hit_point={np.array(norm).shape} idx={np.array(idx).shape} obj={np.array(obj).shape} mw={np.array(mw).shape}")
          else:
              ray_casting_image[alfa][beta][0]=0
              ray_casting_image[alfa][beta][1]=0
              ray_casting_image[alfa][beta][2]=0
       #print( ray_casting_image)
       #ray_casting_image=ray_casting_image.reshape(cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION,1).repeat(3,axis=2)
       #ray_casting_image=Image.fromarray(np.uint8(ray_casting_image), mode="RGB")
       #ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32).reshape(1,3,cfg.RAY_CASTING_IMAGE_RESOLUTION,cfg.RAY_CASTING_IMAGE_RESOLUTION)
       ray_casting_image=torch.tensor(np.array(ray_casting_image).transpose(2, 0, 1), dtype=torch.float32)

       return ray_casting_image


    def generate_ray_directions(self):
     alfas=z_directions=np.arange(0               ,    2*np.pi     ,    np.pi/(cfg.RAY_CASTING_IMAGE_RESOLUTION/2)) # 256 values
     betas=y_directions=np.arange(-np.pi/2,    np.pi/2     ,    np.pi/cfg.RAY_CASTING_IMAGE_RESOLUTION) #  256 values
     unit_vector=[1,0,0]
     ray_directions={}
     for i in range(alfas.shape[0]):
         if i not in ray_directions:
             ray_directions[i]={}
         for j in range(betas.shape[0]):
                   alfa=z_directions[i]
                   beta=y_directions[j]
                   gamma=0
                   rotation_around_z= [  [ np.cos(alfa) , -np.sin(alfa), 0 ],  [ np.sin(alfa), -np.cos(alfa), 0 ],  [ 0 , 0, 1] ]
                   rotation_around_y= [  [ np.cos(beta) ,0, np.sin(beta)],  [0,1,0 ],  [-np.sin(beta) , 0, np.cos(beta)] ]
                   rotation_around_x= [  [1,0,0], [ 0,np.cos(gamma) , -np.sin(gamma) ],  [ 0, np.sin(gamma), np.cos(gamma) ] ]
                   rotation=np.matmul(rotation_around_z,np.matmul(rotation_around_y,rotation_around_x))
                   #yaw=rotation_around_z= [  [ cos_alfa , - sin_alfa, 0 ],  [ sin_alfa , - cos_alfa, 0 ],  [ 0 , 0, 1] ]
                   #pitch=rotation_around_y= [  [ cos_beta , 0,  sin_beta],  [ 0, 1, 0 ],  [ -sin_beta , 0, cos_beta] ].
                   #roll=rotation_around_x= [ [1,0,0 ], [0, cos_gamma,-sin_gamma], [0,sin_gamma,cos_gamma]]
                   #R=rotation_around_z * rotation_around_y * rotation_around_x
                   ray_direction=np.matmul(unit_vector,rotation)
                   ray_directions[i][j]=ray_direction

     return ray_directions


