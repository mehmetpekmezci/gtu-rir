cp $(ls -1art output/$(ls -a1rt output/| grep RIR|tail -1)/Model/mesh_net_epoch*.pth|tail -1) pre-trained-models/gae_mesh_net_trained_model.pth
