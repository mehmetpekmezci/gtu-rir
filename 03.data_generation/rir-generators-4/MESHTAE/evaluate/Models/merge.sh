exit 0
rm -f netG.pth
rm -f mesh_net.pth
cat *_netG_*.* | tar xvf -
ln -s *_netG_*.pth netG.pth
ln -s *_mesh_net_*.pth mesh_net.pth

