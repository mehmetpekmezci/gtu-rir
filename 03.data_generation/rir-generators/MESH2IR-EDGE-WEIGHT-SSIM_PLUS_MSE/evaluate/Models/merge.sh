rm -f *netG*.pth
rm -f mesh_net.pth
cat *netG_*.* | tar xvf -
ln -s *netG_*.pth netG.pth
ln -s *mesh_net_*.pth mesh_net.pth

