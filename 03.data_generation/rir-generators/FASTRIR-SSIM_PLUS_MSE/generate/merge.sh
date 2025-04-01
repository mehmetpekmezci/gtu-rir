cat *_netG_*.pth.* | tar xvf -
netgfile=$(ls *netG*.pth)
ln -s $netgfile netg.pth

