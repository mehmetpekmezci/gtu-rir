for i in *_netG_*.pth
do
   tar cvf - $i| split -b 50m - $i.
done
