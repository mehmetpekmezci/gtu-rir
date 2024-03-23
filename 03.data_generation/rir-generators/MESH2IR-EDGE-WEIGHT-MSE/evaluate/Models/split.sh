for i in *netG_*.pth
do
   tar cvf - $i| split -b 50m - $i.
done
