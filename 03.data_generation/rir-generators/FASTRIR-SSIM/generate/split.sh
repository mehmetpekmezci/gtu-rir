for i in *.pth
do
   tar cvf - $i| split -b 50m - $i.
done
