PTH_FILE=$(ls *netG_*.pth | tail -1)

tar cvf - $PTH_FILE| split -b 50m - $PTH_FILE.
