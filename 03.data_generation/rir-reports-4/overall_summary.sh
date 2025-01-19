ROOM_NO=208
echo "ROOM_NO=$ROOM_NO"
echo "TOTAL_WALL-OBJ_COUNT-OBJ_M2-FACE_COUNT-HEAD_COUNT-MSE-SSIM-GLITCH_COUNT"
TOTAL_WALL=$(cat ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$ROOM_NO.mesh.0.obj.base_wall_surface)
for((i=0;i<5;i++))
do
  OBJ_COUNT=$(cat ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$ROOM_NO.mesh.$i.obj.obj_count 2>/dev/null)
  if [ ! -f ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$ROOM_NO.mesh.$i.obj.obj_count ]
  then
	  OBJ_COUNT=0
  fi

  OBJ_M2=$(cat ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$ROOM_NO.mesh.$i.obj.obj_surface_area 2>/dev/null)
  if [ ! -f ~/RIR_DATA/GTURIR-ROOM-MESH/gtu-cs-room-$ROOM_NO.mesh.$i.obj.obj_surface_area ]
  then
	  OBJ_M2=0
  fi

  for j in ~/RIR_REPORT.$i/GTURIR/MESHTAE.node*/summary/*.txt
  do  
	  #echo $j
	  FACE_COUNT=$(echo $j|sed -e 's/.*node.//'|sed -e 's/.head.*//')
	  HEAD_COUNT=$(echo $j|sed -e 's/.*head.//'|sed -e 's#/.*##')
	  MSE=$(cat $j|grep MEAN_MSE|cut -d= -f2)
	  SSIM=$(cat $j|grep MEAN_SSIM|cut -d= -f2)
	  GLITCH_COUNT=$(cat $j|grep GLITCH_COUNT|cut -d= -f2)
	  echo "$TOTAL_WALL-$OBJ_COUNT-$OBJ_M2-$FACE_COUNT-$HEAD_COUNT-$MSE-$SSIM-$GLITCH_COUNT"
  done
done
exit

