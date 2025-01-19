rm -Rf $HOME/sonuclar
mkdir -p $HOME/sonuclar
ROOM_NO=208
echo "ROOM_NO=$ROOM_NO" > $HOME/sonuclar/ROOM_NO
echo "TOTAL_WALL-OBJ_COUNT-OBJ_M2-FACE_COUNT-HEAD_COUNT-MSE-SSIM-GLITCH_COUNT" > $HOME/sonuclar/overall.txt
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
	  echo "$TOTAL_WALL-$OBJ_COUNT-$OBJ_M2-$FACE_COUNT-$HEAD_COUNT-$MSE-$SSIM-$GLITCH_COUNT" >> $HOME/sonuclar/overall.txt
  done
  for j in ~/RIR_REPORT.$i/GTURIR/MESHTAE.node*/
  do
	  FACE_COUNT=$(echo $j|sed -e 's/.*node.//'|sed -e 's/.head.*//')
	  HEAD_COUNT=$(echo $j|sed -e 's/.*head.//'|sed -e 's#/.*##')
	  cp -Rf $j/METADATA_GTURIR/Meshes $HOME/sonuclar/Meshes.$i.$FACE_COUNT.$HEAD_COUNT
	  half=$(($(cat $j/room-$ROOM_NO/*/GLITCH_COUNT.db.txt | wc -l)/2))
	  avg_value=$(cat $j/room-$ROOM_NO/*/GLITCH_COUNT.db.txt|cut -d= -f2| sort -n| tail -$half|head -1)
	  avg_record=$(cat $j/room-$ROOM_NO/*/GLITCH_COUNT.db.txt| grep "=$avg_value"| head -1|cut -d= -f1)
	  cp $j/room-$ROOM_NO/*/$avg_record*.png $HOME/sonuclar/avg_record.$i.$FACE_COUNT.$HEAD_COUNT.$avg_record.png
	  echo $avg_value > $HOME/sonuclar/avg_value.$i.$FACE_COUNT.$HEAD_COUNT.$avg_record.txt
  done
done
exit

