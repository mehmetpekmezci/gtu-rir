#!/bin/bash

function print_to_summary {
      RIRGRAPH=$1
      SONG=$2
      SONGGRAPH=$3
       
      echo "<td align='center' width='250'>" >> summary.html
      echo "<table>" >> summary.html

      if [ "$RIRGRAPH" = "" ]
      then
          echo "<tr><td align='center' width='200'> </td></tr>" >> summary.html
      elif [ "$RIRGRAPH" = "plot0.png" ]
      then
          echo "<tr><td align='center'> NO RIR </td></tr>" >> summary.html
          echo "<tr><td align='center' ><a href='$RIRGRAPH' target='_blank' ><img src='$RIRGRAPH'   width='200' /></a> </td></tr>" >> summary.html
          echo "<tr><td align='center' ><hr> </td></tr>" >> summary.html
      else
          echo "<tr><td align='center'> RIR DIFF </td></tr>" >> summary.html
          echo "<tr><td align='center' ><a href='$RIRGRAPH' target='_blank' ><img src='$RIRGRAPH'   width='200' /></a> </td></tr>" >> summary.html
          echo "<tr><td align='center' ><hr> </td></tr>" >> summary.html
      fi

      if [ "$SONG" = "" ]
      then
           echo "<tr><td width='200' align='center' > NO MICROPHONE RECORDED SONG  </td></tr>" >> summary.html
      elif [ "$RIRGRAPH" = "plot0.png" ]
      then
           echo "<tr><td align='center'> RECORDED SONG AUDIO </td></tr>" >> summary.html
           echo "<tr><td align='center' > <audio controls style='width: 200px;'><source src='$SONG' type='audio/wav'></audio> </td></tr>" >> summary.html
           echo "<tr><td align='center' ><hr> </td></tr>" >> summary.html
      else
           echo "<tr><td align='center'> RIR*SONG AUDIO </td></tr>" >> summary.html
           echo "<tr><td align='center' > <audio controls style='width: 200px;'><source src='$SONG' type='audio/wav'></audio> </td></tr>" >> summary.html
           echo "<tr><td align='center' ><hr> </td></tr>" >> summary.html
      fi

      if [ "$SONGGRAPH" = "" ]
      then
           echo "<tr><td align='center'  width='200'> </td></tr>" >> summary.html
      elif [ "$RIRGRAPH" = "plot0.png" ]
      then
           echo "<tr><td align='center'> RECORDED/TRANSMITTED SONG GRAPH </td></tr>" >> summary.html
           echo "<tr><td align='center' > <a href='$SONGGRAPH' target='_blank' ><img src='$SONGGRAPH'   width='200' /></a></td></tr>" >> summary.html
      else
           echo "<tr><td align='center'> RIR*SONG GRAPH </td></tr>" >> summary.html
           echo "<tr><td align='center' > <a href='$SONGGRAPH' target='_blank' ><img src='$SONGGRAPH'   width='200' /></a></td></tr>" >> summary.html
      fi

      echo "</table>" >> summary.html
      echo "</td>" >> summary.html
}


echo "<!DOCTYPE html>" > summary.html
echo "<html>" >> summary.html
echo "<body>" >> summary.html
echo "<table border='1'>" >> summary.html
echo "<tr> \
        <td colspan='7'>  \
          <table>  \
              <tr>  \
                  <td><b>TRANSMITTED SONG IT SELF:</b></td> \
                  <td> \
                     <table> \
                       <tr> \
                         <td><img src='transmittedSongSignal.wav.single.png'   width='200' /></td> \
                       </tr> \
                       <tr> \
                         <td><audio controls style='width: 200px;'><source src='transmittedSongSignal.wav' type='audio/wav'></audio></td> \
                       </tr> \
                     </table> \
                   </td> \
               </tr> \
          </table> \
         </td> \
     </tr>" >> summary.html
echo "<tr bgcolor='black'><td height='10' colspan='7'></td></tr>" >> summary.html


color="#adb9c8"



echo "<tr bgcolor='$color'> \
            <td >DATASET</td> \
            <td >MICROPHONE RECORED SONG</td> \
            <td>REAL RIR CONVOLUTION</td> \
            <td>(MSE Loss Fn. FAST-RIR) GENERATED RIR CONVOLUTION</td> \
            <td>(SSIM Loss Fn. FAST-RIR) GENERATED RIR CONVOLUTION</td> \
            <td>(MSE Loss Fn. MESH2IR) GENERATED RIR CONVOLUTION</td> \
            <td>(SSIM Loss Fn. MESH2IR) GENERATED RIR CONVOLUTION</td> \
       </tr>" >> summary.html





for dataset in $(echo GTURIR BUT_REVERBDB)
do
   for sample in $(echo MAX-MSE AVG-MSE MIN-MSE)
   do
      if [ "$color" = "#adb9c8" ]
      then
          color="#7f8183"
      else
          color="#adb9c8"
      fi
      echo "<tr bgcolor='$color'>" >> summary.html
      INFO="$sample-$dataset"
      SAMPLE_WAV_FILE=$(find $sample-$dataset/ -name *fastrir-ssim*RIR-DEPTH-*wav | head -1)
      ROOM_ID=$(echo $SAMPLE_WAV_FILE| sed -e 's/.RIR-DEPTH.*//'|sed -e 's/.*astrir-ssim.//'| sed -e 's/SkalskyDvur_//') 
      ROOM_DEPTH=$(echo $SAMPLE_WAV_FILE| sed -e 's/.*RIR-DEPTH-//'| cut -d- -f1| cut -c1-4)
      ROOM_WIDTH=$(echo $SAMPLE_WAV_FILE| sed -e 's/.*-WIDTH-//'| cut -d- -f1| cut -c1-4)
      ROOM_HEIGHT=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-HEIGHT-//'| cut -d- -f1| cut -c1-4)
      ROOM_VOLUME=$(expr $ROOM_DEPTH*$ROOM_WIDTH*$ROOM_HEIGHT | bc)
      ROOM_FACE_AREA=$(expr 2*$ROOM_DEPTH*$ROOM_WIDTH+2*$ROOM_DEPTH*$ROOM_HEIGHT+2*$ROOM_WIDTH*$ROOM_HEIGHT | bc)
      MX=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-MX-//'|cut -d- -f1| cut -c1-4)
      MY=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-MY-//'|cut -d- -f1| cut -c1-4)
      MZ=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-MZ-//'|cut -d- -f1| cut -c1-4)
      SX=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-SX-//'|cut -d- -f1| cut -c1-4)
      SY=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-SY-//'|cut -d- -f1| cut -c1-4)
      SZ=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-SZ-//'|cut -d- -f1| cut -c1-4)
      RT60=$(echo $SAMPLE_WAV_FILE| head -1| sed -e 's/.*-RT60-//'| cut -c1-4)
      DIST=$(echo "sqrt(($MX-$SX)^2+($MY-$SY)^2+($MZ-$SZ)^2)"| bc)
      INFO="$INFO<hr><br/>ROOM_ID=$ROOM_ID<br/>ROOM_DIMENS.=[$ROOM_DEPTH,$ROOM_WIDTH,$ROOM_HEIGHT]<br/>VOLUME=$ROOM_VOLUME<br/>AREA=$ROOM_FACE_AREA<br/>Microhpone Pos.=[$MX,$MY,$MZ]<br/>Speaker Pos.=[$SX,$SY,$SZ]<br/>RT60=$RT60<br/>Microphone-Speaker Distance =$DIST"
      echo "<td align='left' valign='top'  width='200'>$INFO</td>" >> summary.html

      if [ "$dataset" = "GTURIR" ]
      then
          SONG=$(find $sample-$dataset/ -name real.song.*wav | head -1) 
          SONGGRAPH=$(find $sample-$dataset/ -name real.song.*.r.a.g.png)
	  RIRGRAPH='plot0.png'
          print_to_summary $RIRGRAPH $SONG $SONGGRAPH
      else
          print_to_summary '' '' ''
      fi

      SONG=$(find $sample-$dataset/ -name real.rir.*reverbed.wav) 
      RIRGRAPH=$(find $sample-$dataset/ -name real.rir*.single.png) 
      SONGGRAPH=$(find $sample-$dataset/ -name real.rir.*.r.a.g.png) 
      print_to_summary $RIRGRAPH $SONG $SONGGRAPH

      SONG=$(find $sample-$dataset/ -name 1-*reverbed.wav) 
      RIRGRAPH=$(find $sample-$dataset/ -name 1-*.wave.png) 
      SONGGRAPH=$(find $sample-$dataset/ -name 1-*.r.a.g.png) 
      print_to_summary $RIRGRAPH $SONG $SONGGRAPH
      
      SONG=$(find $sample-$dataset/ -name 2-*reverbed.wav) 
      RIRGRAPH=$(find $sample-$dataset/ -name 2-*.wave.png) 
      SONGGRAPH=$(find $sample-$dataset/ -name 2-*.r.a.g.png) 
      print_to_summary $RIRGRAPH $SONG $SONGGRAPH
      
      SONG=$(find $sample-$dataset/ -name 3-*reverbed.wav) 
      RIRGRAPH=$(find $sample-$dataset/ -name 3-*.wave.png) 
      SONGGRAPH=$(find $sample-$dataset/ -name 3-*.r.a.g.png) 
      print_to_summary $RIRGRAPH $SONG $SONGGRAPH
      
      SONG=$(find $sample-$dataset/ -name 4-*reverbed.wav) 
      RIRGRAPH=$(find $sample-$dataset/ -name 4-*.wave.png) 
      SONGGRAPH=$(find $sample-$dataset/ -name 4-*.r.a.g.png) 
      print_to_summary $RIRGRAPH $SONG $SONGGRAPH
      
      echo "</tr>" >> summary.html
   done
done

echo "</table>" >> summary.html
echo "</body>" >> summary.html
echo "</html>" >> summary.html

