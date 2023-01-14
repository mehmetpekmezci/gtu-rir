DATA_DIR=$(realpath "../../../data/single-speaker/");
cd $DATA_DIR

for mp4File in $(find $DATA_DIR/room-*/pics -name '*.mp4')
do
	mp4FileDir=$(dirname $mp4File)
	if [ ! -f $mp4FileDir/mp4_to_image.*.jpg ]
	then
	     echo ffmpeg -i $mp4File -vf fps=1 $mp4FileDir/mp4_to_image.%d.jpg
	     ffmpeg -i $mp4File -vf fps=1 $mp4FileDir/mp4_to_image.%d.jpg
	fi
done

for picsDir in $DATA_DIR/room-*/pics
do
   #rm -f $picsDir/*.squared.* ; 
   for i in $picsDir/*.jpg
   do   
	echo $i | grep '\.squared\.'
	if [ $? != 0 ]
	then
	  f=$(echo $i | sed -e 's/.jpg//') ; 
	  if [ ! -f $f.squared.jpg ]
	  then
	    convert $i -resize "3024x3024!" $f.squared.jpg ;
          else
            echo "$f.squared.jpg already exists :)"
	  fi
  	  echo "$f.squared.jpg is done.";
        fi
   done
done
