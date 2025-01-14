for f in $(find . -name '*.jpg')
do
	echo $f
	convert -resize 50% $f $f
done
