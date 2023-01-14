
 2010  apt install imagemagick
 2011  sudo apt install imagemagick
 2012  cd ..
 2013  cp -Rf pictures/ pictures.org
 2014  cd pictures
 2015  for i in *.jpg; do   convert -resize 50% $i $i.1.jpg; done
 2016  rm *.jpg
 2017  cp ../pictures.org/* .
 2018  for i in *.jpg; do   convert -resize 50% $i $i.1.jpg; done
 2019  rm *.jpg
 2020  cp ../pictures.org/* .
 2021  for i in *.jpg; do   convert -resize 30% $i $i.1.jpg; done
 2022  history > convert.sh
