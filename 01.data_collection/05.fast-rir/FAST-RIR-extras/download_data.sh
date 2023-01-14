firefox 'https://drive.google.com/uc?id=17NF1MVtXaWe9zhqWJqmG5tFUZb_9X0M5'
echo "Press y when download is finished:"
read ans
while [ "$ans" != "y" ]
do
     read ans
     echo "Press y when download is finished:"
done

echo "Now copy data.zip to directory where this script resides ($HOME/workspace-home/FAST-RIR)"
echo "Copying finished?(y/n)"
read ans

while [ "$ans" != "y" ]
do
     read ans
     echo "Press y when copying is finished:"
done

unzip data.zip
mkdir output
