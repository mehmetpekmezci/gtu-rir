firefox 'https://drive.google.com/uc?id=1XOyzsZD3s_pkZBlWcH3KtCR9YpjRVbHG'
echo "Press y when download is finished:"
read ans
while [ "$ans" != "y" ]
do
     read ans
     echo "Press y when download is finished:"
done

echo "Now copy generate.zip to directory where this script resides ($HOME/workspace-home/FAST-RIR)"
echo "Copying finished?(y/n)"
read ans

while [ "$ans" != "y" ]
do
     read ans
     echo "Press y when copying is finished:"
done

unzip generate.zip
