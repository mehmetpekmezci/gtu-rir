kill -9 $(ps -ef | grep 'gpu 0'| awk '{print $2}'| tr '\n' ' ')
kill -9 $(ps -ef | grep 'gpu 1'| awk '{print $2}'| tr '\n' ' ')
