import subprocess
process = subprocess.Popen('sudo rfcomm connect 0 3D:45:41:74:43:09 1 &',stdout = subprocess.PIPE,shell=True)
temp,err = process.communicate()
