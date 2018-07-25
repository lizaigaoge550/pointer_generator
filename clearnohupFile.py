import os
import glob
import time
while True:
    #time.sleep(60*60*2)
    for file in glob.glob('./*'):
        if file.endswith('output'):
            print(file)
            os.system("cat /dev/null > " + file)
    time.sleep(60*60*2)
