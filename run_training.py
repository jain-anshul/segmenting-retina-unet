import os, sys
print sys.argv[1]
os.system('python ./src/'+sys.argv[1]+'_training.py')