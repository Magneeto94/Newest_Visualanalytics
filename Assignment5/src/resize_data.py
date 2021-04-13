import os, sys
import shutil
from shutil import copyfile
import random

'''
------------------------Creating the small scripts-----------------------
'''

def main():
    #folder which contains the sub directories
    source_dir = 'training'

    #list sub directories S
    for root, dirs, files in os.walk(source_dir):

    #iterate through them
        for i in dirs: 

            #create a new folder with the name of the iterated sub dir
            path = 'small_training/' + "%s/" % i
            os.makedirs(path)

            #take random sample, here 3 files per sub dir
            filenames = random.sample(os.listdir('training/' + "%s/" % i ), 40)

            #copy the files to the new destination
            for j in filenames:
                shutil.copy2('training/' + "%s/" % i  + j, path)
            
            
'''
----------------Same code but got the test data-----------------------------
'''

    #folder which contains the sub directories we want to copy
    val_source_dir = 'validation/'
    #list sub directories 
    for root, dirs, files in os.walk(val_source_dir):

        #iterate through them
        for i in dirs: 

            #create a new folder with the name of the iterated sub dir (small_training)
            path = 'small_validation/' + "%s/" % i
            os.makedirs(path)

            #take random sample, here 10 files per sub dir
            filenames = random.sample(os.listdir('validation/' + "%s/" % i ), 10)

            #copy the files to the new destination
            for j in filenames:
                shutil.copy2('validation/' + "%s/" % i  + j, path)
                
                
if __name__ == '__main__':
    main()