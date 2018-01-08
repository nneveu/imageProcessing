#Add the imageProcessing folder to your $PYTHONPATH
#Or use the following two lines:
#import sys
#sys.path.append("/path/to/imageProcessing") 

from imageReader import *  
from chargeReader2 import *
import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob 

fiducial_file    = './YAG1_fiducial_11-02-2017_img.dat'
images_file      = './YAG1_M205_11-02-2017_img.dat'
background_file  = './YAG1_M205_11-02-2017_background_img.dat'
charge_data_file = './YAG1_M205_11-02-2017_LC584AL.csv'

#-------------------------------------------------------------------------
#STEP 1: Find YAG screen and calculate fiducial
#  Load fiducal image into an array
(fx, fy, fz, fid_image) = readimage(fiducial_file, header_size=3)

#  Find the YAG circle
#  By default the first fiducial image is used. 
#  See function definition to adjust this.
#  This function returns a dictionary with dimensions of the YAG screen
#  To access the data, use the following keys:
#    circle_dim['radius']   = radius of YAG screen in pixels
#    circle_dim['center_x'] = x center of YAG screen in pixels
#    circle_dim['center_y'] = y center of YAG screen in pixels
circle_dim = circle_finder(fid_image, min_r=0.367, max_r=0.38)
fiducial   = fiducial_calc(circle_dim['radius'])

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
