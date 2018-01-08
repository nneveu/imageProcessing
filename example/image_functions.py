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
beam_images_file = './YAG1_M205_11-02-2017_img.dat'
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

#  Calculate fiducial (mm/pixels conversion)
fiducial   = fiducial_calc(circle_dim['radius'])
print('fiducial = ', fiducial, 'mm/pixel')

#-------------------------------------------------------------------------
#STEP 2: Cut image array based on charge window.
#Only analyze images of the correct charge.

#  Get charge for each image
#  The default ICT calibration is 1.25
#  Look at function definition (ict_cal) if this needs to be adjusted. 
charge_array, scaled_volts = ict_charge(charge_data_file, data_type='csv')

#  Load images with beam
(dx, dy, Nframes, image_array) = readimage(beam_images_file, header_size=3)
#  Cut images based on charge window
#  usage = select_on_charge(images, charge, min_charge, max_charge)
charge_images = select_on_charge(image_array, charge_array, 0.95, 1.05)

#  To see the ICT curves plotted, uncomment the following:
#  By defualt the first 10 images are made. 
#  This number can be adjusted, see func definition.
#volts_array, time = csv_to_volts_array(charge_data_file)
#plot_ict_curves(scaled_volts, time_array=time)

#-------------------------------------------------------------------------
#STEP 3: Mask everything outside the YAG screen.
#This means all data outside the YAG screen is set to 0.
#This uses the information from step one.  

#  Assuming the beam size is not bigger than the YAG screen, 
#  I mask a little more than the YAG sceen radius.
#  I'm doing this because sometimes the YAG edges are bright 
print('radius before adjustment', circle_dim['radius'])
#  Masking 20% of the YAG screen edges
adjusted_radius = int(circle_dim['radius']*0.8)
print('radius after adjustment', adjusted_radius)

#  Load background images
(bx, by,b_Nframes, background_array) = readimage(yag_back, header_size=3)

#  Masking everything outside YAG screen
masked_background    = mask_images(background_array, circle_dim)
masked_charge_images = mask_images(charge_images, circle_dim)



#(bx, by,b_Nframes, background_array) = readimage(background_file, header_size=3)



#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
