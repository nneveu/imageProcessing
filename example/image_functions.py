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
print('Radius before adjustment:', circle_dim['radius'])
#  Masking 20% of the YAG screen edges
circle_dim['radius'] = int(circle_dim['radius']*0.8)
print('Radius after adjustment:', circle_dim['radius'])

#  Load background images
(bx, by,b_Nframes, background_array) = readimage(background_file, header_size=3)

#  Masking everything outside YAG screen
masked_background    = mask_images(background_array, circle_dim)
masked_charge_images = mask_images(charge_images, circle_dim)

#  Uncomment the following to look at the masked images
#view_each_frame(masked_background)
#view_each_frame(masked_charge_images)

#-------------------------------------------------------------------------
#STEP 4: Subtract background and filter xrays
#This step could come before or after filtering image(?)
#(My guess is subtraction should be first, or filter twice??)
#Clean up noise with median filter

#  Average background shots
ave_background = average_images(masked_background)

#  Subtract background
no_background = background_subtraction(masked_charge_images, ave_background)
#view_each_frame(no_background)

#  Apply median filter to all frames
#  There is also a guassian filter available
#  n=3 averages 3x3 array with pixel in middle
#  Currently edges are reflected as default
filtered_images = do_filter(no_background, n=3)
#view_each_frame(no_background)

#-------------------------------------------------------------------------
#STEP 5: Calculate beam sizes with guassian fit.
#Note - you only need to give a base file name. Extensions will be added.
#Optional - crop image and plot with distribution
#(I crop the images to make it square, but this is not required)
#(I hope to add an automatic crop soon)


#  Optional - crop image
#  Get shape of array 
x, y, z = filtered_images.shape
#  usage = np.zeros((dy, dx, dz))
crop_array = np.zeros((350,350,z))
for i in range(0,z):
    crop_array[:,:,i] = crop_image(filtered_images[:,:,i], x_min=80, x_max=430, y_min=80, y_max=430)
#view_each_frame(crop_array)

#  Optional - average cropped images
ave_crop = average_images(crop_array)

#  Plot distributions on averaged image
#  You can give a output file name and title
#  The background is subtracted to produce a white background
outfile = './yag_plus_dist'
add_dist_to_image(ave_crop, fiducial, outfile,title="M=205", background=20)


#  Calculate beam sizes
beamsize_file = './beamsizes'
beamsizes = fit_gaussian(crop_array, fiducial, beamsize_file)







