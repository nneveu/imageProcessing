# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 09:57:59 2016
Updated Sep-Dec 2017  

@author: nneveu 

Sources include:
Wiki
Scipy man pages
Stack Overflow

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import r2_score
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import UnivariateSpline
from skimage.measure import compare_ssim as ssim
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage import color
from lmfit import  Model
from lmfit.models import ConstantModel, StepModel, GaussianModel, LorentzianModel, VoigtModel
from mpl_toolkits.axes_grid1 import make_axes_locatable
from random import *


def readimage(imagefile, header_size=6, order_type='F'):
    #This function reads in image data
    # It assumes the first three bits are the 
    # Horizontal size (X), Vertical size (Y),
    # and number of frames (Nframes) respectively
    
    # count=-1 -> reads all data
    # sep='' -> read file as binary
    # header_size=6 for old data aquisition (at AWA)  
    # header_size=3 for new python data aquisition (at AWA)
 
    # header info vert/horiz pixels and number of frames
    data    = np.fromfile(imagefile, dtype=np.uint16)
    dx      = int(data[1])
    dy      = int(data[0])

    if header_size==6:
        Nframes = int(data[2])+1
        length  = dx*dy*Nframes   
        n = header_size #+ 1 
        images  = data[n:]
    else:
        Nframes = int(data[2])
        length  = dx*dy*Nframes   
        n = header_size + 1 
        images  = data[n:]
     
 
    if length != np.size(images):
        print('ERROR array size does not match dimensions, check header_size')
        print('length', length)
        print('image size', np.size(images))
        print(dx, dy, Nframes, int(data[3]), int(data[4]), int(data[5]), int(data[6]))
    #==========================================================
    #Reading images into 3D array 
    # X by Y by Frame Number
    # order_type can = 'C', 'F', 'A'
    images_array = np.reshape(images,(dy, dx, -1), order=order_type)
    #images_array = np.reshape(images,(-1, dx, dy), order=order_type)
    return(dx, dy, Nframes, images_array)  

#-------------------------------------------------------------------------------
def do_filter(image_array, use_filter='median', n=3):
    #http://www.scipy-lectures.org/advanced/image_processing/index.html

    plt.close('all')
    #Deinterlace and filter
    # Applies a median filter to all images 
    # in image_array. Returns an array that is 
    # the same shape and size as input array 

    filtered_image = np.empty_like(image_array)    
    #Finding number of frames
    try:
        #x,y,Nframes = image_array.shape 
        Nframes = len(image_array[0,0,:])
        for i in range(0,Nframes):
            if use_filter == 'median':
                filtered_image[:,:,i] = median_filter(image_array[:,:,i],2)
            else:
                filtered_image[:,:,i] = gaussian_filter(image_array[:,:,i], 1) #order 1 looks best? 

    except: 
    #Using filter on all frames
        if use_filter == 'median':
            #Median averages across two pixels
            #Better for salt and pepper background
            filtered_image = median_filter(image_array,n)

        else:
            #Guassian filter not good for salt and pepper background
            filtered_image = gaussian_filter(image_array, n) #order 1 looks best?

    return(filtered_image)  

#-------------------------------------------------------------------------------   
def view_each_frame(image_array): 
    #This function shows each frame one by one
    # If you want to stop looking at the images
    # before reaching the end of the file, 
    # use CTRL+C to stop the python file execution.
    #print(image_array.shape)

    try:
        #x,y,z = image_array.shape
        Nframes = len(image_array[0,0,:])
        for i in range(0,Nframes):
            image = image_array[:,:,i]
            di_image = do_filter(image)
            plt.close('all')
            plt.figure(1) #closing figures from previous functions
            plt.imshow(di_image)
            plt.show()

    except:
        image = image_array
        di_image = do_filter(image)
        plt.close('all')
        plt.figure(1) #closing figures from previous functions
        plt.imshow(di_image)
        plt.show()
#-------------------------------------------------------------------------------
def average_images(image_array):
    # This function takes all images in 
    # image array and averages them to 
    # create one image
    # https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil
    plt.close('all') #closing figures from previous functions

    #Find dimensions of array
    dx, dy, Nframes= image_array.shape

    #Array that will hold final image
    ave_image = np.zeros((dx,dy), np.float)
    
    for i in range(0, Nframes):
        image = image_array[:,:,i]
        hold = np.array(image, dtype=np.float)        
        ave_image = ave_image + hold/Nframes
        ave_image = np.array(np.round(ave_image), dtype=np.uint16)    
    
    #print('Showing average image. Close image to continue.....')
    #plt.figure(2)
    #plt.imshow(ave_image)#, interpolation='none', extent=[np.min(xaxis), np.max(xaxis), np.min(yaxis), np.max(yaxis)])
    #plt.colorbar()
    #plt.show()

    return ave_image

#-------------------------------------------------------------------------------
def background_subtraction(image_array, background_image, max_pixel=1024):
    #https://www.raspberrypi.org/forums/viewtopic.php?t=38239&p=316837
    plt.close('all') #closing figures from previous functions
    
    no_background_image = np.empty_like(image_array)
    float_back = np.array(background_image, dtype=np.float)
 
    try:  
        Nframes = len(image_array[0,0,:])
        for i in range(0,Nframes):
            float_im   = np.array(image_array[:,:,i], dtype=np.float)
            no_background_image[:,:,i] = np.clip(float_im - float_back, 0, max_pixel)
            #implot = plt.imshow(no_background_image[:,:,i])
            frames = True
            #print 'max image', np.max(image_array)
            #print 'max back', np.max(background_image)

    except: 
        float_im = np.array(image_array, dtype=np.float)
        no_background_image = np.clip(float_im - float_back, 0, None)
        frames = False
    no_background_image = np.array(np.round(no_background_image), dtype=np.uint16) 
    no_background_image = np.clip(no_background_image, 0, max_pixel)

    #print("Showing first image with no background, close image to continue...")    
    #plt.figure(3)
    #if frames:
    #    implot = plt.imshow(no_background_image[:,:,0])
    #else:
    #    implot = plt.imshow(no_background_image)
    #
    #plt.colorbar()
    #plt.show()
    return(no_background_image) 

#-------------------------------------------------------------------------------
def fiducial_calc(radius, YAG_D=44.45 ):
    #Most fiducial images are looking at the inner circle of the YAG holder.
    # if this is the case, use a radius = 44.45 mm
    # If looking at the outer edge of the YAG holder, 
    # use radius = 50.038 mm

    #Radii of YAG can give us fiducial
    YAG_r = YAG_D / 2
    fiducial = YAG_r / radius

    return(fiducial)
#-------------------------------------------------------------------------------  
def remove_beam(image, percent_threshold=0.8):
    #Removes brightest part of picture. 
    #Higher threshold means less is removed.
    max_val = np.max(image)
    image[image > max_val*percent_threshold] = int(random()*5) 
    plt.imshow(image)
    plt.colorbar()
    plt.show()
    return (image)

#-------------------------------------------------------------------------------
def select_on_charge(images, charge, min_charge, max_charge):
    #Using a positive convention for inputs.
    #This means a larger negative number is the max charge.
    #    i.e. -40nC is larger than -20nC
    if ((max_charge*min_charge) > 0) & (max_charge<0):
        pass
    elif ((max_charge < 0) & (min_charge > 0)) or ((max_charge > 0) & (min_charge < 0)):
        print("You entered charge values with different signs")
        print("Please check charge values and try again")
        print("Leaving this function now...bye bye!")
        return None 

    elif (max_charge*min_charge) > 0:  
        max_charge = -max_charge
        min_charge = -min_charge
    
    loc = np.where( (charge[0,:] > max_charge) & (charge[0,:] < min_charge) )
    #n_images = len(loc[0])
    print('Number of data sets in specified charge range:', len(loc[0]))#n_images)
    #print np.shape(images)
    #Getting corresponding images
    print('Average charge in specifed range is: ', np.mean(charge[0,loc]))
    charge_images = images[:,:,loc[0]]

    return(charge_images)#, n_images)
#-------------------------------------------------------------------------------
def raw_data_curves(image, oneframe=1 ):
    # At the moment, this function is only finding raw 
    # data curve for one image frame. 
    # proj = projection in x or y direction
    if oneframe == 1:
       f1 = image
    
    dy, dx = np.shape(image)
    #print('raw data', dx, dy)
    #X fit, one for one sum across lines
    proj_x = np.zeros([dx])
    for i in range(0,dx):     
        line = f1[:,i]
        proj_x[i] = np.sum(line)
    
    #Finding y fit
    proj_y = np.zeros([dy])
    for j in range(0,dy): 
        line = f1[j,:]
        proj_y[j] = np.sum(line)
         
    return (proj_x, proj_y)

#-------------------------------------------------------------------------------
def fit_gaussian(images, fiducial, output_filename, clip_tail=50, print_r2='no'):
    #https://lmfit.github.io/lmfit-py/builtin_models.html
    #https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.GaussianModel
    #https://lmfit.github.io/lmfit-py/model.html#lmfit.model.ModelResult
    #https://lmfit.github.io/lmfit-py/builtin_models.html#lmfit.models.RectangleModel 
    
    #Place holders
    chix = 0
    chiy = 0 
    #Finding number and size of images
    dx, dy, n_images  = np.shape(images)

    #Creating empty arrays to hold sigma 
    #value for each image
    sigmax    = np.zeros((n_images))
    sigmay    = np.zeros((n_images))
    print('Using fiducial of:', fiducial, '[mm/pixel]')
    #print(np.shape(images))
    beamsizes = {}

    #choosing Gaussian fit
    mod   = GaussianModel()
    #mod  = LorentzianModel()
    #mod  = VoigtModel()
 
    plt.close('all') #closing figures from previous functions
    pdffile =  output_filename +'_fit_curves.pdf'
    print('Calculating the fits and plotting the results...')
    with PdfPages(pdffile) as pdf:

        for n in range(0,n_images):
            #getting raw data curves 
            xprojection, yprojection = raw_data_curves(images[:,:,n])
            if clip_tail == 0: 
                no_tailx = xprojection
                no_taily = yprojection
            else:
                no_tailx = xprojection[clip_tail:-clip_tail]
                no_taily = yprojection[clip_tail:-clip_tail]

            #Normalized projection
            #norm_x   = no_tailx/np.max(no_tailx)
            #norm_y   = no_taily/np.max(no_taily)
 
            x_points = len(no_tailx) #x_max = x_points*fiducial
            y_points = len(no_taily) #y_max = y_points*fiducial
            
            #Calculating x and y axis in mm, using fiducial (mm/pixel)    
            #The center of the axis is zero, this is an arbitrary choice
            x_axis   = (np.arange(0,x_points) - x_points/2)*fiducial
            y_axis   = (np.arange(0,y_points) - y_points/2)*fiducial
           
            #outx = type_model(xprojection, x_axis)
            #outy = type_model(yprojection, y_axis)
            ##Calc sigmax 
            parsx      = mod.guess(no_tailx, x=x_axis)
            outx       = mod.fit(no_tailx, parsx, x=x_axis)
            #parsx      = mod.guess(norm_x, x=x_axis)
            #outx       = mod.fit(norm_x, parsx, x=x_axis)
            paramsx    = outx.best_values
            sigmax[n]  = paramsx['sigma']
            #print(parsx.keys()) # = odict_keys(['sigma', 'center', 'amplitude', 'fwhm', 'height'])

            ##['chi-square']
            ##Calc sigmay
            parsy = mod.guess(no_taily, x=y_axis)
            outy  = mod.fit(no_taily, parsy, x=y_axis) 
            #parsy = mod.guess(norm_y, x=y_axis)
            #outy  = mod.fit(norm_y, parsy, x=y_axis) 
            paramsy = outy.best_values
            sigmay[n]  = paramsy['sigma']
            
            #Summing chisqr for average later
            chix = chix + outx.chisqr
            chiy = chiy + outy.chisqr
            #print('\nX chi-sq:')
            #print(outx.chisqr/n_images)
            ##print(outx.fit_report())
            #print('\nY chi-sq:')
            #print(outy.chisqr/n_images)
          

            #calc r^2
            if print_r2=='yes':
                #Calc r^2
                coefficient_of_dermination_x = r2_score(no_tailx, outx.best_fit)
                print('rx^2= ',coefficient_of_dermination_x)
                
                coefficient_of_dermination_y = r2_score(no_taily, outy.best_fit)
                print('ry^2= ',coefficient_of_dermination_y)
  
            ##Plotting curves 
            plt.title('Raw data and Gaussian Fit')
            plt.xlabel('[mm]', size=14)
            plt.ylabel('Pixel Intensity [arb. units]', size=14)
            #plt.plot(x_axis, norm_x, 'b.', label='x-axis', markersize=1)
            #plt.plot(y_axis, norm_y, 'k.', label='y-axis', markersize=1)
            plt.plot(x_axis, no_tailx, 'b.', label='x-axis', markersize=1)
            plt.plot(y_axis, no_taily, 'k.', label='y-axis', markersize=1)
            plt.plot(y_axis, outy.best_fit, 'k--')
            plt.plot(x_axis, outx.best_fit, 'b-')
            plt.legend(loc='best')
            pdf.savefig(bbox_inches='tight')
            plt.close()
    plt.close('all')
    
    print('\nAverage X chi-sq:')
    print(outx.chisqr/n_images)
    #print(outx.fit_report())
    print('\nAverage Y chi-sq:')
    print(outy.chisqr/n_images)
    #print('sigmax', sigmax)
    #print('sigmay', sigmay)
    beamsizes['sigmax'] = sigmax 
    beamsizes['sigmay'] = sigmay 
    np.save(output_filename+'.npy', beamsizes)

    #z = np.polyfit(x_axis, xprojection, 30)
    #f = np.poly1d(z)
    #y_new = f(x_axis)
    #popt, pcov = curve_fit(xprojection, x_axis, ydata)
    #plt.plot(x_axis, xprojection)
    #plt.plot(x_axis, y_new)
    return (beamsizes)
#-------------------------------------------------------------------------------
def combo_model(images, fiducial, outfile, fit_type='combo', print_r2='no'):
    plt.close('all')
    #Finding number and size of images
    dy, dx, n_images  = np.shape(images)

    #Creating empty arrays to hold sigma 
    #value for each image
    sigmax    = np.zeros((n_images))
    sigmay    = np.zeros((n_images))
    rx2       = np.zeros((n_images))
    ry2       = np.zeros((n_images))
    print('Using fiducial of:', fiducial, '[mm/pixel]')
    #print(np.shape(images))
    beamsizes = {}

    print('Calculating the fits and plotting the results...')
    pdffile =  'combo_model_fit_curves.pdf'
    with PdfPages(outfile) as pdf:
        for i in range(0,n_images):
            image = images[:,:,i]
            xprojection, yprojection = raw_data_curves(image)
            xaxis   = (np.arange(0,dx) - dx/2)*fiducial
            yaxis   = (np.arange(0,dy) - dy/2)*fiducial
        
            gauss_mod = GaussianModel(prefix='gauss_')
            const_mod = ConstantModel(prefix='const_')
            #stepup_mod  = StepModel(prefix='stepu_')
            #stepdn_mod  = StepModel(prefix='stepd_')
            mod = const_mod + gauss_mod 
           
            cx = const_mod.guess(xprojection) #c=np.min(xprojection))
            gx = gauss_mod.guess(xprojection, x=xaxis)
            parsx = cx + gx 
            outx = mod.fit(xprojection, parsx, x=xaxis)
            paramsx    = outx.best_values
            #print(paramsx.keys())
            sigmax[i]  = paramsx['gauss_sigma']
           
            cy = const_mod.guess(yprojection) #c=np.min(xprojection))
            gy = gauss_mod.guess(yprojection, x=yaxis)
            parsy = cy + gy 
            outy = mod.fit(yprojection, parsy, x=yaxis)
            paramsy    = outy.best_values
            sigmay[i]  = paramsy['gauss_sigma']
            

            #coefficient_of_dermination_x = r2_score(xprojection, outx.best_fit)   
            #coefficient_of_dermination_y = r2_score(yprojection, outy.best_fit)
            rx2[i] = r2_score(xprojection, outx.best_fit)
            ry2[i] = r2_score(yprojection, outy.best_fit)  

            #Calc r^2
            if print_r2 == 'yes':
                print('image number:', i)
                print('rx^2= ',coefficient_of_dermination_x)
                print('ry^2= ',coefficient_of_dermination_y)
    
            #sigmax[n]  = paramsx['sigma']
            #print(parsx.keys()) # = odict_keys(['sigma', 'center', 'amplitude', 'fwhm', 'height'])
            #print(outx.fit_report())
            #print(outy.fit_report())
    
            plt.plot(xaxis, xprojection, 'b.', markersize=1,  label='raw x data')
            #plt.plot(xaxis, outx.init_fit, 'k--', label='inital guess')
            plt.plot(xaxis, outx.best_fit, 'b-', label='best fit')
            
            plt.plot(yaxis, yprojection, 'k.', markersize=1,  label='raw y data')
            #plt.plot(xaxis, outy.init_fit, 'k--', label='inital guess')
            plt.plot(yaxis, outy.best_fit, 'k-', label='best fit')
            plt.legend(loc='best')
            pdf.savefig(bbox_inches='tight', dpi=1000)
            plt.close()
    plt.close('all')

    beamsizes['sigmax']=sigmax
    beamsizes['sigmay']=sigmay

    return(beamsizes, rx2, ry2)

#-------------------------------------------------------------------------------
def fwhm_calc(images, fiducial, outfile_base):
    #This function calcs fwhm of the data points
    #Getting shape of image
    
    try:
        dx, dy, n_images = images.shape
    except:
        n_images = 1
        dx, dy = images.shape

    #Array for holding fwhm values
    fwhm = np.zeros((n_images))
    #Calculating x and y axis in mm, using fiducial (mm/pixel)    
    #The center of the axis is zero, this is an arbitrary choice
    xaxis   = (np.arange(0,dx) - dx/2)*fiducial
    yaxis   = (np.arange(0,dy) - dy/2)*fiducial

    for i in range(0,dz):
        image = images[:,:,i]
        #Calc projection and normalize
        xprojection, yprojection = raw_data_curves(image)
        xprojnorm = (xprojection - np.min(xprojection))/(np.max(xprojection)-np.min(xprojection))#*15 -20  
        yprojnorm = (yprojection - np.min(yprojection))/(np.max(yprojection)-np.min(yprojection))#*15 -20 
        
        fwhm[i] = 0

    return(fwhm) 



#-------------------------------------------------------------------------------
def crop_image(image, x_min=0, x_max=480, y_min=0, y_max=640):
    #Must be one frame
    #dx, dy = image.shape()    
    cropped = image[y_min:y_max, x_min:x_max]
    #plt.close('all') #closing figures from previous functions
    #plt.figure(400)
    #plt.imshow(cropped)
    #plt.show()
    return(cropped)
#--------------------------------------------------------------------------------
def add_dist_to_image(crop, fiducial, filename, title='no title set', background=1):
   plt.close("all") #closing figures from previous functions
   #Currently only takes 1 frame

   #This function is mostly plot formatting.
   # step 0 - calc axis values in mm
   # step 1 - calc x and y projection
   # step 2 - normalize projections
   # step 3 - format plots

   #Getting shape of image
   dy,dx = crop.shape
   #print(dx,dy)

   #Calculating x and y axis in mm, using fiducial (mm/pixel)    
   #The center of the axis is zero, this is an arbitrary choice
   xaxis   = (np.arange(0,dx) - dx/2)*fiducial
   yaxis   = (np.arange(0,dy) - dy/2)*fiducial

   #Calc projection and normalize
   xprojection, yprojection = raw_data_curves(crop, oneframe=1)
   xprojnorm = (xprojection - np.min(xprojection))/(np.max(xprojection)-np.min(xprojection))#*15 -20  
   yprojnorm = (yprojection - np.min(yprojection))/(np.max(yprojection)-np.min(yprojection))#*15 -20 
 
   #Figure formatting and plotting
   fig, ax = plt.subplots(figsize=(10.5, 10.5))
   ax.set_aspect(1.)
   divider = make_axes_locatable(ax)
   axHistx = divider.append_axes("top", 1.25, pad=0.1, sharex=ax)
   axHisty = divider.append_axes("right", 1.25, pad=0.1, sharey=ax)
   # make some labels invisible
   axHistx.xaxis.set_tick_params(labelbottom=False)
   axHisty.yaxis.set_tick_params(labelleft=False)
   #print(len(xaxis), len(xprojnorm))
   axHisty.plot(yprojnorm, -yaxis, linewidth=3) #xprojnorm, -xaxis, linewidth=3)
   axHistx.plot(xaxis, xprojnorm, linewidth=3)#yaxis, yprojnorm, linewidth=3)#, orientation='horizontal') 
 
   cmap = plt.cm.viridis 
   cmap.set_under(color='white')    
   color = ax.imshow(crop, interpolation='none', cmap=cmap, vmin=background, extent=[np.min(xaxis), np.max(xaxis), np.min(yaxis), np.max(yaxis)])
   #ax.plot(xaxis, xprojnorm, '--', linewidth=5, color='firebrick')
   #ax.plot(yaxis, yprojnorm, '--', linewidth=5, color='firebrick') 
   ax.tick_params(labelsize=12)
   axHistx.set_title(title, size=20)
   ax.set_xlabel('X [mm]', size=18)
   ax.set_ylabel('Y [mm]', size=18)
   plt.colorbar(color,ax=ax, orientation="horizontal", shrink=0.7, pad=0.1)
   plt.savefig(filename+'.pdf', dpi=1000, bbox_inches='tight')
   #plt.show()

#-------------------------------------------------------------------------------- 
def similarity_check(image_array):
    #http://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    Nframes = len(image_array[0,0,:])
    s_ave  = 0
    for i in range(0,Nframes):
        s = ssim(image_array[:,:,0], image_array[:,:,i])
        s_ave = s_ave + s/Nframes 
            
    return s_ave

#-------------------------------------------------------------------------------- 
def createCircularMask(h, w, center=None, radius=None):
    #https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
    #mask = createCircularMask(dy, dx, center=[cx,cy], radius=np.mean(radii))

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

#-------------------------------------------------------------------------------- 
def mask_images(image_array, circle_dim): #, im_center, im_radius):
    #This function takes in the YAG circle dimensions
    # and uses that info to mask all data outside the YAG circle. 
    im_center = [circle_dim['center_x'], circle_dim['center_y']]
  
    #Getting dimensions of image array 
    try: 
        h, w, z = image_array.shape  
    except:
        h, w = image_array.shape
        z = 1

    #Creating mask using YAG circle dimensions, see function above
    mask = createCircularMask(h, w, center=im_center, radius=circle_dim['radius'])
    masked_img = image_array.copy()
   
    if z > 1: 
        for i in range(0,z):
            hold              = masked_img[:,:,i]
            hold[~mask]       = 0
            masked_img[:,:,i] = hold   
    else:
        masked_img[~mask] = 0 
       
    #print('Showing masked image. Close picture to continue..')
    #plt.imshow(masked_img[:,:,0])
    #plt.show() 
    return(masked_img)
#-------------------------------------------------------------------------------- 
def circle_finder(image, sigma=0.25, min_r=0.25, max_r=0.35, n=0):
    #This function finds the yag screen and returns the 
    # dimensions of the circle in a dictionary. 
    # This info can be used to find the fiducial 
    # of the image and create a mask.

    # n = image location, if there is more than one image in array and 
    #     you do not want to use the first image as the fiducial. 
    #     Default is to use the first image, assuming all images
    #     in the fiducial file are nearly identical

    # min/max_r = guess at min radius size, in percentage of pixels
    #             This number will be used to search for the YAG screen. 
    #             If the YAG is larger than half the image, 
    #             0.25 is a good guess for the radius 
    #             i.e. radius is on scale of 1/4 size of image
    
    # sigma = 

    # image = fiducial image where the YAG circule is clear, preferably, with no beam. 
    #         If there is beam, use the remove_beam() function first.
   
    #Sources referenced:
    #http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    #https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    #https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
    #https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
    
    #Getting dimensions of image
    #Grabbing first image if multiple shots
    #You can choose a alternate image by adjusting n
    try: 
        dx, dy, dz = image.shape
        image = image[:,:,n]
    except:
        dx, dy = image.shape
    print('\nFinding circle in image with dimensions:', image.shape)
    #plt.imshow(image)
    #plt.show()

    v = np.median(image)
   
    #Make edges sharper  
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(np.max(image), (1.0 + sigma) * v))
    edges = canny(image, sigma=1, low_threshold=lower, high_threshold=upper)

    #Making array of possible radius values 
    #for YAG screen in pixels
    lower_limit = int(max(dx,dy)*min_r)
    upper_limit = int(max(dx,dy)*max_r)
    hough_radii = np.arange(lower_limit, upper_limit, 1)
    print('Checking this many radii possibilities: ', len(hough_radii))
    if len(hough_radii) > 40: 
        print('This number is larger than 40, adjust min_r and max_r to reduce posibilities')
    print('Range of radius values is '+str(np.max(hough_radii))+'-'+ str(np.min(hough_radii))+' pixels')
    #Hough transform accumulator  
    hough_res = hough_circle(edges, hough_radii)    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=3)

    #Averaging the 3 best options to 
    #get center and radius of YAG screen
    center_y = int(round(np.mean(cy)))
    center_x = int(round(np.mean(cx)))
    radius   = int(round(np.mean(radii)))
    print('- Now showing results - ')
    print('radius:', radius, '\ncenter_x:', center_x, '\ncenter_y:', center_y)
    print('Now showing image with resulting circle.')
    print('This is for visual confirmation and no further input is needed.')
    print('If the circle is not centered on the YAG, adjust min_r and min_x in circle_finder.')
    print('Continue with rest of script by closing picture.\n')

    plt.close('all') #closing figures from previous functions
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 2))
    circy, circx = circle_perimeter(center_y, center_x, radius) 
      
    #rescaling to 8bit for easy inspection
    #This does not effect result, only for eye double check
    min_val = np.min(image)
    max_val = np.max(image)
    test    = image
    test    = test.clip(min_val, max_val, out=test)
    test   -= min_val 
    np.floor_divide(test, (max_val - min_val + 1) / 256, out=test, casting='unsafe')
    test    = test.astype(np.uint8)
    image2 = color.gray2rgb(test) 
    #plt.imshow(image2)
    #plt.show()
 
    if (all(x <= 480 for x in circx) and all(y <= 640 for y in circy)):
        #Circle fits in original image
        image2[circy, circx] = (255, 255, 0) #(220, 20, 20)
        ax.imshow(image2)
        plt.show()
    elif (any(x > 480 for x in circx) or any(y > 640 for y in circy)):
        print('Circle is bigger than image, padding array...')
            #Amount of padding needed
        padx = int((np.max(circx) - 480) / 2)
        pady = int((np.max(circy) - 640) / 2)
        if padx < 0: 
            padx=0
        if pady <0:
            pady=0
        print(padx, pady)
        pad_image = np.pad(image2,((pady+1, pady+1), (padx+1, padx+1), (0, 0)), mode='constant', constant_values=0)
        print('new image size', np.shape(pad_image))
        pad_image[circy, circx] = (255, 255, 0) #(220, 20, 20)
        ax.imshow(pad_image)
        plt.show()
    else: 
            print('Somethings wrong, circle dimensions out of bounds.')

    #Making dictionary with YAG circle dimensions
    circle_dimensions = {}
    circle_dimensions['radius']   = radius
    circle_dimensions['center_x'] = center_x
    circle_dimensions['center_y'] = center_y

    return(circle_dimensions)


#==============================================================================
#old stuff
#==============================================================================
# mask = rotate>0
# rotate = skimage.transform.rotate(edges, 0.0, resize=True)
# crop = rotate[np.ix_(mask.any(1),mask.any(0))]
# crop2 = image[np.ix_(mask.any(1),mask.any(0))]
#==============================================================================
# #nonzeroCols = ~np.all(edges==False, axis=0)
# #nonzeroRows = ~np.all(yag1==False, axis=1)
# #nonzeroCols = (edges==False).all(axis=1)
# topcols = ~np.all(edges[:,0:100]==False, axis=0)
# botcols = ~np.all(edges[:,400:480]==False, axis=0)
# cols = ~np.all(edges==False, axis=0)
# #rows = ~np.all(edges==False, axis=1)
#==============================================================================
#==============================================================================
#plt.imshow(denoise_bilateral(image, multichannel=False))#, sigma_range=0.1, sigma_spatial=15))
#==============================================================================
# # image_result = inpaint.inpaint_biharmonic(f1, mask)#, multichannel=True)
# # i = scipy.ndimage.map_coordinates(z, np.vstack((x,y)))
# #Need to flip x and y values in this array
# #x, y = np.mgrid[640:0:-1, 480:0:-1]
# #plt.pcolor(x,y, f1, cmap='RdBu', vmin=np.min(f1), vmax=np.max(f1))
# #plt.pcolormesh(x,y, f1, cmap='copper', norm=LogNorm(vmin=1, vmax=np.max(f1)))
#==============================================================================

#cmap colors:
    #Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, 
    #BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, 
    #GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, 
    #OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, 
    #Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, 
    #PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, 
    #Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, 
    #RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, 
    #Set2_r, Set3, Set3_r, Spectral, Spectral_r, Wistia, Wistia_r, 
    #YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, 
    #afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, 
    #brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, 
    #copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, 
    #gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, 
    #gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, 
    #gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, 
    #gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, inferno, inferno_r, 
    #jet, jet_r, magma, magma_r, nipy_spectral, nipy_spectral_r, ocean, 
    #ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, 
    #rainbow_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, 
    #summer, summer_r, terrain, terrain_r, viridis, viridis_r, winter, winter_r


