# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 09:57:59 2016
Updated on Fri Sep 23-27 

@author: nneveu (the best!)

Load YAG screen images.

Sources include:
Wiki
Scipy man pages
Stack Overflow

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, median_filter
from skimage.measure import compare_ssim as ssim
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from lmfit import  Model
from lmfit.models import GaussianModel, LorentzianModel, VoigtModel

def readimage(imagefile):
    #This function reads in image data
    # It assumes the first three bits are the 
    # Horizontal size (X), Vertical size (Y),
    # and number of frames (Nframes) respectively
    
    # count=-1 -> reads all data
    # sep='' -> read file as binary
    
    # header info vert/horiz pixels and number of frames
    header  = np.fromfile(imagefile, dtype=np.uint16, count=5,sep='')
    print header
    dx      = int(header[0])
    dy      = int(header[1])
    Nframes = int(header[2])
    length  = dx*dy*Nframes    
    images  = (np.fromfile(imagefile, dtype=np.uint16, count=-1,sep=''))[6:]
    #==========================================================
    #Reading images into 3D array 
    # X by Y by Frame Number
    #print header 
    #print images
    images_array = np.reshape(images,(dx, dy, -1), order='F')

    return(dx, dy, Nframes, images_array)  

#-------------------------------------------------------------------------------
def difilter(image_array, use_filter='median'):
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
            filtered_image = median_filter(image_array,2)

        else:
            #Guassian filter not good for salt and pepper background
            filtered_image = gaussian_filter(image_array, 1) #order 1 looks best?

    return(filtered_image)  

#-------------------------------------------------------------------------------   
def view_each_frame(image_array): 
    #This function shows each frame one by one
    # If you want to stop looking at the images
    # before reaching the end of the file, 
    # use CTRL+C to stop the python file execution.
    print image_array.shape

    try:
        #x,y,z = image_array.shape
        Nframes = len(image_array[0,0,:])
        for i in range(0,Nframes):
            image = image_array[:,:,i]
            di_image = difilter(image)
            plt.figure()
            plt.imshow(di_image)
            plt.show()

    except:
        image = image_array
        di_image = difilter(image)
        plt.figure()
        plt.imshow(di_image)
        plt.show()
#-------------------------------------------------------------------------------
def average_images(image_array):#, fiducial='no'):
    #This function takes all images in 
    # image array and averages them to 
    # create one image
    # https://stackoverflow.com/questions/17291455/how-to-get-an-average-picture-from-100-pictures-using-pil

    #Find dimensions of array
    dx, dy, Nframes= image_array.shape
    #dx = len(image_array[:,0,0])
    #dy = len(image_array[0,:,0])
    #print dx, dy

    #Array that will hold final image
    ave_image = np.zeros((dx,dy), np.float)
    #Number of frames to average over
    #Nframes = len(image_array[0,0,:])
    
    for i in range(0, Nframes):
        image = image_array[:,:,i]
        hold = np.array(image, dtype=np.float)        
        ave_image = ave_image + hold/Nframes
        ave_image = np.array(np.round(ave_image), dtype=np.uint16)    

    plt.imshow(ave_image)#, interpolation='none', extent=[np.min(xaxis), np.max(xaxis), np.min(yaxis), np.max(yaxis)])
    plt.colorbar()
    plt.show()

    return ave_image

#-------------------------------------------------------------------------------
def background_subtraction(image_array, background_image, max_pixel=1024):
    #https://www.raspberrypi.org/forums/viewtopic.php?t=38239&p=316837
    #Find dimensions of array
    #dx = len(image_array[:,0,0])
    #dy = len(image_array[0,:,0])
    #Nframes = len(image_array[0,0,:])
    no_background_image = np.empty_like(image_array)
    float_back = np.array(background_image, dtype=np.float)
 
    try:  
        Nframes = len(image_array[0,0,:])
        for i in range(0,Nframes):
            float_im   = np.array(image_array[:,:,i], dtype=np.float)
            no_background_image[:,:,i] = np.clip(float_im - float_back, 0, max_pixel)
            implot = plt.imshow(no_background_image[:,:,i])
    #print 'max image', np.max(image_array)
    #print 'max back', np.max(background_image)

    except: 
        float_im = np.array(image_array, dtype=np.float)
        no_background_image = np.clip(float_im - float_back, 0, None)
    
    no_background_image = np.array(np.round(no_background_image), dtype=np.uint16) 
    no_background_image = np.clip(no_background_image, 0, max_pixel)
    
    #implot = plt.imshow(no_background_image)
    #plt.colorbar()
    #plt.show()
    return no_background_image 

#-------------------------------------------------------------------------------
def fiducial_calc(image, sigma=0.25, min_r=0.25, max_r=0.35, YAG_D=44.45):
    #min/max_r = guess at min radius size, in terms of percentage of pixels
    #This number will be used to search for yag screen. 
    #So, if YAG is about or larger than half the screen, 0.25 is a good
    # guess for the radius - i.e. radius is on scale of 1/4 size of image

    #http://scikit-image.org/docs/dev/auto_examples/edges/plot_circular_elliptical_hough_transform.html
    #https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
    #https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
    #https://stackoverflow.com/questions/14464449/using-numpy-to-efficiently-convert-16-bit-image-data-to-8-bit-for-display-with
    #Only looks at one image right now
    dx, dy = image.shape

    v = np.median(image)
    # apply automatic Canny edge detection using the computed median

    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(np.max(image), (1.0 + sigma) * v))
    edges = canny(image, sigma=1, low_threshold=lower, high_threshold=upper)

    #Making array of possible radius values 
    #for YAG screen in pixels
    lower_limit = int(max(dx,dy)*min_r)
    upper_limit = int(max(dx,dy)*max_r)
    hough_radii = np.arange(lower_limit, upper_limit, 1)
    print 'Checking this many radii possibilities: ', len(hough_radii)
    print 'Max radius', np.max(hough_radii), 'Min radius', np.min(hough_radii) 
    print 'If this number is larger than 40, adjust min_r and max_r to reduce posibilities'
    #Hough transform accumulator  
    hough_res = hough_circle(edges, hough_radii)    
    # Select the most prominent 3 circles
    accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii,total_num_peaks=3)
    # Draw them
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 2))

    #rescaling to 8bit for easy inspection
    #This does not effect result, purely for eye double check
    min_val = np.min(image)
    max_val = np.max(image)
    test    = image
    test    = test.clip(min_val, max_val, out=test)
    test   -= min_val 
    np.floor_divide(test, (max_val - min_val + 1) / 256, out=test, casting='unsafe')
    test    = test.astype(np.uint8)

    image2 = color.gray2rgb(test) 
    print image2.shape 
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = circle_perimeter(center_y, center_x, radius)
        image2[circy, circx] = (220, 20, 20)

    ax.imshow(image2)
    plt.show()
    #plt.imshow(hough_res[-1])
    #plt.show()
    #print accums, radii, cx, cy
    
    #Find mean of radii
    radius = np.mean(radii)
    print "radii", radii, "radius", radius
    #Radii of YAG can give us fiducial
    YAG_r = YAG_D / 2
    fiducial = YAG_r / radius

    return(fiducial)
#-------------------------------------------------------------------------------  
def remove_beam(image, percent_threshold=0.8):
    #Removes brightest part of picture. 
    #Higher threshold means less is removed.
    max_val = np.max(image)
    image[image > max_val*percent_threshold] = 0
    plt.imshow(image)
    plt.show()
    return (image)

#-------------------------------------------------------------------------------
def select_on_charge(images, charge, max_charge, min_charge):
    #Using a positive convention for inputs
    max_charge = -max_charge
    min_charge = -min_charge
    
    loc = np.where( (charge[0,:] > max_charge) & (charge[0,:] < min_charge) )
    n_images = len(loc[0])
    print 'Number of data sets in specified range:', n_images
    #print np.shape(images)
    #Getting corresponding images
    print 'Average charge is: ', np.mean(charge[0,loc])
    charge_images = images[:,:,loc[0]]

    return(charge_images, n_images)
#-------------------------------------------------------------------------------
def raw_data_curves(image, oneframe=1 ):
    # At the moment, this function is only finding the fit for one 
    # one frame (frame 1). 
    if oneframe == 1:
       f1 = image
    
    dx, dy = np.shape(image)
    #X fit, one for one sum across lines
    fit_x = np.zeros([dx])
    for i in xrange(0,dx):     
        line = f1[i,:]
        fit_x[i] = np.sum(line)
    
    #Finding y fit

    fit_y = np.zeros([dy])
    for i in xrange(0,dy): 

        line = f1[:,i]
        fit_y[i] = np.sum(line)
         
    return (fit_x, fit_y)

#-------------------------------------------------------------------------------
def fit_data(images, fiducials, key):
    dx, dy, n_images  = np.shape(images)
    sigmax    = np.zeros((n_images))
    sigmay    = np.zeros((n_images))
    fiducial  = fiducials[0][key]
    print 'fiducial:', fiducial
    print np.shape(images)
    beamsizes = {}

    mod = GaussianModel()

    for n in range(0,n_images):
        #getting raw data curves 
        raw_x, raw_y = raw_data_curves(images[:,:,n]) 
        x_points = len(raw_x) #x_max = x_points*fiducial
        y_points = len(raw_y) #y_max = y_points*fiducial
        
        x_axis   = (np.arange(0,x_points) - x_points/2)*fiducial
        y_axis   = (np.arange(0,y_points) - y_points/2)*fiducial
       
        #Calc sigmax 
        parsx = mod.guess(raw_x, x=x_axis)
        outx  = mod.fit(raw_x, parsx, x=x_axis)
        paramsx = outx.best_values
        sigmax[n]  = paramsx['sigma']
        #Calc sigmay
        parsy = mod.guess(raw_y, x=y_axis)
        outy  = mod.fit(raw_y, parsy, x=y_axis) 
        paramsy = outy.best_values
        sigmay[n]  = paramsy['sigma']
         
    print 'sigmax', sigmax
    print 'sigmay', sigmay
    beamsizes['sigmax'] = sigmax 
    beamsizes['sigmay'] = sigmay 
    np.save('beamsizes_'+key+'.npy', beamsizes)

    #mod = GaussianModel()
    #mod  = LorentzianModel()
    #mod = VoigtModel()
    #pars = mod.guess(raw_x, x=x_axis)
    #out  = mod.fit(raw_x, pars, x=x_axis)
    #params = out.best_values
    #sigma  = params['sigma']
    #print sigma
    #print(out.fit_report(min_correl=0.25))

    plt.figure(200)
    plt.plot(x_axis, raw_x,         'bo')
    plt.plot(y_axis, raw_y, 'ko')
    plt.plot(y_axis, outy.best_fit, 'k--')
    plt.plot(x_axis, outx.best_fit, 'b-')
    plt.show()

    #z = np.polyfit(x_axis, raw_x, 30)
    #f = np.poly1d(z)
    #y_new = f(x_axis)
    #popt, pcov = curve_fit(raw_x, x_axis, ydata)
    #plt.figure(300)
    #plt.plot(x_axis, raw_x)
    #plt.plot(x_axis, y_new)
    #plt.show()
    return (beamsizes)
#-------------------------------------------------------------------------------
def crop_image(image, x_min=0, x_max=480, y_min=0, y_max=640):
    #Must be one frame
    #dx, dy = image.shape()    
    cropped = image[x_min:x_max, y_min:y_max]
    #plt.figure(400)
    #plt.imshow(cropped)
    #plt.show()
    return(cropped)
#--------------------------------------------------------------------------------
def add_dist_to_image(crop, fiducial, basename):
   from mpl_toolkits.axes_grid1 import make_axes_locatable

   dx, dy = crop.shape

   xaxis   = (np.arange(0,dx) - dx/2)*fiducial
   yaxis   = (np.arange(0,dy) - dy/2)*fiducial

   fitx, fity = raw_data_curves(crop, oneframe=1)
   fitxnorm = (fitx - np.min(fitx))/(np.max(fitx)-np.min(fitx))#*15 -20  
   fitynorm = (fity - np.min(fity))/(np.max(fity)-np.min(fity))#*15 -20 
 
   plt.close("all")
   fig, ax = plt.subplots(figsize=(10.5, 10.5))
   ax.set_aspect(1.)
   divider = make_axes_locatable(ax)
   axHistx = divider.append_axes("top", 1.25, pad=0.1, sharex=ax)
   axHisty = divider.append_axes("right", 1.25, pad=0.1, sharey=ax)
   # make some labels invisible
   axHistx.xaxis.set_tick_params(labelbottom=False)
   axHisty.yaxis.set_tick_params(labelleft=False)
   axHisty.plot(fitxnorm, -xaxis, linewidth=3)
   axHistx.plot(yaxis, fitynorm, linewidth=3)#, orientation='horizontal') 
 
   cmap = plt.cm.viridis 
   cmap.set_under(color='white')    
   color = ax.imshow(crop, interpolation='none', cmap=cmap, vmin=1, extent=[np.min(xaxis), np.max(xaxis), np.min(yaxis), np.max(yaxis)])
   #ax.plot(xaxis, fitxnorm, '--', linewidth=5, color='firebrick')
   #ax.plot(yaxis, fitynorm, '--', linewidth=5, color='firebrick') 
   ax.tick_params(labelsize=12)
   #axHistx.set_title('YAG 1: Z = 3.1 m', size=20) 
   axHistx.set_title(basename, size=20)
   ax.set_xlabel('X [mm]', size=18)
   ax.set_ylabel('Y [mm]', size=18)
   plt.colorbar(color,ax=ax, orientation="horizontal", shrink=0.7, pad=0.1)
   plt.savefig(basename+'.pdf', dpi=1000, bbox_inches='tight')
   plt.show()


def similarity_check(image_array):
    #http://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html
    Nframes = len(image_array[0,0,:])
    s_ave  = 0
    for i in range(0,Nframes):
        s = ssim(image_array[:,:,0], image_array[:,:,i])
        s_ave = s_ave + s/Nframes 
            
    return s_ave

 
def createCircularMask(h, w, center=None, radius=None):
    #https://stackoverflow.com/questions/44865023/circular-masking-an-image-in-python-using-numpy-arrays
    #mask = createCircularMask(dy, dx, center=[cy,cx], radius=np.mean(radii))

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

 
# mask = rotate>0
# rotate = skimage.transform.rotate(edges, 0.0, resize=True)
# crop = rotate[np.ix_(mask.any(1),mask.any(0))]
# crop2 = image[np.ix_(mask.any(1),mask.any(0))]
#==============================================================================
#==============================================================================
# #nonzeroCols = ~np.all(edges==False, axis=0)
# #nonzeroRows = ~np.all(yag1==False, axis=1)
# #nonzeroCols = (edges==False).all(axis=1)
# 
# topcols = ~np.all(edges[:,0:100]==False, axis=0)
# botcols = ~np.all(edges[:,400:480]==False, axis=0)
# 
# cols = ~np.all(edges==False, axis=0)
# #rows = ~np.all(edges==False, axis=1)
#==============================================================================

#==============================================================================
# cut1 = edges[cols,:]
# cut2 = cut1[:,rows]
#==============================================================================
#==============================================================================
# cut1 = edges[:,rows]
# cut2 = cut1[cols,:]
#==============================================================================

#yag = np.where(nonzeroCols)
#crop = edges[yag]

#hold = edges[yag]
#crop = hold[:, (hold != 0).sum(axis=0) >= 1] 
    #return(mask)

#crop = edgeDetection(imArray)

#==============================================================================
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
#==============================================================================

#plt.imshow(denoise_bilateral(image, multichannel=False))#, sigma_range=0.1, sigma_spatial=15))
#plt.savefig('denoise.pdf')

#==============================================================================
# #==============================================================================
# # mask   = (f1<52)+0.0                  
# # image_result = inpaint.inpaint_biharmonic(f1, mask)#, multichannel=True)
# # i = scipy.ndimage.map_coordinates(z, np.vstack((x,y)))
# #Need to flip x and y values in this array
# #x, y = np.mgrid[640:0:-1, 480:0:-1]
# #plt.pcolor(x,y, f1, cmap='RdBu', vmin=np.min(f1), vmax=np.max(f1))
# #plt.pcolormesh(x,y, f1, cmap='copper', norm=LogNorm(vmin=1, vmax=np.max(f1)))
# 
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





















