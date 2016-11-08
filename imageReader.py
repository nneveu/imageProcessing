# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 09:57:59 2016

@author: nneveu

Load YAG screen images.

"""
from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt
from skimage.restoration import denoise_tv_chambolle, denoise_bilateral
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
from lmfit.models import GaussianModel
from matplotlib.colors import LogNorm
from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max, canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import skimage 

def readimage(imagefile):
    
    images  = np.fromfile(imagefile, dtype=np.uint16, count=-1,sep='')
    # header info vert/horiz pixels and number of frames
    dx      = int(images[0])
    dy      = int(images[1])
    Nframes = int(images[2])
    hold    = images[4:] # skipping header info
    #==========================================================
    #Reading images into 3D array 
    imagesArray = np.reshape(hold,(dx, dy, -1), order='F')

    return(dx, dy, Nframes, imagesArray)    
    
    
def fit(imagesArray, dx, dy, oneframe=1 ):
    # At the moment, this function is only finding the fit for one 
    # dimension (y), and one frame (frame 1). 
    #f1 = frames[200:320,110:400,0]
    if oneframe == 1:
        #f1 = imagesArray[:,:,0]
        f1 = imagesArray
        
    #X fit, one for one sum across lines
    fit_x = np.zeros([dx])
    for i in xrange(0,dx):     
        line = f1[i,:]
        fit_x[i] = np.sum(line)
    
    #Finding y fit
    #Need to adjust sum bc interlaced in Y dim
    #Grabbing only the columns with data
    #yframe = f1[:, 1::2]

    #plt.imshow(yframe)
    fit_y = np.zeros([dy])#/2])
    for i in xrange(0,dy):#/2):     
         #line = f1[:,i]
        #line = yframe[:,i] 
        line = f1[:,i]
        fit_y[i] = np.sum(line)
         
    return (fit_x, fit_y)
    #plt.plot(fity, '-')

#def edgeDetection(imagesArray):
#Only looks at one image right now

#return (crop)


#==============================================================================
# Main, calling functions    
#==============================================================================
testfile1  = '/Users/nneveu/Documents/DATA/TBA(1-13-16)/DriveOn_WSPE2_WD1_1p756.dat'
testfile2 = '/Users/nneveu/Documents/DATA/TBA(1-13-16)/DriveOff_WSPE2_WD1_1p51.dat'

(dx, dy, Nframes, imArray) = readimage(testfile2)
#print "Dx,Dy,NFrames= ",dx,dy,Nframes

#crop = edgeDetection(imArray)

image = imArray[:,:,0]    
edges = canny(image, sigma=5, low_threshold=50, high_threshold=120)
rotate = skimage.transform.rotate(edges, 0.0, resize=True)
#edges = roberts(image)
#edges = sobel(image)

mask = rotate>0
crop = rotate[np.ix_(mask.any(1),mask.any(0))]
crop2 = image[np.ix_(mask.any(1),mask.any(0))]
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
plt.imshow(crop, cmap='copper')
plt.figure()
plt.imshow(crop2)#, cmap='copper')#, norm=LogNorm())
plt.savefig('test.pdf')
plt.figure()
plt.imshow(image, cmap='copper')

#Plotting distribution of beam 
#==============================================================================
#(fit_x, fit_y) = fit(imArray, dx, dy, oneframe=1)
#(fit_x, fit_y) = fit(crop, len(crop[:,0]), len(crop[0,:]), oneframe=1)
# plt.figure()
# plt.plot(fit_x, 'g', label= 'x fit')
# plt.legend()
# plt.figure()
# plt.plot(fit_y, label = 'y fit')
# plt.legend()
# plt.figure()
#==============================================================================

#print f1.min(), f1.max(), f1.mean()

#edges = sobel(image)  
#==============================================================================
# fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))
# 
# # Detect two radii
# hough_radii = np.arange(10, 480, 180)
# hough_res = hough_circle(edges, hough_radii)
# 
# centers = []
# accums = []
# radii = []
# 
# for radius, h in zip(hough_radii, hough_res):
#     # For each radius, extract two circles
#     num_peaks = 2
#     peaks = peak_local_max(h, num_peaks=num_peaks)
#     centers.extend(peaks)
#     accums.extend(h[peaks[:, 0], peaks[:, 1]])
#     radii.extend([radius] * num_peaks)
# 
# # Draw the most prominent 5 circles
# #image = color.gray2rgb(image)
# for idx in np.argsort(accums)[::-1][:5]:
#     center_x, center_y = centers[idx]
#     radius = radii[idx]
#     cx, cy = circle_perimeter(center_y, center_x, radius)
#     image[cy, cx] = (220, 20, 20)
# 
# ax.imshow(image, cmap=plt.cm.gray)
#==============================================================================

#plt.imshow(denoise_bilateral(image, multichannel=False))#, sigma_range=0.1, sigma_spatial=15))
#plt.savefig('denoise.pdf')


#==============================================================================
# REALLY GOOD EDGE DETECTION 
# image = f1
# edge_roberts = roberts(image)
# edge_sobel = sobel(image)
# 
# fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(8, 4))
# 
# ax[0].imshow(edge_roberts, cmap=plt.cm.gray)
# ax[0].set_title('Roberts Edge Detection')
# 
# ax[1].imshow(edge_sobel, cmap=plt.cm.gray)
# ax[1].set_title('Sobel Edge Detection')
# 
# for a in ax:
#     a.axis('off')
# 
# plt.tight_layout()
# plt.savefig('test2.pdf')
#==============================================================================


#==============================================================================
# mask   = (f1<52)+0.0                  
# image_result = inpaint.inpaint_biharmonic(f1, mask)#, multichannel=True)
#==============================================================================

#==============================================================================
#zi = scipy.ndimage.map_coordinates(z, np.vstack((x,y)))
#Using inpaint to get rid of interlacing
#mask   = (f1<160)+0.0                  
#image_result = inpaint.inpaint_biharmonic(f1, mask)#, multichannel=True)
# plt.imshow(f1)
# plt.savefig('test1.pdf')
#==============================================================================
#plt.imshow(f1, interpolation='mitchell')  

#Need to flip x and y values in this array
#x, y = np.mgrid[640:0:-1, 480:0:-1]
#plt.pcolor(x,y, f1, cmap='RdBu', vmin=np.min(f1), vmax=np.max(f1))
#plt.pcolormesh(x,y, f1, cmap='copper', norm=LogNorm(vmin=1, vmax=np.max(f1)))

#plt.imshow(f1, cmap='RdBu')


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





















