# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 09:57:59 2016

@author: nneveu

Load YAG screen images.

"""
import numpy as np
import matplotlib.pyplot as plt
from skimage.restoration import inpaint
#import cv2

testfile = '/Users/nneveu/Documents/DATA/TBA(1-13-16)/DriveOn_WSPE2_WD1_1p756.dat'
testfile2 = '/Users/nneveu/Documents/DATA/TBA(1-13-16)/DriveOff_WSPE2_WD1_1p51.dat'

#def read_data(filename, width, height):
#==============================================================================
# statinfo = os.stat(testfile)
# print statinfo
#==============================================================================
#==============================================================================
# with open(testfile2, "rb") as f:
#     byte = f.read(1)
#     #while byte != "":
#     for i in xrange(0,10):
#         if byte == "":
#             print 'empty'
#         # Do stuff with byte.
#         byte = f.read(1)
#         print ord(byte)
#         #print repr(byte)
#==============================================================================


im = np.fromfile(testfile2, dtype=np.uint16, count=-1,sep='')
dx      = im[0]
dy      = im[1]
NFrames = im[2] # header info vert/horiz pixels and num frames

#print "Dx,Dy,NFrames= ",dx,dy,NFrames

hold   = im[4:]
frames = np.reshape(hold,(dx, dy, -1), order='F')
f1   = frames[200:320,110:400,0]
#print f1.min(), f1.max(), f1.mean()
mask = (f1<160)+0.0

#==============================================================================
                   
image_result = inpaint.inpaint_biharmonic(f1, mask)#, multichannel=True)



#==============================================================================
# plt.imshow(f1)
# plt.savefig('test1.pdf')
#==============================================================================
#plt.imshow(f1, interpolation='mitchell')  

#
plt.imshow(image_result)
plt.savefig('test2.pdf')




























