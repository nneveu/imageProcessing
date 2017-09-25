# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:08:44 2016
@author: nneveu

Some notes: 
    A. This code depends heavily on the SDDS file format.
       If any thing has changed in the file format, this 
       script may not work.
    B. The file must be in the same directory as the scritpt.
       Otherwise, input the full pathname to to the function.
    C. This function returns the voltage values, and calculates the charge.
       PDF's showing the voltage curves will be made automatically. 
       It is a good idea to check the voltage curves after using this scrtipt.
    D. Charge and voltage information is printed to screen, and saved to a 
       output variables for your use. 
"""

#import os 
import linecache
#import scipy
#from scipy.integrate import simps 
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

"""
Code steps: 
    1. Find the number of channels and steps in each data set.
    2. Find the number of data sets in the file.    
    3. Make arrays to hold the data.
    4. Read in the raw data and calibration info. 
    5. Calculate voltages and time steps
    
To use the 
"""

def ICTcharge(file):    
    #Defining variables for use later in the script
    #lines 1-17 Are header. 
    c=0
    header = 17
    
    #After the 17 line header
    #   line 18 = date and time
    #   line 19 = data set number i.e: 0-100
    #   line 20 = number of steps and channels 
    #file  = 'WSPE_DT1_WD1_1p60.sdds'
    
    
    stepsAndchannels = linecache.getline(file,header+3)
    #Loading the number of steps and channels from the 20th line
    #   The number of steps is the first 4 characters in the line
    #   The number of channels is the 5th character in the line
    steps = int(stepsAndchannels[0:4])
    channels = int(stepsAndchannels[5])
    
    
    date = linecache.getline(file,header+1)
    #The date is repeated before every set of data, 
    #   and the first occurance is on line 18. 
    #   This line is used to find how many data sets are in the file
    #Finding the number of datasets using c as a counter
    f = open(file, 'r')
    for line in f:
        if date[0:10] in line:
            c = c+1
    f.close()
     
    #Now we have the information needed to define arrays
    #   that will hold all the data we read in.
    #   There will be an array for each channel 
    #   And each column will be 1 set of data. 
    #   So in the case of 1002 steps and 95 data sets, 
    #   The arrays will be 1002 rows X 95 columns.
    #   Arrays will be stored in dictionaries, i.e. voltagedata. 
    #   If you want ch1 data, you can call it with the key 'ch1', 
    #   and likewise, with 'ch2'.
    voltagedataRaw = {} 
    voltagedataVolts = {}
    calibration = {}
    charge = {}

    myoffset = {}
    verticalscale = {}
    voltagedataMean = {}  
    verticalOffsetVoltage = {}
    verticalPositionWaveform = {}
    
    for i in np.arange(0,channels):
        key = 'ch'+str(i+1)
        voltagedataRaw[key]     = np.zeros([steps,c])
        voltagedataVolts[key]   =np.zeros([c])
        calibration[key]        =np.zeros([c])   
        charge[key]             =np.zeros([1])
        
        verticalOffsetVoltage[key]      =np.zeros([c]) 
        verticalPositionWaveform[key]   =np.zeros([c]) 
        verticalscale[key]              =np.zeros([c])
        myoffset[key]                   =np.zeros([c])
        voltagedataMean[key]            =np.zeros(steps)
    
    #Now we know how many data sets are in the file,
    #   and we have defined arrays to hold the data. 
    #   Next is To read in and calculate the voltages and charge. 
    
    #This for loops over the number of datasets in the file, c.
    for n in np.arange(0,c):
        
        #Reading calibration info for the current data set
        #First, the line holding all the calibration info is stored
        #Then, by knowing the data locations do not change, we can 
        #extract the data we want by indexing.
        
        #This loop saves the calibration data locations for each channel
        #    location = header + number of steps * number of data sets
        #          + 4*(n+1) data set info lines
        #          + channels*(n+1) to pass up other calibration data
        #          + (i-1) to get correct channel number
        
        for ch in np.arange(0,channels):
            key = 'ch'+str(ch+1)
            
            #Finding the calibration data location
            calibration[key][n] = header +(n+1)*(steps)+ (n+1)*4 + channels*n + (ch+1)
            #Reading the Calibration data, and separating it            
            calibrationstring   = linecache.getline(file,int(calibration[key][n]))
            calibrationstring   = calibrationstring.split()
            #triggerpos    = float(calibrationstring[7:19])
            deltaT        = float(calibrationstring[3])
            verticalscale[key][n] = float(calibrationstring[4])
            verticalPositionWaveform[key][n] = float(calibrationstring[5])
            verticalOffsetVoltage[key][n]    = float(calibrationstring[6])
            #print verticalPositionWaveform
            
            #Finding where the data starts for this set
            #Each file consist of:
            #   17 header line
            #   3 info lines about the data set
            #   Data lines = number of steps
            #   3 calibration lines
            #Therefore, to get to the start of the data
            #   start = header line + 3*(n+1) info lines per data set
            #           + (channels+1)*n calibration lines per data set
            #           + 1 to get to next line, which is data
            start = header+steps*n+3*(n+1)+(channels+1)*n+1   
            stop  = start+steps
            
            #Grabbing the raw data for dataset n then saving to array
            for linenumber in np.arange(start,stop):
                
                #Setting up a counter to cycle through the data points                
                vstep = linenumber-start
                #Reading the voltages
                datastring = linecache.getline(file,linenumber)
                datastring = datastring.split()
                #Storing the raw voltage data
                voltagedataRaw[key][vstep,n]=float(datastring[ch]) 
            
            #Calculating the voltage offset by averaging the first 300 pts.
            npointsOffset = 100
            myoffset[key][n] = np.mean(voltagedataRaw[key][0:npointsOffset,n])
            
            #Making sure the voltage offset value is not close to the peak value 
            test = np.abs(np.min(voltagedataRaw[key][:,n]))-np.abs(myoffset[key][n])
            if (np.abs(test) < 0.05):
                #Warning message
                print 'The offset value is', myoffset[key][n], 'which is close to the max voltage reading', np.min(voltagedataRaw[key][:,n])
                print 'If you feel those numbers are acceptable, no need to do anything.'
                print 'To double check the zero line, look at the voltage curve, the zero line is plotted in green.' 
                print 'To adjust the zero line, change the number of points used to calculate the offset.'
                print 'See line 160 of the script ReadScopeTracesSDDSAvg.py to adjust this  number.'
    
    print 'For file', file
    for key in voltagedataVolts: 
        
        #Calculating the voltage in Volts
        voltagedataVolts[key] = (voltagedataRaw[key]-verticalPositionWaveform[key]-myoffset[key])*verticalscale[key]
        
        #Calculating the mean of the voltage data
        for datapoint in np.arange(0,steps):
            voltagedataMean[key][datapoint]=np.mean(voltagedataVolts[key][datapoint,:] ) 
           
        #Calculating the charge over the averaged datasets
        charge[key] = np.trapz(voltagedataMean[key],dx=deltaT)
        #charge[key][0] = simps(voltagedataMean[key],dx=deltaT)
        charge[key] = (charge[key])*(10**9/1.25)
        print 'Charge of', key, '=', charge[key]
           
    
    if np.abs(charge[key]) < 0.2:
        print 'Data is very noisy, please look at voltage curve to verify charge for:', key
        
    print '\n'
    #closing the file
    linecache.clearcache()  
    
    #Calculating the time steps in seconds
    timesteps = np.arange(0,steps)*deltaT
    file1 = file.split('\\')[1]
    filepdf = file1.split('.')[0]
    pdffile = 'ICTcruve' + filepdf +'.pdf'
    
    with PdfPages(pdffile) as pdf:
        
        for key in charge:
             mytitle = 'ICT Voltage Curve '+ key
             plt.title(mytitle, size=18)
             plt.xlabel('time [s]', size=14)
             plt.ylabel('Voltage [V]', size=14)
             

             plt.plot(timesteps, voltagedataMean[key])
             plt.plot([0,steps*deltaT], [0,0])
            
             pdf.savefig()
             plt.close()
    
    
    return (voltagedataMean, charge)