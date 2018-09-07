# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 17:08:44 2016
Updated on Thu Sep 28          2017
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
import re
#import os 
from linecache import getline
#import scipy
from scipy.integrate import simps 
import numpy as np
#import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import sys
from itertools import islice 
"""
Code steps: 
    1. Find the number of channels and steps in each data set.
    2. Find the number of data sets in the file.    
    3. Make arrays to hold the data.
    4. Read in the raw data and calibration info. 
    5. Calculate voltages and time steps
    
To use the 
"""
def csv_to_volts_array(ict_file):
    header = 6
    #first occurence of step number per data set
    #This assumes they are the same
    steps = int(getline(ict_file, 4).split(' ')[-1].strip())
   
    #Counting how many data sets
    date = getline(ict_file,2).split(':')[1]
    print('The data was taken on',date)
    with open(ict_file,'r') as f:
        n_data_sets = sum(len(re.findall(date , line)) for line in f)
    f.close()
    print('A total of', n_data_sets, 'data sets were taken')
  
    #Loading data  
    #Making arrays with charge and no text
    #Only one channel possible
    volts_array = np.zeros((steps,n_data_sets))
    time_array  = np.zeros((steps,n_data_sets))
    current_col = 0
    with open(ict_file, 'r') as f:
        for ind, line in enumerate(f,1):
            if date in line:
                #print(date)
                for i in range(0,steps):
                    #Adding 4 skips other header 
                    data = getline(ict_file, ind+5+i).split(',')
                    #print(data[0])#, data[1])
                    time_array[i,current_col]  = float(data[0])
                    volts_array[i,current_col] = float(data[1].strip())
                current_col = current_col +1
 
    return(volts_array, time_array)

def sdds_to_volts_array(ict_file):
    #lines 1-17 Are header. 
    header = 17
    #After the 17 line header
    #   line 18 = date and time
    #   line 19 = data set number i.e: 0-100
    #   line 20 = firt line of data

    #Needed to change function because step numbers 
    #are not longer recorded on line 20 of sdds file
    
    #Get date for looping through data
    #Grab portion of data that will not change (month, day)
    date = (getline(ict_file,header+1))[0:10] #.split(':')
    print("The data was taken on,", date[1:])
   
    #Make empty list for line numbers 
    datelines = []
    #Open file to loop through data
    with open(ict_file) as sddsfile:
        #Find all lines with date on them
        for linenumber, line in enumerate(sddsfile): 
            if date in line:
                datelines.append(linenumber)
    sddsfile.close()

    #Print number of data sets taken:
    shots = len(datelines)
    print(shots, 'data shots were taken.')

    #Calculate how many time steps are in each dataset
    steps = datelines[1]-datelines[0]-3
    print('Each shot has', steps, 'steps.')
    #Make array to hold charge data 
    #(remove header and all other text)
    data = []
    volts = [0]*steps
    volts_array = np.zeros((steps,shots))
    #Loop through each data set (shot)
    for i in range(0,shots):
        #with open(ict_file) as f:
            ##Grab lines with time and voltage data
            #data  = list(islice(f, datelines[i-1]+3, datelines[i]))
            ##Grab only voltage data for shot i
            #volts = [float(element.split('\t')[1]) for element in data]
        #Faster way to read data than above method
        for j in range(0,steps):
            linenumber = datelines[i]+4+j
            #Calculate dT using first and second time step
            if j == 0:
                t0 = getline(ict_file, linenumber).split('\t')[0]
                t1 = getline(ict_file, linenumber+1).split('\t')[0]
                dT = float(t1)-float(t0)
            volts[j] = float(getline(ict_file, linenumber).split('\t')[1])
            
        #Place voltage data in volts_array
        volts_array[:,i] = volts
    
    return(volts_array, dT)


def sdds_to_volts_array_preJuly2018(ict_file): 
    #https://stackoverflow.com/questions/20414989/how-many-times-a-word-occurs-in-a-file?answertab=votes#tab-top   
    #Defining variables for use later in the script
    #lines 1-17 Are header. 
    header = 17
    
    #After the 17 line header
    #   line 18 = date and time
    #   line 19 = data set number i.e: 0-100
    #   line 20 = number of steps and channels 
    #file  = 'WSPE_DT1_WD1_1p60.sdds'
    
    #Loading the number of steps and channels from the 20th line 
    steps_channels = getline(ict_file,header+3)
    steps = 5001
    #steps    = int(steps_channels.split(' ')[0]) 
    #channels = int(steps_channels.split(' ')[1].rstrip()) 
    #print('Number of channels is: ', channels)
    #print('Number of steps is: ', steps) 
    
    date = (getline(ict_file,header+1)).split(':')
    #Getting month, day, year only -> [:3]
    date = ':'.join(date[:3])
    print('Date is: ' , date)
    #The date is repeated before every set of data, 
    #   and the first occurance is on line 18. 
    #   This line is used to find how many data sets are in the file
    
    #Finding the number of datasets using shots as a counter
    with open(ict_file,'r') as f:
        shots = sum(len(re.findall( date, line)) for line in f)
    f.close()
    print('This many shots were taken: ', shots)

    #Making arrays with charge and no text
    #Only one channel possible
    volts_array = np.zeros((steps,shots))
    cal_array   = np.zeros((3,shots))
    current_col = 0
    with open(ict_file, 'r') as f:
        for ind, line in enumerate(f,1):
            if date in line:
                #print date
                for i in range(0,steps):
                    #print(getline(ict_file, ind+3+i))
                    data = float(getline(ict_file, ind+3+i).split('\t')[1])
                    print(data) 
                    volts_array[i,current_col] = data
                ###cal = getline(ict_file, ind+steps+4)#.split()[:-2]
                ###print(cal)
                ####Calibration array has 3 numbers:
                ####0 - deltaT
                ####1 - vertical scaling?
                ####2 - vertical position, not needed?
                ###cal_array[0,current_col] = float(cal[3])
                ###cal_array[1,current_col] = float(cal[4])
                ###cal_array[2,current_col] = float(cal[5])

                #print cal_array[:,current_col]
                current_col = current_col +1    

    #print volts_array.shape
    return(volts_array, cal_array)


def calc_offset(volts, ave_over):
    #Calculating the voltage offset by averaging the first 200 pts
    offset = np.mean(volts[:ave_over])
    #offset = np.mean(volts_array[-ave_over:,n]) #ave on back
    #print 'offset', offset
    #Making sure the voltage offset value is not close to the peak value
    min_val = np.min(volts)
    max_val = np.max(volts)
    if min_val < 0:
        test = np.abs(min_val) - offset
    elif min_val >= 0:
        test = offset - min_val
    #print np.abs(test)
    if (np.abs(test) < 0.0005):
        #Warning message
        print('The offset value is', offset, 'which is close to the max voltage reading', np.min(volts))
        print('If you feel those numbers are acceptable, no need to do anything.')
        print('To double check the zero line, look at the voltage curve, the zero line is plotted in green.')
        print('To adjust the zero line, change the number of points used to calculate the offset.')
        print('Change function input "ave_over" to adjust pts: ict_charge_csv(array, ave_over=NNN)')
    return(offset)


def dT_csv(time_array):
    deltaT = time_array[1,0] - time_array[0,0] 
    return(deltaT)
     
def dT_sdds(cal_array, n):
    deltaT  = cal_array[0,n]
    vscale  = cal_array[1,n] 
    #vposition = cal_array[2,n] #scopes attempt at offset, not needed?
    return(deltaT, vscale) 

def ict_charge(ict_file, data_type='none', ave_over=200, ict_cal=1.25, pre2018=False):
    #Load the voltage data from the ict file
    if data_type=='csv':
        volts_array, time_array = csv_to_volts_array(ict_file) 
    elif data_type=='sdds' and pre2018:
        volts_array, cal_array = sdds_to_volts_array_preJuly2018(ict_file)
    elif data_type=='sdds' and pre2018==False:
        volts_array, dT = sdds_to_volts_array(ict_file)

    steps   = len(volts_array[:,0]) 
    n_shots = len(volts_array[0,:]) 
    charge_array = np.zeros((1,n_shots)) 

    #Pre July 2018 procedure
    if data_type=='csv' or pre2018==True:
        scaled_volts = np.empty_like(volts_array)
        #This for loops over the number of datasets in the file, c. 
        for n in np.arange(0,n_shots): 
            volts  = volts_array[:,n]
            if data_type=='csv':
                deltaT = dT_csv(time_array)
                offset = calc_offset(volts, ave_over)  
                #Calculating the voltage in Volts 
                volts = volts - offset  
            elif data_type=='sdds' and pre2018:
                deltaT, vscale = dT_sdds(cal_array, n)
                offset = calc_offset(volts, ave_over)
                volts = (volts-offset)*vscale #-vposition)*vscale  
            elif data_type=='sdds':
                pass
            else:
                print('Invalid data type, exiting function')
                break
            scaled_volts[:,n]  = volts
            #Calculating the charge over the averaged datasets 
            charge = simps(volts, dx=deltaT)
            #charge = np.trapz(volts, dx=deltaT) 
            charge_array[0,n] = charge*(10**9/ict_cal)

    elif pre2018==False:
        for n in np.arange(0,n_shots):
            charge = simps(volts_array[:,n], dx=dT)
            charge_array[0,n] = charge*(10**9/ict_cal)

            if np.abs(charge_array[0,n]) < 0.2:
                print('Data is very noisy, please look at voltage curve to verify charge for shot:', n, '\n')

    print('Min Charge =', np.max(charge_array),'Max Charge=',  np.min(charge_array))
    print('Std is ', np.std(charge_array), ', Mean is ', np.mean(charge_array))

    if pre2018==True:
        return(charge_array, scaled_volts)
    if pre2018==False:
        return(charge_array, volts_array)
   
 
def plot_ict_curves(scaled_volts, cal=0, base_file='test', n_pdfs=10, time_array=0):
    #Calculating the time steps in seconds
    steps  = len(scaled_volts[:,0])
    try:
        deltaT = cal[0,n_pdfs]
        timesteps = np.arange(0,steps)*deltaT 
    except:
        #print 'Using csv file data'
        timesteps = time_array[:,0]
        deltaT = time_array[1,0] - time_array[0,0]
    
    pdffile = 'ICTcurve_' + base_file +'.pdf'
    print('Making a pdf of the first', n_pdfs, 'shots') 
    with PdfPages(pdffile) as pdf:
        
        for v in range(0,n_pdfs):
             mytitle = 'ICT Voltage Curve '+ str(v) 
             plt.title(mytitle, size=18)
             plt.xlabel('time [s]', size=14)
             plt.ylabel('Voltage [V]', size=14)
             
             plt.plot(timesteps, scaled_volts[:,v])
             plt.plot([0,steps*deltaT], [0,0])
            
             pdf.savefig()
             plt.close()
    plt.close('all') 
