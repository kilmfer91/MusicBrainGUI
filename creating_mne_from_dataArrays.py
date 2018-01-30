# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:43:53 2017

@author: Fercho
"""

import numpy as np
import math 
import matplotlib.pyplot as plt
import mne
from mne import io, EvokedArray

from mne.decoding import Vectorizer, get_coef

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# import a linear classifier from mne.decoding
from mne.decoding import LinearModel


'''************************
**** Main parameters'''#***
sfreq = 128
seconds = 117
tmin = 10.0
tmax = 20.0
'''************************
************************'''


times = np.arange(0,seconds,1./sfreq )
envelop = np.arange(0,seconds,1./sfreq)

#'''
#A dataset of EEG signals
title = "YS_sin_actividad"
dataset = np.loadtxt(title + ".csv",delimiter=",",usecols=[1,3,5,7,9,11,13,15,17,19,21,23,25,27]) 

# split into input (X) and output (Y) variables
Xtrain = np.transpose(dataset[:,0:14])

#Creating the montage for the neuroheadset
ch_names = ['F3','FC5','F7','T7','P7','O1','O2','P8','T8','F8','AF4','FC6','F4','AF3']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
montage = mne.channels.read_montage('standard_1020',ch_names = ch_names)

info = mne.create_info(ch_names = ch_names, sfreq = sfreq,ch_types = ch_types, montage = montage)

#Creating a dummy mne.io.RayArray object
raw = mne.io.RawArray(Xtrain,info)
raw.filter(.5, 60, fir_design='firwin')

#***********************
#WINDOWING THE RAW DATA
duration = 2.
event_id = 1
events = mne.make_fixed_length_events(raw,event_id,duration = duration)
for i in range(0,len(events)):
    if i % 2 == 0:
        events[i,2] = 3


#Create :class:'Epochs <mne.Epochs>' object
epochs = mne.Epochs(raw,events = events, event_id = event_id, tmin = tmin,
                    tmax = tmax, baseline = None, verbose = True,preload=True)
for i in range(0,len(epochs.events)):
    if i % 2 == 0:
        epochs.events[i,2] = 3

#*********************
#PLOTTING THE ARRAY
start = tmin * 1000
end = tmax * 1000
times = np.arange((start * 1.0)/sfreq,(end * 1.0)/sfreq,1.0/sfreq)
plt.plot(times,Xtrain[:,start:end].T)

#'''  
#Plotting some topographic images    
iniSeg = int(tmin)
finSeg = int(tmax)
for i in range(iniSeg,finSeg):
    
    tmin = i * 1.0
    tmax = (i + 1) * 1.0
    
    #Create :class:'Epochs <mne.Epochs>' object
    epochs = mne.Epochs(raw,events = events, event_id = event_id, tmin = tmin,
                        tmax = tmax, baseline = None, verbose = True,preload=True)
    for i in range(0,len(epochs.events)):
        if i % 2 == 0:
            epochs.events[i,2] = 3
    #epochs.plot(scalings = 'auto',block = True,n_epochs=10)
    X = epochs.pick_types(meg=False, eeg=True)
    y = epochs.events[:, -1]
    
    # Define a unique pipeline to sequentially:
    clf = make_pipeline(
        Vectorizer(),                       # 1) vectorize across time and channels
        StandardScaler(),                   # 2) normalize features across trials
        LinearModel(LogisticRegression()))  # 3) fits a logistic regression
    clf.fit(X, y)
    
    
    coef = get_coef(clf, 'patterns_', inverse_transform=True)
    evoked = EvokedArray(coef, epochs.info, tmin=epochs.tmin)
    fig = evoked.plot_topomap(title='EEG Patterns', size = 3, show=False)
    fig.savefig(title + "_ti_" + str(tmin) + "_tf_" + str(tmax) +  '.png')

