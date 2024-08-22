#note on libraries: can be installed using "pip install $LIBRARY_NAME" aaa

#own files
import TDS_Sim
import Example_Run

#libraries for generating model data
import random
import matplotlib.pyplot as plt
import joblib

#libraries for saving data
import h5py               # library for storing datasets, see https://docs.h5py.org/en/stable/ for examples
import pandas as pd

#libraries for machine learning
import tensorflow as tf   #see https://www.tensorflow.org/tutorials/keras/regression for examples

# Booleans to turn seperate parts on/off (useful for testing)
GenerateData = True
TrainModel = True
VerifyModel = True
FitExperiments = True

# General parameters
NumDataPoints = 1000    #how many samples to generate
n_cpu_cores = 20        #how many cpu cores to run simulations on
DataFileName = "LearningData.hdf5"  #name of file to save data into

# Step 1: Generate a dataset usable for machine learning, and save this to a file
# Notes:  This can use the GenerateDataPoint function within the Example_Run.py file
#         (or variations on this) to generate a single data point (see also the _main_
#         in this file for an example of how to run multiple points in parrallel). To 
#         make it easier to re-run later parts, recommending that this data is saved to
#         a datafile such that the machine learning steps do not need to generate this 
#         data every single time something is changed in the later parts. 
def GenerateLearningData():
    pass

# Step 2: Machine Learning on part of this dataset
# Notes:  Start simple!, for instance first consider a dataset with all the same trapping site
#         concentrations and only train and test the model to identify trapping energies. For
#         the machine learning, likely can use the tensorflow python library, with regression
#         type learning models (e.g. see https://www.tensorflow.org/tutorials/keras/regression). 
#         It's not a linear problem, so non-linear layers (or multiple stacks of layers) will be
#         needed, although for the moment the focus is on obtaining a working model, not on the 
#         efficiency of the model itself, so please feel free to put in as many layers as needed.

#         You might also need to apply some pre-filtering on the data. The dataset initially will
#         contain outputs for each time/temperature increment. Reducing this to a more standard
#         data format (e.g. one data point every 1/5/10 K) will reduce the amount of input variables
#         , while also making application to later experimental curves easier.
def PerformLearning():
    pass

# Step 3: Use trained machine learning model on remainder of dataset, showing applicability
# Notes:  This is showing that the model trained on a large part of the dataset is able to succesfully
#         work on the remainder of the datset. Show this not just in numbers, but more importantly 
#         generate some nice visuals showing how good the model is (also important for presentations/
#         reports).
def VerifyTrainedModel():
    pass

# Step 4: Interface for fitting experimentally obtaiend data
def FitExperimentalData():
    pass

if (GenerateData):
    GenerateLearningData()
if (TrainModel):
    PerformLearning()