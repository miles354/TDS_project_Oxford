# Libraries and imports to carry out machine learning, data storage, manipulation, visualisation and analysis

# Own files
import TDS_Sim
import Example_Run

# Libraries for generating model data
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib

# Libraries for saving data
import h5py

# Libraries for machine learning
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Boolean functions to turn on/off separate parts of the code (useful for testing)
GenerateData = True
TrainModel = True
VerifyModel = True
FitExperiments = True

# General parameters
NumDataPoints = 10 # Number of samples to generate
n_cpu_cores = 10 # Number of cores running simulations in parallel
DataFileName = "1Trap(RandomEnergy),FixedConc.hdf5" # Name of file to save data into
n_traps = 1 # Number of traps

# Path to the data file

file_path = r"C:\Users\miles\OneDrive\Desktop\TDS_ML\TDS_ML\TDS_ML_Regression\1Trap(RandomEnergy),FixedConc.hdf5"
# Helper function - returns the second element of the list
def takeSecond(elem):
    return elem[1] 

# Data generation function
def GenerateDataPoint2(i):
    PlotWhileSolving = False # Whether to generate plots or not when simulating
    model = "Rates" # Using Mcnabb-Foster Rates equations
    NumTraps = n_traps # Number of traps

    N_Elements = 10 # For obtaining the solution of number of linear finite elements used
    dT_Step = 120.0 # Time increments (s)

    Thickness = 10.0e-3 # Sample Thickness

    Diff_RT = 1.0e-9 # Diffusion Rate at Room Temperature
    E_diff = 2 * 4.0e3 # Diffusion Energy (Useful for Teemperature Scaling)

    k_T = [1.0e4, 1.0e0] # Boundary Flux Desorption and Absorption rates-respectively

    time_Charge = 12.0 * 3600.0 # Time spends charging the sample (s)
    Temp_Charge = 273.15 + 20.0 # Temperature of sample while charging (K)
    pCharge = 1.0e6 # Hydrogen fugacity while charging (Pa)

    time_rest = 600.0 # Time spent being transferred from charging to TDS

    TDS_HeatingRate = 1.0 / 60.0 # Heating rate during TDS
    TDS_Time = 3600.0 * 24 # Time spent over which TDS is performed

    # Generates the trapping energies
    traps = []
    for t in range(0, NumTraps):
        validPoint = False
        while not validPoint:
            E_abs = 20.0e3
            E_des = random.uniform(40.0e3, 150.0e3)
            N = 30.0# Randomised H trapping site concentration of sample 

            # Ensures there is a minimal difference between the Desorption and Absorption Energies
            goodDist = True
            for E in traps:
                if abs(E[1] - E_des) < 10e3:
                    goodDist = False
            validPoint = goodDist

        traps.append([E_abs, E_des, N])
        traps.sort(key=takeSecond)

    # Saves trapping sites concentrations and energies as vectors
    N_traps = []
    E_traps = []
    for t in traps:
        N_traps.append(t[2])
        E_traps.append([t[0], t[1]])

    print(str(i) + ":")
    print("\t N=" + str(N_traps))
    print("\t E=" + str(E_traps))

    # Performs the complete TDS (Charging, Resting, TDS) simulation on the sample
    Sample = TDS_Sim.TDS_Sample(Thickness, N_Elements, dT_Step, PlotWhileSolving)
    Sample.Set_Params(model, Diff_RT, E_diff, N_traps, E_traps, k_T)
    Sample.Charge(time_Charge, Temp_Charge, pCharge)
    Sample.Rest(time_rest)
    [T, J] = Sample.TDS(TDS_Time, TDS_HeatingRate)

    # Debugging: Print temperature and flux values
    print("Temperature values (T):", T)
    print("Hydrogen Flux values (J):", J)

    return NumTraps, N_traps, E_traps, T, J

# Runs simulations in parallel using multiple CPU cores
def GenerateLearningData(NumDataPoints, DataFileName, n_cpu_cores):
    Res = joblib.Parallel(n_jobs=n_cpu_cores)(joblib.delayed(GenerateDataPoint2)(i) for i in range(0, NumDataPoints))

    # Extracts data from results
    correct_solution_vector = np.array([Res[i][2][0][1] for i in range(0, NumDataPoints)])
    tds_curves_matrix = np.array([Res[i][4] for i in range(0, NumDataPoints)])
    temperature_data = np.array([Res[0][3]])
    concentration_trapping_sites = np.array([Res[i][1] for i in range(0, NumDataPoints)])

    # Saves data into a h5py file
    with h5py.File(DataFileName, 'w') as hf:
        hf.create_dataset('correct_solution_vector', data=correct_solution_vector)
        hf.create_dataset('tds_curves_matrix', data=tds_curves_matrix)
        hf.create_dataset('temperature_data', data=temperature_data)
        hf.create_dataset('concentration_trapping_sites', data=concentration_trapping_sites)

# Checks if the Boolean Function 'GenerateData is true/false', if so- will generate the synthetic TDS Curves
if __name__ == "__main__":
    if GenerateData:
        GenerateLearningData(NumDataPoints, DataFileName, n_cpu_cores)

        with h5py.File(file_path, 'r') as hf:
            loaded_correct_solution_vector = hf['correct_solution_vector'][:]
            loaded_tds_curves_matrix = hf['tds_curves_matrix'][:]
            loaded_temperature_data = hf['temperature_data'][:]
            loaded_concentration_trapping_sites = hf['concentration_trapping_sites'][:]

        # Plots simulated TDS Curves using Hydrogen Flux, as a function to the Temperature Ramping
        fig = plt.figure(figsize=(10, 6))
        for i in range(0, NumDataPoints):
            plt.plot(loaded_temperature_data[0], loaded_tds_curves_matrix[i], label=f'Sample {i+1}')
        plt.xlabel('Temperature (K)')
        plt.ylabel('Hydrogen Flux (ppm ms^-1)')
        plt.title('Synthetic TDS Curves')
        plt.legend()
        plt.savefig("1Trap(RandomEnergy),FixedConc.png", dpi=600)
        plt.show()
        input("Press Enter to continue (closes all figures)...")  

        print("Correct solution vector:", loaded_correct_solution_vector)
        print("TDS curves matrix:", loaded_tds_curves_matrix)
        print("Temperature data:", loaded_temperature_data)
        print("Concentration trapping sites:", loaded_concentration_trapping_sites)

def GenerateLearningData():
    pass

# Loads vector data from h5py file
with h5py.File(file_path, 'r') as hf:
    tds_curves_matrix = hf['tds_curves_matrix'][:]
    correct_solution_vector = hf['correct_solution_vector'][:]
    concentration_trapping_sites = hf['concentration_trapping_sites'][:]

# Verify data shapes and content
print("TDS Curves Matrix Shape:", tds_curves_matrix.shape)
print("Correct Solution Vector Shape:", correct_solution_vector.shape)
print("Concentration Trapping Sites Shape:", concentration_trapping_sites.shape)

# Preprocess data (downsample to every 10th data point)
tds_curves_matrix_downsampled = tds_curves_matrix[:, ::10]

# Normalize the data
scaler = StandardScaler()
tds_curves_matrix_normalized = scaler.fit_transform(tds_curves_matrix_downsampled)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    tds_curves_matrix_normalized, correct_solution_vector, test_size=0.2, random_state=42)

# Build a more complex neural network model for non-linear regression
model = models.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(512, activation='relu'),
    layers.Dense(256, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1)  # Single output neuron for regression
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2)

# Plot training & validation MAE values
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()
plt.savefig("1Trap(RandomEnergy),FixedConcTraining_and_Validation_MAE.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")

# Plot training & validation MSE values
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE')
plt.legend()
plt.savefig("1Trap(RandomEnergy),FixedConcTraining_and_Validation_MSE.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")

# Evaluate the model
loss, mae, mse = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")
print(f"Test Mean Squared Error: {mse}")

# Make predictions
predictions = model.predict(X_test)

# Check if predictions are non-zero
print("Predictions:", predictions[:10])

# Plot true vs predicted values
plt.figure(figsize=(10, 6))

# Scatter plot of true vs predicted values
plt.scatter(y_test, predictions, label='Predicted vs True Values')

# Plot a line y=x for reference
x = np.linspace(min(y_test), max(y_test), 100)
plt.plot(x, x, color='red', linestyle='--', label='Ideal Fit')

plt.xlabel('True Values (E_des)')
plt.ylabel('Predicted Values (E_des)')
plt.title('True vs Predicted Values')
plt.legend()
plt.savefig("1Trap(RandomEnergy),FixedConcTruevsPred.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")  

# Prints an R^2 score to evaluate the accuracy of my model
R2 = "R2 from simulations+learning: " + str(r2_score(y_test, predictions))
print(R2)

def PerformLearning():
    pass

# Step 3: Use trained machine learning model on remainder of dataset, showing applicability
# Notes:  This is showing that the model trained on a large part of the dataset is able to succesfully
#         work on the remainder of the datset. Show this not just in numbers, but more importantly 
#         generate some nice visuals showing how good the model is (also important for presentations/
#         reports).
def VerifyTrainedModel():
    pass

# Step 4: Interface for fitting experimentally obtained data
def FitExperimentalData():
    pass

if GenerateData:
    GenerateLearningData()
if TrainModel:
    PerformLearning()