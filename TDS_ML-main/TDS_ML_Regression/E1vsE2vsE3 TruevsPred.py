import os
import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import joblib
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

import TDS_Sim
import Example_Run

# Set boolean flags to control code execution
GenerateData = False
TrainModel = True
VerifyModel = True
FitExperiments = True

# General parameters for data generation
NumDataPoints = 3000
n_cpu_cores = 5
DataFileName = "6VarsTrap(RandomEnergy),RandomConc.hdf5"
n_traps = 3

# Path to the data file

file_path = r"C:\Users\miles\OneDrive\Desktop\TDS_ML\TDS_ML\TDS_ML_Regression\6VarsTrap(RandomEnergy),RandomConc.hdf5"

# Helper function to sort list elements by their second element
def takeSecond(elem):
    return elem[1]

# Function to generate a single data point for the simulation
def GenerateDataPoint2(i):
    # Simulation parameters
    PlotWhileSolving = False
    model = "Rates"
    NumTraps = n_traps

    N_Elements = 10
    dT_Step = 120.0

    Thickness = 10.0e-3

    Diff_RT = 1.0e-9
    E_diff = 2 * 4.0e3

    k_T = [1.0e4, 1.0e0]

    time_Charge = 12.0 * 3600.0
    Temp_Charge = 273.15 + 20.0
    pCharge = 1.0e6

    time_rest = 600.0

    TDS_HeatingRate = 1.0 / 60.0
    TDS_Time = 3600.0 * 24

    # Generate trap data
    traps = []
    for t in range(0, NumTraps):
        validPoint = False
        while not validPoint:
            E_abs = 20.0e3
            E_des = random.uniform(40.0e3, 150.0e3)
            N = random.uniform(30.0, 100.0)

            # Ensure generated traps are adequately spaced
            goodDist = True
            for E in traps:
                if abs(E[1] - E_des) < 10e3:
                    goodDist = False
            validPoint = goodDist

        traps.append([E_abs, E_des, N])
        traps.sort(key=takeSecond)

    # Extract concentrations and energy values
    N_traps = []
    E_traps = []
    for t in traps:
        N_traps.append(t[2])
        E_traps.append([t[0], t[1]])

    print(f"{i}:")
    print(f"\t N={N_traps}")
    print(f"\t E={E_traps}")

    # Run the TDS simulation
    Sample = TDS_Sim.TDS_Sample(Thickness, N_Elements, dT_Step, PlotWhileSolving)
    Sample.Set_Params(model, Diff_RT, E_diff, N_traps, E_traps, k_T)
    Sample.Charge(time_Charge, Temp_Charge, pCharge)
    Sample.Rest(time_rest)
    [T, J] = Sample.TDS(TDS_Time, TDS_HeatingRate)

    # Debugging: Print temperature and flux values
    print("Temperature values (T):", T)
    print("Hydrogen Flux values (J):", J)

    return NumTraps, N_traps, E_traps, T, J

# Function to generate learning data by running simulations in parallel
def GenerateLearningData(NumDataPoints, DataFileName, n_cpu_cores):
    Res = joblib.Parallel(n_jobs=n_cpu_cores)(joblib.delayed(GenerateDataPoint2)(i) for i in range(0, NumDataPoints))

    # Extract and organize data
    correct_solution_matrix = []
    for i in range(NumDataPoints):
        if len(Res[i][2]) == 3:
            correct_solution_matrix.append([Res[i][1][0], Res[i][1][1], Res[i][1][2], Res[i][2][0][1], Res[i][2][1][1], Res[i][2][2][1]])
        else:
            print(f"Data point {i} does not have exactly 3 energy values and 3 concentrations and is skipped.")
    correct_solution_matrix = np.array(correct_solution_matrix)
    
    tds_curves_matrix = np.array([Res[i][4] for i in range(0, NumDataPoints)])
    temperature_data = np.array([Res[0][3]])
    concentration_trapping_sites = np.array([Res[i][1] for i in range(0, NumDataPoints)])

    # Save data to a h5py file
    with h5py.File(file_path, 'r') as hf:
        hf.create_dataset('correct_solution_matrix', data=correct_solution_matrix)
        hf.create_dataset('tds_curves_matrix', data=tds_curves_matrix)
        hf.create_dataset('temperature_data', data=temperature_data)
        hf.create_dataset('concentration_trapping_sites', data=concentration_trapping_sites)

# Main execution block to generate data
if __name__ == "__main__":
    if GenerateData:
        GenerateLearningData(NumDataPoints, DataFileName, n_cpu_cores)

    # Load data for preprocessing and model training
    try:
        with h5py.File((file_path), 'r') as hf:
            print("Keys in the file:", list(hf.keys()))  # Print all keys in the HDF5 file

            tds_curves_matrix = hf['tds_curves_matrix'][:]
            correct_solution_matrix = hf['correct_solution_matrix'][:]
            concentration_trapping_sites = hf['concentration_trapping_sites'][:]

            print("TDS Curves Matrix Shape:", tds_curves_matrix.shape)
            print("Correct Solution Matrix Shape:", correct_solution_matrix.shape)
            print("Concentration Trapping Sites Shape:", concentration_trapping_sites.shape)

    except KeyError as e:
        print(f"KeyError: {e}")
        print("Available keys in the file:")
        with h5py.File(DataFileName, 'r') as hf:
            print(list(hf.keys()))

# Downsample the TDS curves matrix for training
tds_curves_matrix_downsampled = tds_curves_matrix[:, ::10]

# Normalize the data
scaler = StandardScaler()
tds_curves_matrix_normalized = scaler.fit_transform(tds_curves_matrix_downsampled)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    tds_curves_matrix_normalized, correct_solution_matrix, test_size=0.2, random_state=42)

# Verify shapes of split data
print("X_train Shape:", X_train.shape)
print("y_train Shape:", y_train.shape)
print("X_test Shape:", X_test.shape)
print("y_test Shape:", y_test.shape)

# Build a neural network model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(6)  # Output layer with 6 neurons for 6 target variables
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])

# Train the model
history = model.fit(X_train, y_train, epochs=500, validation_split=0.2)

# Plot training and validation MAE
plt.figure(figsize=(10, 6))
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error')
plt.title('Training and Validation MAE')
plt.legend()
plt.savefig("6VarsTrap(RandomEnergy),RandomConc_Training_and_Validation_MAE.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")

# Plot training and validation MSE
plt.figure(figsize=(10, 6))
plt.plot(history.history['mse'], label='Training MSE')
plt.plot(history.history['val_mse'], label='Validation MSE')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation MSE')
plt.legend()
plt.savefig("6VarsTrap(RandomEnergy),RandomConc_Training_and_Validation_MSE.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")

# Evaluate the model on the test set
loss, mae, mse = model.evaluate(X_test, y_test)
print(f"Test Mean Absolute Error: {mae}")
print(f"Test Mean Squared Error: {mse}")

# Make predictions on the test set
predictions = model.predict(X_test)

# Print first 10 predictions for verification
print("Predictions:", predictions[:10])

# Extract true and predicted values for plotting
true_C1 = y_test[:, 0]
true_C2 = y_test[:, 1]
true_C3 = y_test[:, 2]
true_E1 = y_test[:, 3]
true_E2 = y_test[:, 4]
true_E3 = y_test[:, 5]
predicted_C1 = predictions[:, 0]
predicted_C2 = predictions[:, 1]
predicted_C3 = predictions[:, 2]
predicted_E1 = predictions[:, 3]
predicted_E2 = predictions[:, 4]
predicted_E3 = predictions[:, 5]

# 3D scatter plot to compare true vs. predicted values for E1, E2, and E3
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(true_E1, true_E2, true_E3, marker='*', label='True Energy Coordinates', color='b')
ax.scatter(predicted_E1, predicted_E2, predicted_E3, marker='o', label='Predicted Energy Coordinates', color='r')

# Draw lines connecting true and predicted values for energies
for i in range(len(true_E1)):
    ax.plot([true_E1[i], predicted_E1[i]], [true_E2[i], predicted_E2[i]], [true_E3[i], predicted_E3[i]], color='gray', linestyle='-', linewidth=0.5)

# Set labels and title for the plot
ax.set_xlabel('E1')
ax.set_ylabel('E2')
ax.set_zlabel('E3')
ax.set_title('True vs Predicted Values for E1, E2, and E3')
ax.legend()
plt.savefig("6VarsE1_vs_E2_vs_E3_True_vs_Predicted.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")

# 3D scatter plot to compare true vs. predicted values for C1, C2, and C3
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(true_C1, true_C2, true_C3, marker='*', label='True Concentration Coordinates', color='b')
ax.scatter(predicted_C1, predicted_C2, predicted_C3, marker='o', label='Predicted Concentration Coordinates', color='r')

# Draw lines connecting true and predicted values for concentrations
for i in range(len(true_C1)):
    ax.plot([true_C1[i], predicted_C1[i]], [true_C2[i], predicted_C2[i]], [true_C3[i], predicted_C3[i]], color='gray', linestyle='-', linewidth=0.5)

# Set labels and title for the plot
ax.set_xlabel('C1')
ax.set_ylabel('C2')
ax.set_zlabel('C3')
ax.set_title('True vs Predicted Values for C1, C2, and C3')
ax.legend()
plt.savefig("6VarsC1_vs_C2_vs_C3_True_vs_Predicted.png", dpi=600)
plt.show()
input("Press Enter to continue (closes all figures)...")