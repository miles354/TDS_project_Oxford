import TDS_Sim
import numpy as np
import random
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize, StandardScaler
import seaborn as sns
from itertools import cycle

# STEP 1: GENERATE SYNTHETIC TDS DATA
# Settings to use when this is the main file
NumSamples = 4000  # How many samples to generate
n_cpu_cores = 5  # How many CPU cores to run simulations on

def takeSecond(elem):
    return elem[1]

if __name__ == "__main__":
    random.seed(42)

# Generates a single TDS curve based on (mostly) random binding energies and trapping site concentration
def GenerateDataPoint(i):
    PlotWhileSolving = False  # Whether to generate plots when simulating (much slower)
    model = "Rates"  # Currently using Mcnabb-foster
    NumTraps = random.randint(1, 3)  # Number of trapping sites

    N_Elements = 10  # For obtaining the solution, number of linear finite elements used
    dT_Step = 120.0  # Time increment used

    Thickness = 10.0e-3  # Sample thickness

    Diff_RT = 1.0e-9  # Diffusion rate at room temperature
    E_diff = 2*4.0e3  # Diffusion energy (used for temperature scaling)

    k_T  = [1.0e4, 1.0e0]  # Boundary flux absorption-desorption rates

    time_Charge = 12.0*3600.0  # Time left charging
    Temp_Charge = 273.15+20.0  # Temperature at which samples are charged
    pCharge = 1.0e6  # Hydrogen fugacity during charging

    time_rest = 600.0  # Time sample is being transferred from charging to TDS

    TDS_HeatingRate = 1.0/60.0  # Heating rate of TDS
    TDS_Time = 3600.0*24  # Total time over which TDS is performed

    # Generate trapping energies
    traps = []
    for t in range(0, NumTraps):
        validPoint = False
        while not validPoint:
            E_abs = 20.0e3
            E_des = random.uniform(40.0e3, 150.0e3)
            N = random.uniform(30.0, 100.0)

            # Make sure energies are slightly distinct
            goodDist = True
            for E in traps:
                if abs(E[1] - E_des) < 10e3:
                    goodDist = False
            validPoint = goodDist

        traps.append([E_abs, E_des, N])
        traps.sort(key=takeSecond)

    # Save trapping sites and energies as vectors
    N_traps = []
    E_traps = []
    for t in traps:
        N_traps.append(t[2])
        E_traps.append([t[0], t[1]])

    print(str(i) + ":")
    print("\t N=" + str(N_traps))
    print("\t E=" + str(E_traps))

    # Perform TDS experiment within simulation
    Sample = TDS_Sim.TDS_Sample(Thickness, N_Elements, dT_Step, PlotWhileSolving)  # Initializes material
    Sample.Set_Params(model, Diff_RT, E_diff, N_traps, E_traps, k_T)  # Sets material parameters
    Sample.Charge(time_Charge, Temp_Charge, pCharge)  # Performs charging
    Sample.Rest(time_rest)  # Leave at atmospheric pressure
    T, J = Sample.TDS(TDS_Time, TDS_HeatingRate)  # Perform TDS

    return NumTraps, N_traps, E_traps, T, J  

# Test function to check if all works
if __name__ == "__main__":
    # Run simulations in parallel
    Sample_NTraps = []  # Number of traps for each sample
    SampleResults = []  # TDS results (Temperature and Flux) for each sample
    Sample_E_Traps = []

    # Generate data points in parallel using joblib to speed up the process
    Res = joblib.Parallel(n_jobs=n_cpu_cores)(joblib.delayed(GenerateDataPoint)(i) for i in range(NumSamples))
    print(f"Number of samples in Res: {len(Res)}")#
    for i, sample in enumerate(Res):
       print(f"Sample {i}:", sample)

    # Save to data structure
    for i in range(0, NumSamples):
        Sample_NTraps.append(Res[i][0])
        Sample_E_Traps.append(Res[i][2])
        SampleResults.append([Res[i][3], Res[i][4]])

    Features = []  # Features extracted from the TDS curves for model training
    Labels = []  # Labels (number of traps) for classification

    # After data generation
    print(f"Generated {len(Sample_NTraps)} samples.")
    print(f"Generated {len(Sample_E_Traps)} trap energies.")
    print(f"Generated {len(SampleResults)} TDS results.")

# STEP 2: TRAIN RANDOM FOREST CLASSIFIER TO PREDICT NUMBER OF TRAPS
    # Process the results and extract features and labels
    for i in range(NumSamples):
        # Append the number of traps for each sample
        Labels.append(Sample_NTraps[i])
        # Append the TDS curve (Temperature and Flux)
        T = np.interp(np.linspace(0, len(SampleResults[i][0])-1, num=100), np.arange(len(SampleResults[i][0])), SampleResults[i][1])
        Features.append(T)

    print(Features)

    # Convert lists to numpy arrays for machine learning processing
    Features = np.array(Features)
    Labels = np.array(Labels)

    print("Features shape:", Features.shape)
    print("Labels shape:", Labels.shape)

    # Convert the padded features list to a numpy array
    Features_array = np.array(Features)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(Features_array, Labels, train_size=0.8, test_size=0.2, random_state=42)

    print("Features shape:", Features_array.shape)
    print("Labels shape:", Labels.shape)

    # Create and train a Random Forest Classifier on the training data
    rf_clf = RandomForestClassifier(n_estimators=500, max_depth=50, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Make predictions on the test data
    y_pred = rf_clf.predict(X_test)

    # Evaluate the model's performance numerically and saves this
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    print("Classification Accuracy:", accuracy)
    print("Classification Report:\n", report)

    # Check which classes are being predicted
    predicted_classes = np.unique(y_pred)
    true_classes = np.unique(y_test)
    print("Classes predicted by the classifier:", predicted_classes)
    print("True classes in the test set:", true_classes)

    # Save the classification accuracy and report to a separate file
    with open("RT_classification_report.txt", "w") as f:
        f.write(f"Classification Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=[1, 2, 3])
    print("Confusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3], yticklabels=[1, 2, 3])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix for Number of Traps Prediction with Random Tree')
    plt.savefig("RT_Confusion_Matrix.png")
    plt.show()

    # Calculate feature importances
    importances = rf_clf.feature_importances_
    print("Feature importances:", importances)

    # Perform cross-validation to assess the model's stability and performance
    scores = cross_val_score(rf_clf, Features_array, Labels, cv=5)
    print("Cross-validation scores:", scores)
    print("Average cross-validation score:", np.mean(scores))

    # AUC-ROC curve plotting

    # Predict probabilities for ROC AUC
    y_prob = rf_clf.predict_proba(X_test)

    # Binarize the labels for ROC AUC calculation
    y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i + 1, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig("RT_AUC_ROC_Curves.png", dpi=600)
    plt.show()

# STEP 3: TRAIN A RANDOM FOREST REGRESSOR TO QUANTIFY ENERGY
# Prepare data for regressor

    Energy_Features = []
    Energy_Labels = []

    for i in range(NumSamples):
        NumTraps = Res[i][0]
        N_traps = Res[i][1]
        E_traps = Res[i][2]
        T = np.interp(np.linspace(0, len(Res[i][3])-1, num=100), np.arange(len(Res[i][3])), Res[i][3])

    # Use predicted number of traps to filter data
    for trap_index in range(NumTraps):
        energy_features = np.concatenate((T, np.array(E_traps[trap_index])))
        Energy_Features.append(energy_features)
        Energy_Labels.append(E_traps[trap_index])

    Energy_Features = np.array(Energy_Features)
    Energy_Labels = np.array(Energy_Labels)

    X_train_energy, X_test_energy, y_train_energy, y_test_energy = train_test_split(Energy_Features, Energy_Labels, train_size=0.8, test_size=0.2, random_state=42)
    print("Energy Features shape:", Energy_Features.shape)
    print("Energy Labels shape:", Energy_Labels.shape)
    # Train Random Forest Regressor
    rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_regressor.fit(X_train_energy, y_train_energy)

    # Make predictions and evaluate
    y_pred_energy = rf_regressor.predict(X_test_energy)
    print("Regression Mean Squared Error:", mean_squared_error(y_test_energy, y_pred_energy))  # Prepare data for regressor 