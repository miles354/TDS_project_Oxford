import TDS_Sim
import random
import matplotlib.pyplot as plt
import joblib

# Settings to use when this is the main file
if __name__ == "__main__":
    random.seed(17)
    NumSamples = 1 #how many samples to generate
    n_cpu_cores = 5 #how many cpu cores to run simulations on

def takeSecond(elem):
    return elem[1]

# Generates a single TDS curve based on (mostly) random binding energies and trapping site concentration
def GenerateDataPoint(i):
    PlotWhileSolving = False #whether to generate plots when simulating (much slower)
    model = "Rates"     #currently using Mcnabb-foster
    NumTraps = random.randint(1,3)      #number of trapping sites

    N_Elements = 10     #for obtaining the solution, number of linear finite elements used
    dT_Step = 120.0     # time increment used

    Thickness = 10.0e-3 #sample thickness

    Diff_RT = 1.0e-9    #diffusion rate at room temperature
    E_diff = 2*4.0e3    #diffusion energy (used for temperature scaling)

    k_T  = [1.0e4, 1.0e0]   #boundary flux absorbtion-desorbtion rates)

    time_Charge = 12.0*3600.0   #time left charging
    Temp_Charge = 273.15+20.0   #temperature at which samples are charged
    pCharge = 1.0e6             #hydrogen fugacity during charging

    time_rest = 600.0           #time sample is being transfered from charging to TDS

    TDS_HeatingRate = 1.0/60.0  #heating rate of TDS
    TDS_Time = 3600.0*24        #total time over which TDS is performed

    #generate trapping energies
    traps = []
    for t in range(0,NumTraps):
        validPoint = False
        while validPoint == False:
            E_abs = 20.0e3
            E_des = random.uniform(40.0e3, 150.0e3)
            N = random.uniform(30.0, 100.0)

            #make sure energies are slightly distinct
            goodDist = True
            for E in traps:
                if abs(E[1]-E_des)<10e3: 
                    goodDist = False
            validPoint = goodDist

        traps.append([E_abs, E_des, N])
        traps.sort(key=takeSecond)

    #save trapping sites and energies as vectors
    N_traps = []
    E_traps = []
    for t in traps:
        N_traps.append(t[2])
        E_traps.append([t[0], t[1]])

    print(str(i)+":")
    print("\t N="+str(N_traps))
    print("\t E="+str(E_traps))

    #perform TDS experiment within simulation
    Sample = TDS_Sim.TDS_Sample(Thickness, N_Elements, dT_Step, PlotWhileSolving) #initializes material
    Sample.Set_Params(model, Diff_RT, E_diff, N_traps, E_traps, k_T)    #Sets material parameters
    Sample.Charge(time_Charge, Temp_Charge, pCharge)    #performs charging
    Sample.Rest(time_rest)                              #leave at atmospheric pressure
    [T,J] = Sample.TDS(TDS_Time, TDS_HeatingRate)       #perform TDS

    return NumTraps, N_traps, E_traps, T, J  

# Test function to check if all works
if __name__ == "__main__":
    # Run simulations in parrallel
    Sample_NTraps = []
    SampleResults = []
    Sample_N_Traps = []
    Sample_E_Traps = []

    Res = joblib.Parallel(n_jobs=n_cpu_cores)(joblib.delayed(GenerateDataPoint)(i) for i in range(0,NumSamples))

    #save to data structure
    for i in range(0,NumSamples):
        Sample_NTraps.append(Res[i][0])
        Sample_N_Traps.append(Res[i][1])
        Sample_E_Traps.append(Res[i][2])
        SampleResults.append([Res[i][3],Res[i][4]])

    #plot results
    fig = plt.figure()
    for i in range(0,NumSamples,1):

        #plot resulting TDS curve
        p1 = plt.plot(Res[i][3],Res[i][4]*1000)
        plt.draw()
        plt.xlabel("Temperature")
        plt.ylabel("Hydrogen Flux")

        plt.pause(1.0e-10)
        
    plt.savefig("TDS_Samples.png", dpi=600)
    plt.show()
    input("Press Enter to continue (closes all figures)...")  