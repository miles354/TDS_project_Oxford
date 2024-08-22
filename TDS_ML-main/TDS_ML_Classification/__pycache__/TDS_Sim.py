import numpy as np
import scipy as sp
import math as math
import matplotlib.pyplot as plt

class TDS_Sample:
    LumpedCap = True

    #physical constants and references
    R = 8.3144          # universal gas constant
    RoomTemp = 293.15   # Room temperature [K]
    f_H2_room = 0.0     # Hydrogen partial pressure in room [Pa]
    NL = 1.0e6          # Concentration of lattice sites [mol/m^3]
    f0 = 1.0e6          # Reference pressure [Pa]

    conv_crit = 1.0e-9  # concergence criteria, below which the simulations consider themselves converged

    # Initializes the data structures needed for simulations
    def __init__(self, 
                 Thickness: float,  #sample thickness [m]
                 N_Elements: int,   #number of finite elements per half thickness
                 TStep: float,      #size of time increments [s]
                 viz: bool):        #plotting while simulating (very slow)
        
        self.Visualize = viz    #plotting while simulating (very slow)
        self.Thickness = Thickness/2    #Thickness of the metal sample [m], using symmetry, only simulating half the sample

        self.N_Elems = N_Elements       #number of finite elements used
        self.TStep = TStep              #time step size used [s]

        # State vector, saving lattice concentrations
        self.NDofs = self.N_Elems+1
        self.StateVec = np.zeros([self.NDofs,1])
        self.OldStateVec = np.zeros([self.NDofs,1])

        # pre-calculate shape functions
        self.N_ip = 2   #Number of integration points per element
        self.N = np.zeros((self.N_Elems, self.N_ip, 2)) # Element shape functions
        self.G = np.zeros((self.N_Elems, self.N_ip, 2)) # Element shape function gradients
        self.w = np.zeros((self.N_Elems, self.N_ip))    # Integration point weights

        for el in range(0, self.N_Elems):
            dx = self.Thickness/self.N_Elems    #element size
            x_elem = (np.array([0.0, 1.0])+el)*dx
            
            # integration locations and weights for an element of length [0,1]
            x_ip, w_ip = sp.special.roots_legendre(self.N_ip)
            x_ip = (x_ip+1)/2
            w_ip = w_ip/2

            for ip in range(0,self.N_ip):
                self.N[el][ip][0] = x_ip[ip]
                self.N[el][ip][1] = 1-x_ip[ip]
                self.G[el][ip][0] = -1/dx
                self.G[el][ip][1] = 1/dx
                self.w[el][ip] = w_ip[ip]*dx
        
        # required for plotting while solving
        self.XPlot = np.arange(0, self.Thickness+1.0e-10, dx)
        if (self.Visualize):
            self.PlotState(np.zeros(1), np.zeros(1))

    # plots the current hydrogen concentration and TDS curve
    def PlotState(self,
                  tvec: np.ndarray,     #time vector
                  jvec: np.ndarray):    #TDS flux vector
        
        fig, self.ax = plt.subplots(2)
        self.p1, = self.ax[0].plot(self.XPlot, self.StateVec)
        self.p2, = self.ax[1].plot(tvec/3600, jvec)
        plt.draw()
        plt.pause(1.0e-10)

    # updates any plots
    def UpdatePlot(self, 
                   tvec: np.ndarray,     #time vector
                   jvec: np.ndarray):    #TDS flux vector
        
        self.p1.set_ydata(self.StateVec)
        self.ax[0].set_ylim(min(0.0,np.min(self.StateVec)), np.max(self.StateVec)*1.1)
        self.p2.set_data(tvec/3600, jvec)
        self.ax[1].set_xlim(0, np.max(tvec)/3600)
        self.ax[1].set_ylim(np.min(jvec), np.max(jvec))
        plt.draw()
        plt.pause(1.0e-10)
    
    # sets maerial parameters
    def Set_Params(self, 
                   model: str, # Mode to use for trapping sites, either 'Rates' or 'Oriani'
                   Diff_RT: float, #Diffusion rate at room temperature [m/s]
                   E_diff: float, #Energy level of lattice sites (used for diffusion time dependence)
                   N_traps: list[float], #Concentration of trapping sites [mol/m^3]
                   E_traps: list[float], #Energy levels of trapping sites
                   k_T: list[float]): #Tafel reaction absorption/desorption rate constants
        
        self.model = model
        self.N_traps = N_traps
        self.E_traps = E_traps
        self.E_diff = E_diff
        self.k_T = k_T

        #calculate temperature dependent rate parameters based on room temp inputs
        self.E_T = -self.R*self.RoomTemp*math.log(k_T[0])
        self.E_Tr = -self.R*self.RoomTemp*math.log(k_T[1])

        self.D0 = Diff_RT*math.exp(self.E_diff/self.R/self.RoomTemp)

        #initialize data structure for storing trapping site occupancy
        if (self.model=="Rates"):
            self.HistNew = np.zeros([self.NDofs, len(self.N_traps)])
            self.HistOld = np.zeros([self.NDofs, len(self.N_traps)])

    #Copies New state to Old state, to proceed to next time increment
    def Commit(self):
        if (self.model=="Rates"):
            self.HistOld = self.HistNew.copy()
        
    #Simulation performing charging cycle
    def Charge(self, 
               t_charge: float,     #time for which sample is charged [s]
               Temp_Charge: float,  #Temperature at which sample is charged [K]
               pCharge: float):     #Pressure at which sample is charged [Pa]
        
        t = 0.0 #current time

        self.T = Temp_Charge
        self.p = pCharge

        # Resize vector to store time data
        jVec = np.zeros(math.ceil(t_charge/self.TStep))
        tVec = np.zeros(math.ceil(t_charge/self.TStep))

        #time stepping loop
        step = 0
        while t<t_charge:
            tVec[step] = t     
            jVec[step] = self.DoStep()

            t += self.TStep
            self.Commit()
            step += 1

            if ((self.Visualize) & (step%100==0)):
                self.UpdatePlot(tVec[0:step], jVec[0:step])

        if (self.Visualize):
            self.UpdatePlot(tVec, jVec)

    #leave the sample at room temperature/pressure (e.g. during transferring from charging to TDS)
    def Rest(self, 
             t_rest: float): #time sample is left [s]
        
        t = 0.0

        self.T = self.RoomTemp
        self.p = self.f_H2_room

        jVec = np.zeros(math.ceil(t_rest/self.TStep))
        tVec = np.zeros(math.ceil(t_rest/self.TStep))

        step = 0
        while t<t_rest:
            tVec[step] = t     
            jVec[step] = self.DoStep()

            t += self.TStep
            self.Commit()
            step += 1

            if (((self.Visualize) & (step%100==0))):
                self.UpdatePlot(tVec[0:step], jVec[0:step])

        if (self.Visualize):
            self.UpdatePlot(tVec[0:step], jVec[0:step])
        
    #simulate TDS discharging
    def TDS(self, 
            TDS_Time: float, #time for which TDS is performed [s]
            TDS_HeatingRate: float): #rate at which temperature is increased [K/s]
        
        t = 0.0

        self.T = self.RoomTemp
        self.p = self.f_H2_room

        # Initialize ouytput vectors
        jVec = np.zeros(math.ceil(TDS_Time/self.TStep)) #hydrogen flux [mol/m^2/s]
        tVec = np.zeros(math.ceil(TDS_Time/self.TStep)) #time vector [s]
        TVec = np.zeros(math.ceil(TDS_Time/self.TStep)) #Temperature Vector [K]

        #perform actual simulation
        step = 0
        while t<TDS_Time:
            self.T = self.RoomTemp + t*TDS_HeatingRate
            tVec[step] = t     
            jVec[step] = self.DoStep()
            TVec[step] = self.T

            t += self.TStep
            self.Commit()
            step += 1

            if (((self.Visualize) & (step%100==0))):
                self.UpdatePlot(tVec[0:step], jVec[0:step])

        if (self.Visualize):
            self.UpdatePlot(tVec[0:step], jVec[0:step])   
            
        return TVec, jVec   

    # Calculates a single time increment, return exit flux for this increment
    def DoStep(self):
        conv = False
        K, f, j = self.GetKf()

        it = 0
        while (conv==False & it<20):
            dState = -np.linalg.solve(K, f)
            self.StateVec += dState

            K, f, j = self.GetKf()
            err = np.tensordot(f, dState)

            it += 1
            if (err<self.conv_crit):
                conv = True
        

        self.OldStateVec = self.StateVec

        return j

    #Gets the tangent matrix, required within DoStep to perform non-linear NR iterations
    def GetKf(self):

        # Initialize tangent matrix and force vector
        K = np.zeros([self.NDofs, self.NDofs])
        f = np.zeros([self.NDofs, 1])

        #integrate all elements
        for el in range(0,self.N_Elems):

            C = self.StateVec[el:el+2,0]    #lattice concentrations relevant to current element
            COld = self.OldStateVec[el:el+2,0]  # lattice concentrations at previous time increment

            WLumped = self.N[el][0]*0.0     #Lumped integration weights

            #Sum over integration points
            for ip in range(0,self.N_ip):

                #Calculate local concentrations, and time derivative
                cloc = np.matmul(self.N[el][ip], C)
                dcloc= np.matmul(self.N[el][ip], C-COld)/self.TStep

                #lumped integration weights
                WLumped += self.w[el][ip]*self.N[el][ip]

                if (self.LumpedCap==False):
                    #capacity
                    NtN = np.outer(self.N[el][ip].T,self.N[el][ip])
                    f[el:el+2,0]        += self.w[el][ip]*self.N[el][ip].T*dcloc
                    K[el:el+2,el:el+2]  += self.w[el][ip]*NtN/self.TStep

                    #traps (only working for Oriani as non-lumped)
                    if (self.model == "Oriani"):
                        for E,Nt in zip(self.E_traps, self.N_traps):
                            eTerm = math.exp(E/self.R/self.T)
                            cap = Nt/self.NL * eTerm/((1.0+max(0.0,cloc)/self.NL*eTerm)**2)
                            dcap = -2.0*Nt/self.NL*eTerm*eTerm/self.NL/(1.0+max(0.0,cloc)/self.NL*eTerm)**3

                            f[el:el+2,0] += self.w[el][ip] * self.N[el][ip].T * cap * dcloc
                            K[el:el+2,el:el+2]  += self.w[el][ip] * (cap/self.TStep *  + dcap*dcloc) * NtN

                #diffusion
                GtG = np.outer(self.G[el][ip].T,self.G[el][ip])
                D_eff = self.D0*math.exp(self.E_diff/self.R/self.T)
                f[el:el+2,0]        += self.w[el][ip]*D_eff*np.matmul(GtG,C)
                K[el:el+2,el:el+2]  += self.w[el][ip]*D_eff*GtG    

                if (self.LumpedCap==True):
                    for n in range(0,2):
                        #capacity
                        f[el+n,0]     += WLumped[n]*(C[n]-COld[n])/self.TStep
                        K[el+n,el+n]  += WLumped[n]/self.TStep

                        #traps
                        if (self.model == "Oriani"):
                            for E,Nt in zip(self.E_traps, self.N_traps):
                                eTerm = math.exp(E/self.R/self.T)
                                cap = Nt/self.NL * eTerm/((1.0+max(0.0,C[n])/self.NL*eTerm)**2)
                                dcap = -2.0*Nt/self.NL*eTerm*eTerm/self.NL/(1.0+max(0.0,C[n])/self.NL*eTerm)**3

                                f[el+n,0] += WLumped[n] * cap * (C[n]-COld[n])/self.TStep
                                K[el+n,el+n]  += WLumped[n] * (cap + dcap*(C[n]-COld[n]))/self.TStep 

                        if (self.model == "Rates"):
                            for i,E,Nt in zip(range(0,len(self.E_traps)), self.E_traps, self.N_traps):
                                CT_Old = self.HistOld[el+n,i] 
                                CT_New = self.HistNew[el+n,i]
                                CL_New = C[n]

                                CT_New, v_trap, dv_dCL = self.TrappingDynamic(CT_Old, CT_New, CL_New, E, Nt)

                                f[el+n,0] += WLumped[n] * v_trap  
                                K[el+n,el+n] += WLumped[n] * dv_dCL

                                self.HistNew[el+n,i] = CT_New

                

        if (np.isnan(K).any() or np.isnan(f).any()):
            print("D_eff=")
            print(D_eff)
            print("C=")
            print(self.StateVec)
            input("NanHewr Enter to continue...")

        #boundary
        Cb = self.StateVec[0,0]
        k_in = math.exp(-self.E_T/self.R/self.T)
        k_out =  math.exp(-self.E_Tr/self.R/self.T)
        j = - k_in * math.sqrt(self.p/self.f0) + k_out * max(0.0,Cb)
        f[0]   += j
        if (Cb>=0):
            K[0,0] += k_out

        return K, f, j

    def TrappingDynamic(self, CT_Old, CT_New, CL_New, E, Nt):
        #returns the hydrogen concentration at trapping sites, and the rate of hydrogen absorption into trapping sites and its derivative w.r.t. C_L 

        Debye = 1e13 #units: collisions/atom/s
        c_fac = 1e2*Nt  #This should be Debye*Nt, but results in unstable simulations due to the fast rate

        ## return mapping scheme, defining local state vector as [CT_New, v_trapping]
        conv = False
        it = 0

        Loc_State = np.zeros([2,1])
        Loc_State[0,0] = CT_New
        while (conv == False):
            # apply sensible limits to CL, CT to retain stability
            CL = max(0.0,CL_New)
            CT = max(0.0,Loc_State[0,0])

            #calculate absorption and desorption rates as:
            #  v_abs = k_abs \theta_L * (1-\theta_T) 
            v_abs = CL/self.NL*(1-CT/Nt)*math.exp(-E[0]/self.R/self.T)
            dv_abs_dL = (1-CT/Nt)/self.NL*math.exp(-E[0]/self.R/self.T)
            dv_abs_dT = -CL/self.NL/Nt*math.exp(-E[0]/self.R/self.T)

            #  v_des = k_des \theta_T * (1-\theta_L) 
            v_des = CT/Nt*(1-CL/self.NL)*math.exp(-E[1]/self.R/self.T)
            dv_des_dL = -1.0/self.NL*CT/Nt*math.exp(-E[1]/self.R/self.T)
            dv_des_dT = (1-CL/self.NL)/Nt*math.exp(-E[1]/self.R/self.T)

            #net trapping rate
            v = c_fac*(v_abs-v_des)
            dv_dT = c_fac*(dv_abs_dT-dv_des_dT)
            dv_dL = c_fac*(dv_abs_dL-dv_des_dL)

            # Force vector and tangent vector, solving:
            # f_1 = \dot{C_T} - v = 0   (change in trapping concentration resulting from absorption)
            # f_2 = v - v = 0           (setting the state variable equal to the trapping rate)
            fvec = np.zeros([2,1])
            fvec[0,0] = (Loc_State[0,0]-CT_Old)/self.TStep - Loc_State[1,0]
            fvec[1,0] = v - Loc_State[1,0]

            KMat = np.zeros([2,2])
            KMat[0,0] = 1.0/self.TStep
            KMat[0,1] = -1.0
            KMat[1,0] = dv_dT 
            KMat[1,1] = -1.0

            # update increment
            dLocState = -np.linalg.solve(KMat,fvec)
            Loc_State +=  dLocState

            err = np.dot(dLocState.T, fvec)

            # check for convergence
            it += 1
            if (it>20 or err<1.0e-9):
                conv = True

        #extract a consistent tangent matrix, and provide outputs
        CMat = np.zeros([2,1])
        CMat[1,0] = dv_dL

        dMat = np.matmul(np.linalg.inv(KMat), CMat)

        dv_dCL = dMat[1,0]
        CT_New = Loc_State[0,0] 
        v_trap = Loc_State[1,0]

        return CT_New, v_trap, dv_dCL


# Test function to check if all works
if __name__ == "__main__":
    Diff_RT = 1.0e-9
    E_diff = 2*4.0e3

    #model = "Oriani"
    model = "Rates"

    N_traps = [10.0, 10.0]
    if (model=="Oriani"):
        E_traps = [40.0e3, 70.0e3]
    if (model=="Rates"):
        E_traps = [[20.0e3, 60.0e3], [20.0e3, 90.0e3]]

    k_T  = [1.0e4, 1.0e0]

    time_Charge = 12.0*3600.0
    Temp_Charge = 273.15+20.0
    pCharge = 1.0e6

    time_rest = 1800.0

    N_Elements = 10
    dT_Step = 6.0
    Thickness = 10.0e-3

    TDS_HeatingRate = 60.0/60.0
    TDS_Time = 2000/TDS_HeatingRate

    Sample = TDS_Sample(Thickness, N_Elements, dT_Step, False)
    Sample.Set_Params(model, Diff_RT, E_diff, N_traps, E_traps, k_T)
    Sample.Charge(time_Charge, Temp_Charge, pCharge)
    Sample.Rest(time_rest)
    [T,J] = Sample.TDS(TDS_Time, TDS_HeatingRate)

    if True:
        fig = plt.figure()
        p1 = plt.plot(T, J)
        plt.draw()
        plt.pause(1.0e-10)

        input("Press Enter to continue...")