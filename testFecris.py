import Fecris as venus
import numpy as np
import KalmanFilterStdEst as kfilt
import matplotlib.pyplot as plt

'''Test function for the Fake ecris.
Generates random point and evaluate the response of the simulated ion source of Fecris.py'''

# parameters of the Kalman filter
cMeas = 1e3 # filter parameter the increase to filter more decrerasse to track faster
slopeMax  = .00001 # slope used to determine if the system is settled

Disp = True # Turns on display at every step

# lists for data storage and display
storeMeas = []
slope = []
FilteredI = []

Settings = np.random.rand(30, 2) # Settings where the Fecris will ber evaluated
myEcris = venus.FECRIS2D(Settings[0]) # creating the ion source object with the initial state
Y = [] # list for output (beam current)
S = [] # list for stability
tmp = []
# getting the fisrt statistics the old way
for k in range(100):
    tmp.append(myEcris.read())
Y.append(np.mean(tmp))
S.append(np.std(tmp))
# going through the evaluations
for k in range(len(Settings)-1):
    unsettled = True
    KF = kfilt.KFobject(myEcris.read(), cMeas) # reinitialize Kalman filter for every new settings
    inftyLoop = 0
    while(unsettled): # going through the transition
        inftyLoop += 1
        measure = myEcris.Transition(Settings[k+1])# getting a transition measurement from the Fecris
        deltaT = np.random.randn() * .3 + .1 # forcing irregular sampling
        KF.EstimateState(measure, deltaT) # Filtering the new measure
        if inftyLoop > 20: # after 20 call we start to evaluate the slope to see if the system has settled
            if np.prod(np.abs(slope[len(slope)-11::]) < slopeMax): # if the last 10 values of the slope meet the settling criterion
                unsettled = False
                Y.append(KF.X[0])# grabing the filtered output value
                S.append(KF.Sig[0]/KF.X[0]) # calculating the stability value
                myEcris.SetState(Settings[k+1]) # !!!! Important !!!! before moving to a new transition we force Fecris to new state
                print('settled at t=', inftyLoop) # how many call were needed for the system to settle

        if inftyLoop > 5000: # if the system never settles
            print('timeout')
            # grabing values but will pollute data set
            Y.append(KF.X[0])
            S.append(KF.Sig[0]/KF.X[0])
            unsettled = False
            myEcris.SetState(Settings[k + 1])
        # storing values for display
        storeMeas.append(measure)
        FilteredI.append(KF.X[0])
        slope.append(KF.X[1])
    if Disp:
        fig = plt.figure()
        ax = fig.add_subplot(122, projection='3d')
        p = ax.scatter(Settings[0 :k + 1, 0], Settings[0 :k + 1, 1], Y[0 :k + 1], c=S[0 :k + 1], cmap='viridis')
        inds = np.where(np.asarray(S[0 :k + 1]) > .05)[0]
        if len(inds) :
            ax.scatter(Settings[inds, 0], Settings[inds, 1], np.asarray(Y)[inds], marker='x', color='red')
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_zlabel(r'$\mathcal{I}$')
        ax.legend(['Evaluted', 'Unstable'])
        fig.colorbar(p, orientation="horizontal")
        ax1 = fig.add_subplot(221)
        ax1.plot(storeMeas)
        ax1.plot(FilteredI)
        ax1.legend(['Output', 'Filtered'])
        ax2 = fig.add_subplot(223)
        ax2.plot(slope)
        ax2.plot(np.ones(len(slope)) * slopeMax, 'r--')
        ax2.plot(-np.ones(len(slope)) * slopeMax, 'r--')
        ax2.legend(['slope', 'limit'])
        plt.show()






