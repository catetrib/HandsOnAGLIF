# -*- coding: utf-8 -*-
"""
Runs AGLIF model for the included neuron


INPUT
User must specify:
1) simulation duration (in ms))
2) neuron name and parameters (from traces and from optimization procedure)
3) block line parameters
4) constant currents to simulate
5) wether to plot spike times


OUTPUT
the following files are generated for each Istim in given currents array:

neuronName\\neuronName+'_t_spk_'+str(Istim)+'pA.txt' : contains a list of spike times
neuronName\\neuronName+'_voltage_'+str(Istim)+'.txt' : contains voltage at each timestep

NB: creaates a folder named neuronName
"""



import AGLIF_040
from AGLIF_040 import AGLIFconstCurr
import numpy as np
import os

import matplotlib.pyplot as plt

'''
-------------------------------------------------------------------
user input 
-------------------------------------------------------------------
'''

''' 
1) -> insert simulation duration (ms)
'''
sim_lenght = 400


'''
2) -> insert neuron parameters
'''
neuronName="neuron18n21006b"
   
# neuron parameters from dati_exp.txt
EL=-63.1#EL,
vres=-24.8#vres
vtm=-13.3#vtm
    

# neuron parameters from neuronName_info.txt
Cm=50117.32962644806
Ith=66.43531847484938
tao=53628.21890375571
sc=5907.443721885336
alpha=[0.008463897813324019, 0.016927795626648038, 0.025391693439972055, 0.033855591253296076, 0.04231948906662009, 0.05078338687994411, 0.05924728469326813, 0.06771118250659215, 0.07617508031991617, 0.08463897813324019, 0.0931028759465642, 0.10156677375988822, 0.11003067157321224, 0.11849456938653626, 0.1269584671998603, 0.1354223650131843, 0.14388626282650832, 0.15235016063983234]
bet=0.0265370220225202
delta1=0.00998116642171794
Idep_ini=7.48053808129948
Idep_ini_vr=1409.3837980336698
psi=0.9559884248444085
#time scale=535.2721777787058  # not used in AGLIF script
A=364153798.5494981
B=-1.3799340747546924e-07
C=-364152518.92218256
alpha=8.67766736758823e-06


'''
3) insert block line parameters
    
    BLOCK LINE example: 
        
    I_monod_sup = 675.0
    I_monod_inf = 325.0
    t_val_max   = 0.19*I - 131.75
    t_val_min   = 96.0 - 0.09*I
    
    => blockLineParams = [I_monod_sup , coeffSup , constSup , I_monod_inf , coeffInf , constInf]
    => blockLineParams = [675.0 , 0.19 , -131.75 , 325.0 , -0.09 , 96.0]
    
    
'''
blockLineParams = [np.inf , 1.0 , np.inf , -np.inf , 1.0 , np.inf]


'''
4)-> define constant currents to analyze
'''
currents=[50,100,150,200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950]#[200,400,600,800,1000]


'''
5) -> to generate voltage output plots: wantToPlot=True
'''
wantToPlot=True


'''
-------------------------------------------------------------------
# END user input 
-------------------------------------------------------------------
'''


# spike times and simulated voltage are saved in a folder that is named as the neuron
os.mkdir(neuronName)
outputFilePath = neuronName 

# equilibrium parameters
v_min = -90
minCurr = -185
# Neuron
zeta = 3.5e-3
eta = 2.5e-3
rho = 1e-3
csi = 3.5e-3

equilibriumParameters = [v_min,minCurr,zeta,eta,rho,csi]


# creation of neuronParameters array 
neuronParameters = [
    EL,#EL,
    vres,#vres
    vtm,#vtm
    Cm,#Cm
    Ith,#ith
    tao,#tao_m
    sc,#sc = 
    bet,
    delta1,
    Idep_ini,#cost_idep_ini (Idep_start)
    Idep_ini_vr,#Idep_ini_vr (Idep0)
    psi,#psi1
    A,#a
    B,#b
    C,#c
    alpha,#alp
    0,#istim_min_spikinig_exp
    1000,#istim_max_spikinig_exp
    sim_lenght,#sim_lenght
    blockLineParams
    ]

'''
---------------------------------------------------------------------------------------------------------------------------
run AGLIF model at specified currents
---------------------------------------------------------------------
'''

for Istim in currents:
    
    # builds current array
    change_cur=0.1
    campionamento=20
    d_dt=0.005*campionamento
    corr_list=np.ones(int(sim_lenght/d_dt))*0 
    corr_list[int(change_cur/d_dt):int(sim_lenght/d_dt)+1] = np.ones(len(corr_list[int(change_cur/d_dt):int(sim_lenght/d_dt)+1]))*Istim
        
    # output filenames
    tSpikeOutputFileName = outputFilePath+'\\'+neuronName+'_t_spk_'+str(Istim)+'pA.txt'
    voltageOutputFileName = outputFilePath+'\\'+neuronName+'_voltage_'+str(Istim)+'.txt'
            
    AGLIFconstCurr(neuronParameters,equilibriumParameters,corr_list,tSpikeOutputFileName,voltageOutputFileName)
    
        
if wantToPlot:        
    plt.figure(figsize=(8,8))
    for i in range(len(currents)):
        #print(currents[i])
        spk_time =np.loadtxt(outputFilePath+'\\'+neuronName+'_t_spk_'+str(currents[i])+'pA.txt')
        lo=plt.scatter(spk_time, currents[i] * np.ones(len(spk_time)), marker='|',
                    label='exp',color='b',alpha=0.5);
        plt.xlim(0, sim_lenght)
        
        plt.title(neuronName)
        plt.ylabel('Current (pA)')
        plt.xlabel('time (ms)')
    plt.savefig(neuronName+'_spike_times.png')