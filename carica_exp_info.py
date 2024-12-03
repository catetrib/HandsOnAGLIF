import numpy as np
import json
def neuron_info_load(neuron_nb):
    ############### Caricamento delle info su dati relativi al neurone sperimentale da File

    spk_interval=[]
    spk_time=[]
    spk_time_orig = []
    file1 = open('dati_exp_' + neuron_nb + '.json', 'r')
    experimental_data=json.load(file1)
    EL = np.float16(experimental_data["EL"])
    vrm = np.float16(experimental_data["V_reset"]) / -EL
    vth = np.float16(experimental_data["V_threshold"]) / -EL
    Istm = np.fromiter(experimental_data["spikes_times"].keys(), dtype=np.int32)
    for i in experimental_data["spikes_times"].keys():
        spk_time_orig.append(np.array(experimental_data["spikes_times"][i]).astype(np.float64))
    dur_sign = experimental_data["stimulus_duration"]
    input_start_time = experimental_data["input_start_time"]

    for i in range(len(spk_time_orig)):
        spk_time.append(np.array(spk_time_orig[i]) - input_start_time)
    is_spk_curr=np.array(np.zeros(Istm.__len__()),dtype=bool)
    is_spk_2_curr = np.array(np.zeros(Istm.__len__()), dtype=bool)
    for i in range(len(spk_time)):
        if spk_time[i].size > 0:
            spk_interval.append([spk_time[i][0]])
            is_spk_curr[i] = True  # correnti con almeno uno spike
            if spk_time[i].size > 1:
                is_spk_2_curr[i]  = True # correnti con almeno 2 spikes
            for j in range(len(spk_time[i]) - 1):
                spk_interval[i].append(
                    spk_time[i][j + 1] - spk_time[i][j] - 2)  # creo un vettore di liste con gli ISI per ogni corrente
        else:
            spk_interval.append([])


    return EL,vrm,vth,Istm,spk_time_orig,dur_sign,input_start_time,spk_time,spk_interval,is_spk_curr,is_spk_2_curr
