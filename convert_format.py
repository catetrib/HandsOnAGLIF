import numpy as np
from os import listdir
from os.path import isfile, join
import json

onlyfiles = ['dati_exp_neuron191113002_S75.txt']

for file_name in onlyfiles:
    if (file_name[-3:] == 'txt'):
        print(file_name)
        file1 = open(file_name, 'r')
        Lines = file1.readlines()
        EL = np.float64(Lines[0])
        vrm = np.float64(Lines[1])
        vth = np.float64(Lines[2])
        Istm = np.int32(Lines[3].split(','))
        spk_time_orig = []
        for i in range(len(Istm)):
            try:
                spk_time_orig.append(np.float64(Lines[4 + i].split(',')))
            except:
                spk_time_orig.append([])



        student_details = {
            "input_start_time": 88,
            "stimulus_duration": 400,
            "EL": EL,
            "V_reset": vrm,
            "V_threshold": vth,
            "spikes_times":{}
        }
        for i in range(len(Istm)):
            student_details["spikes_times"].update({str(Istm[i]): tuple(spk_time_orig[len(Istm) - i-1])})

# Convert and write JSON object to file
        with open(file_name[:-3]+'json', "w") as outfile:
            json.dump(student_details, outfile)
