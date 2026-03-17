import numpy as np
import matplotlib.pyplot as plt
from sincfold.utils import dot2matrix

sample_name = "prueba5"
mat_1 = np.loadtxt("middle_matrix/compressed_" + sample_name + ".csv", delimiter=",")
mat_2 = np.loadtxt("middle_matrix/expanded_" + sample_name + ".csv", delimiter=",")
mat_3 = np.loadtxt("middle_matrix/prefinal_" + sample_name + ".csv", delimiter=",")
mat_4 = np.loadtxt("middle_matrix/final_" + sample_name + ".csv", delimiter=",")

dot = ""
with open("sample/test_mock_k3.fasta", "r") as file:
    while line:=file.readline():
        line = line.strip().split(' ')
        if len(line) == 1:
            continue
        if line[1] == sample_name:
            file.readline()
            dot = file.readline().strip()
            break

ideal = dot2matrix(dot).to('cpu').numpy()

plt.imsave("imgs/compressed_"+sample_name+".png", mat_1, cmap='viridis', origin='upper')
plt.imsave("imgs/expanded_"+sample_name+".png", mat_2, cmap='viridis', origin='upper')
plt.imsave("imgs/prefinal_"+sample_name+".png", mat_3, cmap='viridis', origin='upper')
plt.imsave("imgs/final_"+sample_name+".png", mat_4, cmap='viridis', origin='upper')
plt.imsave("imgs/real_"+sample_name+".png", ideal, cmap='viridis', origin='upper')

# read the csv file from output_trial_3/train_log.csv as dataframe and plot the  
