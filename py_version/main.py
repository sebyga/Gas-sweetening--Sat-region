import pyiast
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import time
import os
import scipy.optimize as optim
import matplotlib.pyplot as plt

os.chdir('py_version')

import Ideal_PSA as ipsa
os.chdir('..')

folder_base = os.getcwd()
print('Currrent path is ... :')
print(folder_base)

df_NAME=pd.read_csv("HEAT_0215.csv")

bins_H2S = []
for nam in df_NAME["NAME"]:
    f_tmp = open("iso_"+nam+"_saved.bin",'rb')
    bin_tmp = pickle.load(f_tmp)
    bins_H2S.append(bin_tmp)
    f_tmp.close()
    
bins_CO2 = []
for nam in df_NAME["NAME"]:
    f_tmp = open("iso_"+nam+"_saved.bin",'rb')
    bin_tmp = pickle.load(f_tmp)
    bins_CO2.append(bin_tmp)
    f_tmp.close()

Names_CO2=df_NAME["NAME"].to_numpy()
Names_H2S=df_NAME["NAME"].to_numpy()
dH_CO2 = np.array([df_NAME["CH4_Heat"],df_NAME["H2S_Heat"],df_NAME["CO2_Heat"]]).T
dH_H2S = np.array([df_NAME["CH4_Heat"],df_NAME["H2S_Heat"],df_NAME["CO2_Heat"]]).T

os.chdir(folder_base)

#### TEST "Rec" function ####
x_guess = [0.8,0.1,0.1]
y_feed_in = [0.8,0.05,0.15]
T_feed_in = 317 ## 
P_h = 10
P_l = 0.2
T_tri = [298,]*3
rec_res_tmp = ipsa.Rec(x_guess,P_h,P_l,
              bins_CO2[1],dH_CO2[1], T_tri,
              y_feed_in, T_feed_in)
            

print(' "Rec" funciton output is:')
print(rec_res_tmp)        