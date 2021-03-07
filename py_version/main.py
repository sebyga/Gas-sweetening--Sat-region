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

#### Isotherm Data Importing  ####
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

dH_CO2_tmp = np.array(
    [df_NAME["CH4_Heat"].to_numpy(),
    df_NAME["H2S_Heat"].to_numpy(),
    df_NAME["CO2_Heat"].to_numpy()])
dH_CO2 = np.transpose(dH_CO2_tmp)

dH_H2S_tmp = np.array(
    [df_NAME["CH4_Heat"].to_numpy(),
    df_NAME["H2S_Heat"].to_numpy(),
    df_NAME["CO2_Heat"].to_numpy()])
dH_H2S = np.transpose(dH_H2S_tmp)

os.chdir(folder_base)

#### TEST "Rec" function ####
x_guess = [0.8,0.1,0.1]
y_feed_in = [0.8,0.05,0.15]
T_feed_in = 317 ## 
P_h = 10
P_l = 0.2
T_tri = [298,]*3
rec_res_tmp = ipsa.Rec(x_guess,P_h,P_l,
              bins_CO2[1], dH_CO2[1,:], T_tri,
              y_feed_in, T_feed_in)
            
print(dH_CO2[2,:])

print(' "Rec" funciton output is:')
print(rec_res_tmp)

#### Ideal PSA with various sorbents ####
### CO2 Case
P_h_range = np.arange(2,18.1,0.25)
x_guess = [0.8,0.1,0.1]
y_feed_in = [0.6,0.15,0.25]# (mol/mol) feed composition
T_feed_in = 343         # (K) temperature or 298K
P_l = 1                     # (bar) vacuum pressure
T_tri = [298.15,]*3
rec_result = []
sf_result = []
sf_arg_result = []

Comp_names = [['CH4'],
['H2S'],
['CO2']]

for binn,dH,nam in zip(bins_CO2,dH_CO2,Names_CO2):
    rec_list_tmp = []
    sf_list_tmp = []
    sf_arg_list_tmp = []
    for P in P_h_range:
        rec_tmp,sf_tmp = ipsa.Rec(x_guess,P,P_l,binn,dH, T_tri,y_feed_in, T_feed_in)
        rec_list_tmp.append(rec_tmp)
        sf_list_tmp.append(np.min(sf_tmp))
        sf_arg_list_tmp.append(np.argmin(sf_tmp))
    rec_list = np.array(rec_list_tmp)
    rec_result.append(rec_list)
    sf_arg_list = np.array(sf_arg_list_tmp)
    sf_arg_result.append(sf_arg_list)
    print(nam,': Rec. = {0:.2f} %'.format(rec_list[-1]),
    'Leading Heavy = ', Comp_names[sf_arg_list[-1]])

    sf_list = np.array(sf_list_tmp)
    sf_result.append(sf_list)

### Sort CO2 Case Result!
rec_last = []
for rec_list in rec_result:
    rec_last.append(rec_list[-1])
rec_last = np.array(rec_last)
ind_des = np.argsort(rec_last, )
print(ind_des)

# Arrange with the index
rec_result_sort = np.array(rec_result)[ind_des][::-1]
sf_result_sort = np.array(sf_result)[ind_des][::-1]
Names_CO2_sort= Names_CO2[ind_des][::-1]
sf_arg_result_sort = np.array(sf_arg_result)[ind_des][::-1]
