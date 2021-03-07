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


##### Start: Isotherm Function  ####
Arrh = lambda T,dH ,T_ref: np.exp(-dH/8.3145*(1/T - 1/T_ref)) # Arrhenius equation (Clasius-Clapeyron Equation)

## Isothermal mixture isotherm
def iso_mix(P_par, T, iso_list, dH_list,Tref_list):
    P_norm = []
    for (p,dh,tref) in zip(P_par, dH_list,Tref_list):
        p_n = Arrh(T,dh,tref)*p 
        P_norm.append(p_n)
    P_norm_arr = np.array(P_norm)
    #print(P_norm_mat.T)
    if P_norm_arr.ndim > 1:
        for i in range(len(P_norm[0])):
            p_tmp = P_norm_arr[i,:]
            p_tmp[p_tmp<0.000001] = 0.000001
            q_IAST_tmp = pyiast.iast(p_tmp,
                                     iso_list,
                                     warningoff=True)
    else:
        try:
            p_tmp = P_norm_arr
            p_tmp[p_tmp<0.000001] = 0.000001
            #print(p_tmp)
            q_IAST_tmp = pyiast.iast(p_tmp,
                                    iso_list,
                                     warningoff=True)
        except:    
            try:
                #print('Initial guess error with P = ',P_par)
                x_IG = np.ones(len(p_tmp))/len(p_tmp)
                q_IAST_tmp = pyiast.iast(p_tmp,
                                        iso_list,adsorbed_mole_fraction_guess = x_IG,
                                        warningoff=True)
            except:
                try:
                    arg_min = np.argmin(p_tmp)
                    p_tmp[p_tmp<0.000001] = 0.000001
                    x_IG = 0.05*np.ones(len(p_tmp))
                    x_IG[arg_min] = 1 - 0.05*(len(p_tmp)-1)
                    #print(x_IG)
                    q_IAST_tmp = pyiast.iast(p_tmp,
                                            iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)

                except:
                    try:
                        arg_max = np.argmax(p_tmp)
                        p_tmp[p_tmp<0.000001] = 0.000001
                        x_IG = 0.05*np.ones(len(p_tmp))
                        x_IG[arg_max] = 1 - 0.05*(len(p_tmp)-1)
                        #print(x_IG)
                        q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                                warningoff=True)        
                    except:
                        try:
                            arg_max = np.argmax(p_tmp)
                            p_tmp[p_tmp<0.000001] = 0.000001
                            x_IG = 0.15*np.ones(len(p_tmp))
                            x_IG[arg_max] = 1 - 0.15*(len(p_tmp)-1)
                            #print(x_IG)
                            q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                                warningoff=True)
                        except:
                            try:
                                arg_min = np.argmin(p_tmp)
                                p_tmp[p_tmp<0.000001] = 0.000001
                                x_IG = 0.01*np.ones(len(p_tmp))
                                x_IG[arg_min] = 1 - 0.01*(len(p_tmp)-1)
                                #print(x_IG)
                                q_IAST_tmp = pyiast.iast(p_tmp,
                                            iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)

                            except:
                                arg_max = np.argmax(p_tmp)
                                p_tmp[p_tmp<0.000001] = 0.000001
                                x_IG = 0.01*np.ones(len(p_tmp))
                                x_IG[arg_max] = 1 - 0.01*(len(p_tmp)-1)
                                #print(x_IG)
                                q_IAST_tmp = pyiast.iast(p_tmp,
                                                iso_list,adsorbed_mole_fraction_guess = x_IG,
                                            warningoff=True)                                
           
    return q_IAST_tmp




#### End: Isotherm Function #### 

#### Start: Saturation region Function  ####
def sat_fraction(y_lead,q_lead_array, y_follow,q_follow_array):
    q_sat_lead_part, q_sat_lead_tot, q_des_lead = q_lead_array
    q_sat_fo, q_des_fo = q_follow_array
    numo = y_follow*(q_sat_lead_part - q_des_lead)+y_lead*q_des_fo
    denom = y_follow*(q_sat_lead_part - q_sat_lead_tot)+y_lead*q_sat_fo
    sat_frac = numo/denom
    return sat_frac

#### End: Saturation region Function ####

#### Start: x2x = Single cycle ####
def x2x(x_ini,P_high,P_low,iso_input, dH_input, Tref_input,yfeed,Tfeed):
    iso_1 = iso_input[2] # CH4
    iso_2 = iso_input[0] # H2S
    iso_3 = iso_input[1] # CO2
    iso  = [iso_1,iso_2,iso_3]
    dH_1, dH_2, dH_3 = dH_input[:3]         # (kJ/mol): Heat of adsorption
    dH = np.array([dH_1,dH_2,dH_3])*1000    # (J/mol): Heat of adsorption 
    P_low_part = np.array(x_ini)*P_low      # (bar): partial pressure
    P_high_part = np.array(yfeed)*P_high    # (bar): partial pressure
    ### Uptakes
    q_des = iso_mix(P_low_part,Tfeed,iso,
                    dH_input,Tref_input)
    q_sat_tot = iso_mix(P_high_part,Tfeed,iso,
                        dH_input,Tref_input)
    Dq_tot = q_sat_tot-q_des
    ### Leading component ?
    sat_extent = np.array(yfeed)/Dq_tot # Saturation extent kg/mol
    ind_lead_tot = np.argmax(sat_extent)
    sat_ext_raff = sat_extent[0]
    sat_extent[0] = -10000
    ind_lead = np.argmax(sat_extent)
    
    if ind_lead > 1.95:
        ## CO2 leading case [index = 2]
        yfeed_CO2 = np.array([yfeed[0], yfeed[2]])/(yfeed[0]+yfeed[2])
        P_CO2_part= P_high * yfeed_CO2 # (bar): partial pressure
        q_sat_CO2 = iso_mix(P_CO2_part,Tfeed,[iso[0],iso[2]],
                            [dH[0],dH[2]],[Tref_input[0],Tref_input[2]])
        q_sat_CO2 = np.array([q_sat_CO2[0],0, q_sat_CO2[1]])
        q_lead_pack = [q_sat_CO2[2], q_sat_tot[2], q_des[2]] # CO2 uptakes
        q_follow_pack = [q_sat_tot[1], q_des[1]]
        s_H2S = sat_fraction(yfeed[2],q_lead_pack,yfeed[1],q_follow_pack)
        q_bar_sat = s_H2S*q_sat_tot + (1-s_H2S)*q_sat_CO2
        s = s_H2S
        s_out = np.array([1,s,1])
        #leading_heavy_key = 2

    else:
        ## H2S leading case [index = 1]
        yfeed_H2S = np.array([yfeed[0], yfeed[1]])/(yfeed[0]+yfeed[1])
        P_H2S_part= P_high * yfeed_H2S # (bar): partial pressure
        q_sat_H2S = iso_mix(P_H2S_part,Tfeed,[iso[0],iso[1]],
                            [dH[0],dH[1]],[Tref_input[0],Tref_input[1]])
        q_sat_H2S = np.array([q_sat_H2S[0],q_sat_H2S[1],0])
        q_lead_pack = [q_sat_H2S[1],q_sat_tot[1],q_des[1]]
        q_follow_pack = [q_sat_tot[2],q_des[2]]
        s_CO2 = sat_fraction(yfeed[1],q_lead_pack,yfeed[2],q_follow_pack)
        q_bar_sat = s_CO2*q_sat_tot + (1-s_CO2)*q_sat_H2S
        s = s_CO2
        s_out = np.array([1,1,s])
        #leading_heavy_key = 1
    Dq_exhaust = q_bar_sat - q_des
    x_out = np.array(Dq_exhaust)/np.sum(Dq_exhaust)
    
    sat_extent[0] = sat_ext_raff 
    return x_out,s_out,ind_lead_tot

#### End: x2x = Single Cycle ####

#### Start: Rec = cyclic steady state #### 
def Rec(x_ini,P_high,P_low,iso_input, dH_input, Tref_input,yfeed,Tfeed):
  

    def x_obj(x_in):
        x_exh, s_f,i_lead = x2x(x_in,P_high,P_low,
                                iso_input,dH_input,Tref_input,
                                yfeed,Tfeed)
        return (x_exh-x_in)**2*100
    x00 = x_ini
    solx = optim.least_squares(x_obj,x00,bounds= (0,1))
    x_exh = solx.x

    x_exh, s_f,i_lead = x2x(x_exh,P_high,P_low,
                            iso_input, dH_input, Tref_input, 
                            yfeed,Tfeed)
    if i_lead == 0:
        if x_exh[1]>x_exh[2]:
            y_hvy = yfeed[1]
            x_hvy = x_exh[1]
        else:
            y_hvy = yfeed[2]
            x_hvy = x_exh[2]
        rec = (1-y_hvy/yfeed[0]*x_exh[0]/x_hvy)*100
        return rec,s_f
    else:
        return 0,0


#### End: Rec = cyclic steady state ####

if __name__ == '__main__':
    1
