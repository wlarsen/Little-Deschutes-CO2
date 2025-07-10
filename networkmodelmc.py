
import pandas as pd
import numpy as np
import hydroeval as he
import scipy
import time
import os

from helperfunctions import *

df_priors = pd.read_csv('modeloutputs/test2_priors.csv',index_col=0)

df = pd.read_csv('map/network_merged.csv',index_col=0)
df =df.rename(columns={'d13C_permil':'dic_delta_meas','dic_calc':'dic_meas','pco2':'pco2_meas','Rn_bq_m3':'rn_meas'})

strahlerlist = [1,2,3,4]

def networkmodelmc(index):
    upstreamnetwork = pd.read_csv('map/network_upstream.csv',index_col=0)

    ### set random seed (multiprocessing workers inherit same seed)
    np.random.seed((os.getpid() * int(time.time())) % 123456789)

    ### some constants:
    gw_alk = np.mean(df['Alkalinity_meq_L'])/1000

    co2_delta_atm = -9 
    pCO2_atm = 420
    delta_x=10   
    mintemp=np.mean(df['temperature_filled'])
    
    # random sampling of parameters
    mcresults = pd.DataFrame()
    # k600
    a = np.random.uniform(df_priors.loc['k600_a','min'],df_priors.loc['k600_a','max'])
    b = np.random.uniform(df_priors.loc['k600_b','min'],df_priors.loc['k600_b','max'])
    # source chemistry
    gw_rn = np.random.uniform(df_priors.loc['gw_rn','min'],df_priors.loc['gw_rn','max'])
    gw_DIC = np.random.uniform(df_priors.loc['gw_DIC','min'],df_priors.loc['gw_DIC','max'])/1000
    wet_DIC = np.random.uniform(df_priors.loc['wet_DIC','min'],df_priors.loc['wet_DIC','max'])/1000
    gw_CO2_delta = np.random.uniform(df_priors.loc['gw_CO2_delta','min'],df_priors.loc['gw_CO2_delta','max'])
    wet_CO2_delta = np.random.uniform(df_priors.loc['gw_CO2_delta','min'],df_priors.loc['gw_CO2_delta','max'])
    
    # calc dic delta value from the other variables
    gw_DIC_delta = epsilon_co2_dic(gw_DIC,gw_alk,gw_CO2_delta,mintemp)
    wet_DIC_delta= epsilon_co2_dic(wet_DIC,gw_alk,wet_CO2_delta,mintemp)
    # calc pco2 from other variables
    gw_pco2 = phfromalk(gw_DIC,gw_alk,mintemp)[-1]
    wet_pco2 = phfromalk(wet_DIC,gw_alk,mintemp)[-1]

    # save all of these
    mcresults.loc[index,'a'] = a
    mcresults.loc[index,'b'] = b
    mcresults.loc[index,'gw_rn'] = gw_rn
    mcresults.loc[index,'gw_DIC'] = gw_DIC
    mcresults.loc[index,'wet_DIC'] = wet_DIC
    mcresults.loc[index,'gw_d13c'] = gw_DIC_delta
    mcresults.loc[index,'wet_d13c'] = wet_DIC_delta
    mcresults.loc[index,'gw_d13c_co2'] = gw_CO2_delta
    mcresults.loc[index,'wet_d13c_co2'] = wet_CO2_delta
    mcresults.loc[index,'gw_pco2'] = gw_pco2
    mcresults.loc[index,'wet_pco2'] = wet_pco2

    # initialize
    df_iter = df.copy()
    lastcell = pd.DataFrame()
    
    df_iter.loc[:,'k600_mc'] = a*(df['velocity']*-df['slope'])**b

    df_iter['co2_atm'] = henryco2(df_iter['temperature_filled'])  * pCO2_atm/10**6# this is 2.14e-5 Saccardi Winnick used c_atm = 2.13e-5
    df_iter['co2_iso_atm'] = df_iter['co2_atm']*delta2r(co2_delta_atm)

    s_co2 = schmidtco2(df_iter['temperature_filled'])
    s_rn = schmidtrn(df_iter['temperature_filled'])
    df_iter['kco2'] = (df_iter[f'k600_mc']/(600/s_co2)**-0.5)/(24*60*60) # m/day to m/s
    df_iter['krn'] = (df_iter[f'k600_mc']/(600/s_rn)**-0.5)/(24*60*60)
    df_iter.loc[np.isnan(df_iter['kco2']),'kco2'] = np.nanmin(df_iter.loc[df_iter['kco2']>0,'kco2'])
    df_iter.loc[np.isnan(df_iter['krn']),'krn'] = np.nanmin(df_iter.loc[df_iter['krn']>0,'krn'])

    toponums = list(set(df['topo']))

    for t in list(reversed(toponums)):

        tempdf_topo = df_iter.loc[df_iter['topo']==t,:]

        segmentlist = set(tempdf_topo['stream'])

        for seg in segmentlist:
            tempdf = tempdf_topo.loc[tempdf_topo['stream']==seg,:]

            temp_strahler = tempdf['strahler'].values[0]

            tempindex = tempdf_topo.loc[tempdf_topo['stream']==seg,:].index

            if temp_strahler>1:
                tribs = set(df_iter.loc[df_iter['next_stream']==seg,'stream'])
                tribs=upstreamnetwork[upstreamnetwork['down']==seg].index

                if any(np.isnan(lastcell.loc[lastcell['stream'].isin(tribs),'dic'])):
                    print(f'AHHHHH {seg} tribs are bad :0')

                # discharge weighted mixtures
                mix_DIC = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'dic']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_DIC_delta = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'dic_delta']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_rn = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'rn']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_DIC = np.nanmax([mix_DIC,gw_alk+0.00002]) # in case the downstream mixture has higher alk than tribs

                df_iter.loc[tempindex[0],'dic']= mix_DIC
                df_iter.loc[tempindex[0],'co2'] = phfromalk(mix_DIC,gw_alk,df_iter.loc[tempindex[0],'temperature_filled'])[1][0]
                df_iter.loc[tempindex[0],'dic_iso']= mix_DIC*delta2r(mix_DIC_delta)
                df_iter.loc[tempindex[0],'co2_iso']= delta2r(epsilon_dic_co2(mix_DIC,gw_alk,mix_DIC_delta,df_iter.loc[tempindex[0],'temperature_filled']))*df_iter.loc[tempindex[0],'co2']
                df_iter.loc[tempindex[0],'rn']= mix_rn


            else: # if first order, first value is just groundwater
                df_iter.loc[tempindex[0],'dic']= gw_DIC/1
                df_iter.loc[tempindex[0],'co2'] = phfromalk(gw_DIC,gw_alk,df_iter.loc[tempindex[0],'temperature_filled'])[1][0]
                df_iter.loc[tempindex[0],'dic_iso']= gw_DIC*delta2r(gw_DIC_delta)
                df_iter.loc[tempindex[0],'co2_iso']= delta2r(epsilon_dic_co2(gw_DIC,gw_alk,gw_DIC_delta,df_iter.loc[tempindex[0],'temperature_filled']))*df_iter.loc[tempindex[0],'co2']
                df_iter.loc[tempindex[0],'rn']= gw_rn

            for i in tempindex[1:]:
                if ~np.isnan(df_iter.loc[i,'wetland']): # if wetland is present, add some pz DIC depending on frac wet
                    c_gw = wet_DIC
                    c_delta_gw = wet_DIC_delta
                    c_iso_gw = c_gw*delta2r(c_delta_gw)
                else:
                    c_gw = gw_DIC
                    c_delta_gw = gw_DIC_delta
                    c_iso_gw = c_gw*delta2r(c_delta_gw)

                df_iter.loc[i,'deltaq'] = df_iter.loc[i,'discharge']-df_iter.loc[i-1,'discharge']

                # Rn steady state solution: in - out
                rn_ss_in = (df_iter.loc[i-1,'rn']*df_iter.loc[i-1,'discharge']+gw_rn*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                rn_ss_out = (df_iter.loc[i,'krn']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'rn']-0)*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available (0 in atm)
                if rn_ss_out>=rn_ss_in:
                    rn_ss_out = rn_ss_in # substracting a small number so its early equivalent but result is not zero. might not be necessary

                df_iter.loc[i,'rn'] = rn_ss_in-rn_ss_out

                # CO2 steady state solution: in - out
                dic_ss_in = (df_iter.loc[i-1,'dic']*df_iter.loc[i-1,'discharge']+c_gw*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                co2_ss_out = (df_iter.loc[i,'kco2']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm'])*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available (some co2 in atm)
                if co2_ss_out>=df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm']:
                    co2_ss_out = df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm'] # substracting a small number so its early equivalent but result is not zero. might not be necessary

                df_iter.loc[i,'dic'] = dic_ss_in-co2_ss_out

                # now copy co2_ss_out into flux of co2 out... right now it is normalized to discharge (m/s / m) *(mol/L) * m*m^2 / (m3/s)  -> mol/L * m^3/s  /(m^3 /s) -> mol/L.
                # need to multiply discharge back to get mol/L * m^3 / s, then multiply by 1000 to convert mol/L to mol/m^3 to get mol/s, then convert to g/day
                df_iter.loc[i,'fco2'] = co2_ss_out*df_iter.loc[i,'discharge']*1000*12*(24*60*60)
                df_iter.loc[i,'fco2_perarea'] = co2_ss_out*df_iter.loc[i,'discharge']*1000*12*(24*60*60)/(delta_x*df_iter.loc[i,'width'])

                # now speciate using DIC and alk
                [ph,co2,bicarb,carb,pco2] = phfromalk(df_iter.loc[i,'dic'],gw_alk,df_iter.loc[i,'temperature_filled'])
                df_iter.loc[i,'co2'] = co2

                ## iso
                # isotope CO2 steady state solution: in - out
                dic_iso_ss_in = (df_iter.loc[i-1,'dic_iso']*df_iter.loc[i-1,'discharge']+c_iso_gw*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                co2_iso_ss_out = (df_iter.loc[i,'kco2']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm'])*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available
                if co2_iso_ss_out>=df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm']:
                    co2_iso_ss_out = df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm'] # substracting a small number so its early equivalent but result is not zero. might not be necessary


                df_iter.loc[i,'dic_iso'] = dic_iso_ss_in-co2_iso_ss_out

                df_iter.loc[i,'dic_delta'] = r2delta(df_iter.loc[i,'dic_iso']/df_iter.loc[i,'dic'])

                df_iter.loc[i,'co2_delta'] = epsilon_dic_co2(df_iter.loc[i,'dic'],gw_alk,df_iter.loc[i,'dic_delta'],df_iter.loc[i,'temperature_filled'])

                df_iter.loc[i,'co2_iso'] = delta2r(df_iter.loc[i,'co2_delta'])*df_iter.loc[i,'co2']


                if i==tempindex[-1]: # the last one
                    lastcell = pd.concat([lastcell,df_iter.loc[[i],:]])



    df_iter['pco2_mod'] = df_iter['co2']/henryco2(df_iter['temperature_filled'])*10**6 

    ### evaluations! using hydroeval
    mcresults.loc[index,'rmse_pco2'] = he.evaluator(he.rmse,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)    
    mcresults.loc[index,'nse_pco2'] = he.evaluator(he.nse,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)
    mcresults.loc[index,'kge_pco2'] = he.evaluator(he.kge,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)[0]

    mcresults.loc[index,'rmse_rn'] = he.evaluator(he.rmse,df_iter['rn'].values,df_iter['rn_meas'].values)    
    mcresults.loc[index,'nse_rn'] = he.evaluator(he.nse,df_iter['rn'].values,df_iter['rn_meas'].values)
    mcresults.loc[index,'kge_rn'] = he.evaluator(he.kge,df_iter['rn'].values,df_iter['rn_meas'].values)[0]

    mcresults.loc[index,'rmse_d13c'] = he.evaluator(he.rmse,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)    
    mcresults.loc[index,'nse_d13c'] = he.evaluator(he.nse,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)
    mcresults.loc[index,'kge_d13c'] = he.evaluator(he.kge,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)[0]

    ###
    mcresults.loc[index,'mae_pco2'] = np.nanmean(np.abs(df_iter['pco2_mod'].values-df_iter['pco2_meas'].values))   
    mcresults.loc[index,'mae_rn'] = np.nanmean(np.abs(df_iter['rn'].values-df_iter['rn_meas'].values))   
    mcresults.loc[index,'mae_d13c'] = np.nanmean(np.abs(df_iter['dic_delta'].values-df_iter['dic_delta_meas'].values))   

    # compute the total fco2
    mcresults.loc[index,'fco2'] = np.sum(df_iter['fco2'])

    # normalize fco2 by area
    networkarea = np.sum(df['width']*delta_x)
    mcresults.loc[index,'fco2_area'] = mcresults.loc[index,'fco2'] /networkarea

    for s in strahlerlist:
        mcresults.loc[index,f'fco2_{s}'] = np.sum(df_iter.loc[df_iter['strahler']==s,'fco2'])
        tempnetworkarea = np.sum(df_iter.loc[df_iter['strahler']==s,'width']*delta_x)
        mcresults.loc[index,f'fco2_area_{s}'] = mcresults.loc[index,f'fco2_{s}'] /tempnetworkarea

    return mcresults

# this is for running existing paramter sets later (like the top whatever number)
def networkmodel_notmc(inputs,clip=False):
    upstreamnetwork = pd.read_csv('map/network_upstream.csv',index_col=0)

    index=0
    
    [a,b,gw_DIC,wet_DIC,gw_CO2_delta,wet_CO2_delta,gw_rn] = inputs
    

    ### some constants:
    gw_alk = np.mean(df['Alkalinity_meq_L'])/1000

    co2_delta_atm = -9 
    pCO2_atm = 420
    delta_x=10   
    mintemp=np.min(df['temperature_filled'])
    
    # initialize
    results = pd.DataFrame()

    gw_DIC_delta = epsilon_co2_dic(gw_DIC,gw_alk,gw_CO2_delta,mintemp)
    wet_DIC_delta= epsilon_co2_dic(wet_DIC,gw_alk,wet_CO2_delta,mintemp)
    gw_pco2 = phfromalk(gw_DIC,gw_alk,mintemp)[-1]
    wet_pco2 = phfromalk(wet_DIC,gw_alk,mintemp)[-1]

    # save
    results.loc[index,'a'] = a
    results.loc[index,'b'] = b
    results.loc[index,'gw_rn'] = gw_rn
    results.loc[index,'gw_DIC'] = gw_DIC
    results.loc[index,'wet_DIC'] = wet_DIC
    results.loc[index,'gw_d13c'] = gw_DIC_delta
    results.loc[index,'wet_d13c'] = wet_DIC_delta
    results.loc[index,'gw_d13c_co2'] = gw_CO2_delta
    results.loc[index,'wet_d13c_co2'] = wet_CO2_delta
    results.loc[index,'gw_pco2'] = gw_pco2
    results.loc[index,'wet_pco2'] = wet_pco2

    # initialize
    df_iter = df.copy()
    
    lastcell = pd.DataFrame()
    
    if clip: # if clip, limit our dataset to the cells that arent clipped
        df_iter = df_iter.loc[df_iter['clip']==0,:]

    df_iter['co2_atm'] = henryco2(df_iter['temperature_filled'])  * pCO2_atm/10**6
    df_iter['co2_iso_atm'] = df_iter['co2_atm']*delta2r(co2_delta_atm)

    df_iter.loc[:,'k600_mc'] = a*(df['velocity']*-df['slope'])**b

    s_co2 = schmidtco2(df_iter['temperature_filled'])
    s_rn = schmidtrn(df_iter['temperature_filled'])
    df_iter['kco2'] = (df_iter[f'k600_mc']/(600/s_co2)**-0.5)/(24*60*60) # m/day to m/s
    df_iter['krn'] = (df_iter[f'k600_mc']/(600/s_rn)**-0.5)/(24*60*60)
    df_iter.loc[np.isnan(df_iter['kco2']),'kco2'] = np.nanmin(df_iter.loc[df_iter['kco2']>0,'kco2'])
    df_iter.loc[np.isnan(df_iter['krn']),'krn'] = np.nanmin(df_iter.loc[df_iter['krn']>0,'krn'])

    toponums = list(set(df_iter['topo']))

    for t in list(reversed(toponums)):

        tempdf_topo = df_iter.loc[df_iter['topo']==t,:]

        segmentlist = set(tempdf_topo['stream'])

        for seg in segmentlist:
            tempdf = tempdf_topo.loc[tempdf_topo['stream']==seg,:]

            temp_strahler = tempdf['strahler'].values[0]

            tempindex = tempdf_topo.loc[tempdf_topo['stream']==seg,:].index

            # how to see if a stream is the furthest upstream after clipping?
            # make a boolean here
            if clip:
                furthestupstream = tempdf['clip_strahler1'].values[0]
            else:
                furthestupstream = temp_strahler==1

            if not furthestupstream:
                tribs = set(df_iter.loc[df_iter['next_stream']==seg,'stream'])
                tribs=upstreamnetwork[upstreamnetwork['down']==seg].index

                if any(np.isnan(lastcell.loc[lastcell['stream'].isin(tribs),'dic'])):
                    print(f'AHHHHH {seg} tribs are bad :0')

                # discharge weighted mixtures
                mix_DIC = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'dic']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_DIC_delta = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'dic_delta']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_rn = np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'rn']*lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])/np.sum(lastcell.loc[lastcell['stream'].isin(tribs),'discharge'])
                mix_DIC = np.nanmax([mix_DIC,gw_alk+0.00002]) # in case the downstream mixture has higher alk than tribs

                df_iter.loc[tempindex[0],'dic']= mix_DIC
                df_iter.loc[tempindex[0],'co2'] = phfromalk(mix_DIC,gw_alk,df_iter.loc[tempindex[0],'temperature_filled'])[1][0]
                df_iter.loc[tempindex[0],'dic_iso']= mix_DIC*delta2r(mix_DIC_delta)
                df_iter.loc[tempindex[0],'co2_iso']= delta2r(epsilon_dic_co2(mix_DIC,gw_alk,mix_DIC_delta,df_iter.loc[tempindex[0],'temperature_filled']))*df_iter.loc[tempindex[0],'co2']
                df_iter.loc[tempindex[0],'rn']= mix_rn

            else: # if first order, first value is just groundwater
                df_iter.loc[tempindex[0],'dic']= gw_DIC/1
                df_iter.loc[tempindex[0],'co2'] = phfromalk(gw_DIC,gw_alk,df_iter.loc[tempindex[0],'temperature_filled'])[1][0]
                df_iter.loc[tempindex[0],'dic_iso']= gw_DIC*delta2r(gw_DIC_delta)
                df_iter.loc[tempindex[0],'co2_iso']= delta2r(epsilon_dic_co2(gw_DIC,gw_alk,gw_DIC_delta,df_iter.loc[tempindex[0],'temperature_filled']))*df_iter.loc[tempindex[0],'co2']
                df_iter.loc[tempindex[0],'rn']= gw_rn

            for i in tempindex[1:]:
                if ~np.isnan(df_iter.loc[i,'wetland']): # if wetland is present, add some pz DIC depending on frac wet
                    c_gw = wet_DIC
                    c_delta_gw = wet_DIC_delta
                    c_iso_gw = c_gw*delta2r(c_delta_gw)
                else:
                    c_gw = gw_DIC
                    c_delta_gw = gw_DIC_delta
                    c_iso_gw = c_gw*delta2r(c_delta_gw)

                df_iter.loc[i,'deltaq'] = df_iter.loc[i,'discharge']-df_iter.loc[i-1,'discharge']

                # Rn steady state solution: in - out
                rn_ss_in = (df_iter.loc[i-1,'rn']*df_iter.loc[i-1,'discharge']+gw_rn*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                rn_ss_out = (df_iter.loc[i,'krn']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'rn']-0)*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available (0 in atm)
                if rn_ss_out>=rn_ss_in:
                    rn_ss_out = rn_ss_in # substracting a small number so its early equivalent but result is not zero. might not be necessary

                df_iter.loc[i,'rn'] = rn_ss_in-rn_ss_out
                # in this version of the function, save Rn in and Rn out
                df_iter.loc[i,'rn_in'] = rn_ss_in
                df_iter.loc[i,'rn_out'] = rn_ss_out

                # CO2 steady state solution: in - out
                dic_ss_in = (df_iter.loc[i-1,'dic']*df_iter.loc[i-1,'discharge']+c_gw*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                co2_ss_out = (df_iter.loc[i,'kco2']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm'])*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available (some co2 in atm)
                if co2_ss_out>=df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm']:
                    co2_ss_out = df_iter.loc[i-1,'co2']-df_iter.loc[i-1,'co2_atm'] # substracting a small number so its early equivalent but result is not zero. might not be necessary

                df_iter.loc[i,'dic'] = dic_ss_in-co2_ss_out

                # now copy co2_ss_out into flux of co2 out... right now it is normalized to discharge (m/s / m) *(mol/L) * m*m^2 / (m3/s)  -> mol/L * m^3/s  /(m^3 /s) -> mol/L.
                # need to multiply discharge back to get mol/L * m^3 / s, then multiply by 1000 to convert mol/L to mol/m^3 to get mol/s, then convert to g/day
                df_iter.loc[i,'fco2'] = co2_ss_out*df_iter.loc[i,'discharge']*1000*12*(24*60*60)
                df_iter.loc[i,'fco2_perarea'] = co2_ss_out*df_iter.loc[i,'discharge']*1000*12*(24*60*60)/(delta_x*df_iter.loc[i,'width'])

                # now speciate using DIC and alk
                [ph,co2,bicarb,carb,pco2] = phfromalk(df_iter.loc[i,'dic'],gw_alk,df_iter.loc[i,'temperature_filled'])
                df_iter.loc[i,'co2'] = co2

                ## iso
                # isotope CO2 steady state solution: in - out
                dic_iso_ss_in = (df_iter.loc[i-1,'dic_iso']*df_iter.loc[i-1,'discharge']+c_iso_gw*df_iter.loc[i,'deltaq'])/(df_iter.loc[i,'discharge'])
                co2_iso_ss_out = (df_iter.loc[i,'kco2']/df_iter.loc[i,'depth'])*(df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm'])*delta_x*df_iter.loc[i-1,'xarea']/(df_iter.loc[i,'discharge'])

                # we cant degas more than what is available
                if co2_iso_ss_out>=df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm']:
                    co2_iso_ss_out = df_iter.loc[i-1,'co2_iso']-df_iter.loc[i-1,'co2_iso_atm'] # substracting a small number so its early equivalent but result is not zero. might not be necessary

                df_iter.loc[i,'dic_iso'] = dic_iso_ss_in-co2_iso_ss_out

                df_iter.loc[i,'dic_delta'] = r2delta(df_iter.loc[i,'dic_iso']/df_iter.loc[i,'dic'])

                df_iter.loc[i,'co2_delta']= epsilon_dic_co2(df_iter.loc[i,'dic'],gw_alk,df_iter.loc[i,'dic_delta'],df_iter.loc[i,'temperature_filled'])

                df_iter.loc[i,'co2_iso'] = delta2r(df_iter.loc[i,'co2_delta'])*df_iter.loc[i,'co2']


                if i==tempindex[-1]: # the last one
                    lastcell = pd.concat([lastcell,df_iter.loc[[i],:]])

    df_iter['pco2_mod'] = df_iter['co2']/henryco2(df_iter['temperature_filled'])*10**6  # save modeled pco2 for later

    ### evaluations! using hydroeval
    results.loc[index,'rmse_pco2'] = he.evaluator(he.rmse,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)    
    results.loc[index,'nse_pco2'] = he.evaluator(he.nse,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)
    results.loc[index,'kge_pco2'] = he.evaluator(he.kge,df_iter['pco2_mod'].values,df_iter['pco2_meas'].values)[0]

    results.loc[index,'rmse_rn'] = he.evaluator(he.rmse,df_iter['rn'].values,df_iter['rn_meas'].values)    
    results.loc[index,'nse_rn'] = he.evaluator(he.nse,df_iter['rn'].values,df_iter['rn_meas'].values)
    results.loc[index,'kge_rn'] = he.evaluator(he.kge,df_iter['rn'].values,df_iter['rn_meas'].values)[0]

    results.loc[index,'rmse_d13c'] = he.evaluator(he.rmse,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)    
    results.loc[index,'nse_d13c'] = he.evaluator(he.nse,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)
    results.loc[index,'kge_d13c'] = he.evaluator(he.kge,df_iter['dic_delta'].values,df_iter['dic_delta_meas'].values)[0]

    ###
    # mean absolute error
    results.loc[index,'mae_pco2'] = np.nanmean(np.abs(df_iter['pco2_mod'].values-df_iter['pco2_meas'].values))   
    results.loc[index,'mae_rn'] = np.nanmean(np.abs(df_iter['rn'].values-df_iter['rn_meas'].values))   
    results.loc[index,'mae_d13c'] = np.nanmean(np.abs(df_iter['dic_delta'].values-df_iter['dic_delta_meas'].values))   
    
    # compute the total fco2
    results.loc[index,'fco2'] = np.sum(df_iter['fco2'])

    # normalize fco2 by area
    networkarea = np.sum(df['width']*delta_x)
    results.loc[index,'fco2_area'] = results.loc[index,'fco2'] /networkarea

    for s in strahlerlist:
        results.loc[index,f'fco2_{s}'] = np.sum(df_iter.loc[df_iter['strahler']==s,'fco2'])
        tempnetworkarea = np.sum(df_iter.loc[df_iter['strahler']==s,'width']*delta_x)
        results.loc[index,f'fco2_area_{s}'] = results.loc[index,f'fco2_{s}'] /tempnetworkarea

    return [results,df_iter]
