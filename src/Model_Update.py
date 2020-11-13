# -*- coding: utf-8 -*-
"""
Model Update

Runs OLS model for every meter and saves predictions
"""

import numpy as np
import os
import statsmodels.api as sm
import pandas as pd
import datetime
import pytz
from pytz import timezone
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import scipy
from scipy import stats

dirname = '../data/'
inpath = os.path.join(dirname,'Analysis.csv') 

#read in master file, drop unneeded column
master = pd.read_csv(inpath)#.drop(columns=['Unnamed: 0'])




#list of y variables
list1 = [i for i in master.columns if i.startswith('d_') or i.startswith('i_') or i.endswith('_o')]

ylist = [i for i in master.columns if i.startswith('d_')]

covar = np.setdiff1d(master.columns, list1).tolist()

#now we need building specific variables to be extracted
meternames = [i[2:] for i in master.columns if i.startswith('d_')]

#occ meters
occbuilds = [i[:-2] for i in master.columns if i.endswith('_o')]

#meters without occupancy
notocc = np.setdiff1d(meternames,occbuilds)


meternames = ['Baseball',
 #'BaseballStudentRecPark',
 'BryanBldg',
 'CampusSupplyStore',
 'CarmichaelBldg',
 'ChemicalSafety',
 'ConeArtBldg',
 #'ConeResHall',
 'CurryBldg',
 'Eberhart',
 'ElliottUnivCenter',
# 'Foust',
 'Graham',
 'GrayHome',
 'GroganHall',
 'GuilfordResHall',
 'HHP',
 'HHPSoccerField',
 'JLibraryOldEastAtDoor',
 'JLibraryOldEastPanelP',
 'JacksonLibChiller',
 'MaryFoustHall',
# 'McIverBldgCenterVault',
 #'McIverBldgNE',
 #'McIverBldgNW',
 #'McIverDeckChillerSvc2',
 'McIverDeckChillerSvc3',
 'McIverStPrkgDeck',
 'MooreNursingBldg',
 'MooreStrongResHall',
 'MooreStrongSiteLts011',
 'MooreStrongSiteLts127',
 'Mossman',
 'OaklandAveParkingDeck',
 'PettyBldg',
 #'RagsdaleMendenhall',
 'ReynoldsHall',
 'SchOfMusic',
 'SinkGroundsBldg',
 'SinkOffices',
 'Spencer',
 'SpringGardenAptsCStore',
 'SteamPlant',
 'StoneBuilding',
 'StudentRecCenterMain',
 'SullivanScience',
 'TaylorTheatre',
 'TowerVillage',
 'VisitorCenter',
 'WalkerDeck',
 'WeilWinfield',
 'BaseballFieldSupportBldg',
 'BaseballStadiumPrkLights',
 'BaseballStadiumRecFieldLights',
 #'BaseballStadiumSiteLights',
 'BaseballStadiumWellPump',
 'BryanDataCenter',
 #'FacultyCenter',
 'GoveSiteLighting',
 'HHPSoftballStadium',
 'HinshawSiteLights',
 'JacksonLibraryTower',
 'McIverStPrkgDeckLights',
 'McNuttDataCenterEDPA',
 'McNuttDataCenterEDPB',
 'MooreNursingAnnex',
 'MooreStrongSports',
 'NDChildCare',
 'RagsdaleMendenhallLights',
 'SRC_1H1A',
 'SRC_2H1A',
'SRC_2H3A',
'SRC_2H4A',
'SRC_2L1B',
'SRC_3H3A',
 'SRC_CUB1',
 'SRC_GenATS',
 'SRC_Main',
 'SRC_Pool',
 'SRC_TX_2L1ABC',
 'SRC_TX_2L3ABC',
 'SRC_TX_2L4ABC',
 'SRC_TX_3L3AB',
 'SpencerHonors',
 'SpencerSiteLts',
 'SpringGardenPrkgDeck',
 'StudentRecCenterBasement',
 'StudentRecSiteLts222',
 'TennisCourts',
 'UnderpassNorth',
 'WalkerDeckLighting']

for i in meternames:
    
    # for i in occbuilds
    meter = i
    name = ['d_' + meter]   
    name1 = 'd_' + meter      #LOOP OVER THIS PART
    occname = [meter + '_o']
    
    
    
    #included vars
    incvars = name + covar
    
    #including occupancy if it's one of the appropriate meters
    if any(i in s for s in occbuilds):
        incvars = incvars + occname
        
        #removing jan-may of 2015, because we dont have occupancy data
        master = master.drop(master[(master.Month < 6) & (master.Year == 2015)].index)
        
        
        
    #saving with appropriate vars
    df = master[incvars]
    
    df = df.dropna(axis=0)
    df = df.reset_index(drop=True)
        
        
    #adding constant
    df['Constant'] = 1
    df.index = pd.RangeIndex(len(df.index))
    
    #defining in and out of sample
    Oin = df[df.Year != 2020]
    Oout = df[df.Year == 2020]
    
    #dropping observations with large outliers (> 5 standard deviations larger than the mean)
    # we don't want to use these numbers to predict, but we do want to keep outliers in the out of sample dataset
    #Oin = Oin[Oin[name] < (Oin[name].mean() + Oin[name].std()*5)]
    #the line above takes a really long time. Not quite sure what happened. Might be the virtual desktop - it was working fine before
    
    master1 = Oin.append(Oout)
    
    #in sample
    Xin = Oin.drop(columns = [name1,'CoolDD','CoolDD2','HeatDD','HeatDD2','Datetime','Weekday','Weekend','Weekday_1','hour_0','Month_1','hour','Month','Year','Weekday_1_hour_0',
                          'Weekday_1_hour_0_sem',
     'Weekday_1_hour_1',
     'Weekday_1_hour_10',
     'Weekday_1_hour_10_sem',
     'Weekday_1_hour_11',
     'Weekday_1_hour_11_sem',
     'Weekday_1_hour_12',
     'Weekday_1_hour_12_sem',
     'Weekday_1_hour_13',
     'Weekday_1_hour_13_sem',
     'Weekday_1_hour_14',
     'Weekday_1_hour_14_sem',
     'Weekday_1_hour_15',
     'Weekday_1_hour_15_sem',
     'Weekday_1_hour_16',
     'Weekday_1_hour_16_sem',
     'Weekday_1_hour_17',
     'Weekday_1_hour_17_sem',
     'Weekday_1_hour_18',
     'Weekday_1_hour_18_sem',
     'Weekday_1_hour_19',
     'Weekday_1_hour_19_sem',
     'Weekday_1_hour_1_sem',
     'Weekday_1_hour_2',
     'Weekday_1_hour_20',
     'Weekday_1_hour_20_sem',
     'Weekday_1_hour_21',
     'Weekday_1_hour_21_sem',
     'Weekday_1_hour_22',
     'Weekday_1_hour_22_sem',
     'Weekday_1_hour_23',
     'Weekday_1_hour_23_sem',
     'Weekday_1_hour_2_sem',
     'Weekday_1_hour_3',
     'Weekday_1_hour_3_sem',
     'Weekday_1_hour_4',
     'Weekday_1_hour_4_sem',
     'Weekday_1_hour_5',
     'Weekday_1_hour_5_sem',
     'Weekday_1_hour_6',
     'Weekday_1_hour_6_sem',
     'Weekday_1_hour_7',
     'Weekday_1_hour_7_sem',
     'Weekday_1_hour_8',
     'Weekday_1_hour_8_sem',
     'Weekday_1_hour_9',
     'Weekday_1_hour_9_sem'])
    Yin = Oin[name]
    
    #out of sample
    Xout = Oout.drop(columns = [name1,'CoolDD','CoolDD2','HeatDD','HeatDD2','Datetime','Weekday','Weekend','Weekday_1','hour_0','Month_1','hour','Month','Year','Weekday_1_hour_0',
                            'Weekday_1_hour_0_sem',
     'Weekday_1_hour_1',
     'Weekday_1_hour_10',
     'Weekday_1_hour_10_sem',
     'Weekday_1_hour_11',
     'Weekday_1_hour_11_sem',
     'Weekday_1_hour_12',
     'Weekday_1_hour_12_sem',
     'Weekday_1_hour_13',
     'Weekday_1_hour_13_sem',
     'Weekday_1_hour_14',
     'Weekday_1_hour_14_sem',
     'Weekday_1_hour_15',
     'Weekday_1_hour_15_sem',
     'Weekday_1_hour_16',
     'Weekday_1_hour_16_sem',
     'Weekday_1_hour_17',
     'Weekday_1_hour_17_sem',
     'Weekday_1_hour_18',
     'Weekday_1_hour_18_sem',
     'Weekday_1_hour_19',
     'Weekday_1_hour_19_sem',
     'Weekday_1_hour_1_sem',
     'Weekday_1_hour_2',
     'Weekday_1_hour_20',
     'Weekday_1_hour_20_sem',
     'Weekday_1_hour_21',
     'Weekday_1_hour_21_sem',
     'Weekday_1_hour_22',
     'Weekday_1_hour_22_sem',
     'Weekday_1_hour_23',
     'Weekday_1_hour_23_sem',
     'Weekday_1_hour_2_sem',
     'Weekday_1_hour_3',
     'Weekday_1_hour_3_sem',
     'Weekday_1_hour_4',
     'Weekday_1_hour_4_sem',
     'Weekday_1_hour_5',
     'Weekday_1_hour_5_sem',
     'Weekday_1_hour_6',
     'Weekday_1_hour_6_sem',
     'Weekday_1_hour_7',
     'Weekday_1_hour_7_sem',
     'Weekday_1_hour_8',
     'Weekday_1_hour_8_sem',
     'Weekday_1_hour_9',
     'Weekday_1_hour_9_sem'])
    Yout = Oout[name]
    
    # all X and Y
    Xall = Xin.append(Xout)
    Yall = Yin.append(Yout)


#simple OLS model
    model = sm.OLS(Yin,Xin)
    result1 =model.fit()
    
#saving predictions
    pred = result1.get_prediction(Xall)
    
    allpred = pred.summary_frame()
    
    #reset index before merge
    allpred = allpred.reset_index(drop=True)
    Yall = Yall.reset_index(drop=True)
    
    #Merge and name columns
    allpred = pd.concat([Yall,allpred],axis=1,join='outer')
    allpred.columns= ['Actual','Predicted','mean_se','mean_ci_lower','mean_ci_upper','obs_ci_lower','obs_ci_upper']
    
    
    #merging master data set on
    master1 = master1.reset_index(drop=True)
    
    allpred = pd.concat([allpred,master1],axis=1,join='outer')
    
    allpred[['Actual','Predicted','obs_ci_lower','obs_ci_upper','Datetime']].to_csv(dirname + '/Analysis/' + meter + '_results.csv',index=False)
 
    
 #
       
    # #save data frames for just the 'in sample' and 'out of sample' predictions using the correct subsets of allpred
    # inpred = allpred[:len(Xin)]
    # outpred = allpred[-len(Xout):]


    # #save variance of error terms
    # inpred['res'] = (inpred['Actual'] - inpred['Predicted'])
    # inpred['res2'] = inpred.res*inpred.res
    
    # k = len(list(Xin)) #number of covariates in regression
    # n = len(Xout) #number of observations in out of sample
    # N = len(Xin) #observations in regression sample
    
    # #variance estimator uses N and all in sample predictions
    # varhat = (1/(N-k-1))*(sum(inpred.res2))


    # #saving matrices needed for standard error estimate

    # #mean of Xs in prediction sample
    # xoutmat = np.asmatrix(Xout)
    # xoutbar = np.asmatrix(Xout.mean())

    # #mean of Xs 'in sample'
    # xinbar = np.asmatrix(Xin.mean()).transpose()

    # #in sample X as matrix
    # xinmat = np.asmatrix(Xin)
    
    # #(x'x)^-1   note that there can't be collinearity in the "train sample". I've taken 2020 out of the in sample prediction.
    # inv = np.linalg.inv(xinmat.transpose() @ xinmat)
    
    # #initializing
    # outpred['predse'] = 1
    
    # #standard error for prediction interval - depends on observation
    # j = 0
    # while j < len(outpred):
    #     outpred.iloc[j,outpred.columns.get_loc('predse')] = np.array(np.sqrt(varhat + varhat*(xoutmat[j] @ inv @ xoutmat[j].transpose())))[0].tolist()[0]
    #     j +=1

    # #t statistic for 95% interval (two tailed) for N-k degrees of freedom
    # t = stats.t.ppf(0.975,n-k)
    
    # #upper and lower limits for individual predictions
    # outpred['lowerint'] = outpred.Predicted - outpred.predse*t 
    # outpred['upperint'] = outpred.Predicted + outpred.predse*t
        
    
    # ###########################
    
    
    # #average predicted and actual by weekday
    
    # gd = pd.DataFrame([outpred.groupby(['Weekday']).Actual.mean(),
    #                    outpred.groupby(['Weekday']).Predicted.mean()]).transpose()

    # #creating appropriate prediction interval
    
    # #initializing columns
    # gd['up'] = 1
    # gd['down'] = 1
    # gd['interval'] = 1
   
    
    # j = 0
    # while j < 7:
        
    #     #saving sample size and t stat
    #     n = len(outpred[outpred.Weekday == (j+1)])
    #     t = stats.t.ppf(0.975,N-k)
        
    #     #saving means from observations for the weekday
    #     xoutbar = np.asmatrix(outpred[outpred.Weekday==(j+1)][list(Xout)].mean())
        
        
    #     if np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] < 0.000000001 :
    #         se = np.sqrt(varhat/n)
    #     elif np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] > 0 :
    #         se = np.array(np.sqrt((varhat/n) + varhat*(xoutbar @ inv @ xoutbar.transpose())))[0].tolist()[0]
        
    #     gd.loc[j+1,'interval'] = se*t
    #     gd.loc[j+1,'up'] = gd.Predicted[j+1] + gd.interval[j+1]
    #     gd.loc[j+1,'down'] = gd.Predicted[j+1] - gd.interval[j+1]
        
    #     del(n,t,xoutbar,se)
    #     j +=1
        
        
    # #saving
    # gd.to_csv(dirname + '\\Data\\Analysis\\' + meter + '_day.csv',index=False)


    # ##############


    # #avg by hour
    # gh = pd.DataFrame([outpred.groupby(['hour']).Actual.mean(),
    #                outpred.groupby(['hour']).Predicted.mean()]).transpose()

    # #initializing columns
    # gh['up'] = 1
    # gh['down'] = 1
    # gh['interval'] = 1
    
    # j=0
    # while j < 24:

    #     #saving sample size and t stat
    #     n = len(outpred[outpred.hour == j])
    #     t = stats.t.ppf(0.975,N-k)  # small n should be used but isn't always large enough. This is an approx.
        
    #     #saving means from observations for the weekday
    #     xoutbar = np.asmatrix(outpred[outpred.hour==j][list(Xout)].mean())
        
    #     if np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] < 0.000000001 :
    #         se = np.sqrt(varhat/n)
    #     elif np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] > 0 :
    #         se = np.array(np.sqrt((varhat/n) + varhat*(xoutbar @ inv @ xoutbar.transpose())))[0].tolist()[0]
        
    #     gh.loc[j,'interval'] = se*t
    #     gh.loc[j,'up'] = gh.Predicted[j] + gh.interval[j]
    #     gh.loc[j,'down'] = gh.Predicted[j] - gh.interval[j]
    #     j +=1
    
    # #saving
    # gh.to_csv(dirname + '\\Data\\Analysis\\' + meter + '_hour.csv',index=False)


    # #####
    

    # #avg by month
    # gm = pd.DataFrame([outpred.groupby(['Month']).Actual.mean(),
    #                    outpred.groupby(['Month']).Predicted.mean()]).transpose()

    # #initializing columns
    # gm['up'] = 1
    # gm['down'] = 1
    # gm['interval'] = 1
    
    # gmlen = len(gm)
    
    # j=1
    # while j < gmlen+1:
        
    #     #saving sample size and t stat
    #     n = len(outpred[outpred.Month == (j)])
    #     t = stats.t.ppf(0.975,N-k)
        
    #     #saving means from observations for the weekday
    #     xoutbar = np.asmatrix(outpred[outpred.Month==j][list(Xout)].mean())
    #     if np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] < 0.000000001 :
    #         se = np.sqrt(varhat/n)
    #     elif np.array(xoutbar @ inv @ xoutbar.transpose())[0].tolist()[0] > 0 :
    #         se = np.array(np.sqrt((varhat/n) + varhat*(xoutbar @ inv @ xoutbar.transpose())))[0].tolist()[0]
        
    #     gm.loc[j,'interval'] = se*t
    #     gm.loc[j,'up'] = gm.Predicted[j] + gm.interval[j]
    #     gm.loc[j,'down'] = gm.Predicted[j] - gm.interval[j]
    #     j +=1

    
    # #saving
    # gm.to_csv(dirname + '\\Data\\Analysis\\' + meter + '_mon.csv',index=False)







