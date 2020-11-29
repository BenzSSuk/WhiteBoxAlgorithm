## using and include
# Standard
using CSV
using DataFrames



# custom
# find current dirctory
# if open folder in vscode, pwd() = open folder
currentDir=pwd()
projDir=currentDir;
# this included file may be contain with multiple module 
include(joinpath(projDir,raw"DimensionalReduction\lib\DimensionalReduction.jl"))
import .DimensionalReduction # import module in file that already included

include(joinpath(projDir,raw"Tools\FileAndTable.jl"))
import .FileAndTable


## Load feature
pathFeat=raw"C:\Users\Panna\OneDrive\rawdata\20200502";
featureTable,listFeat=FileAndTable.LoadTable(pathFeat,".csv",0);

## select feature
headerFeature=names(featureTable[2]);
listFeatureSelect=["RT_90s_SD", "RT_90s_mean",
       "RT_90s_SDPP", "RT_90s_meanPP", "RT_90s_HRV_sm_RMSSD",
       "RT_90s_HRV_smnm_SD", "RT_90s_HRV_sm_nm_SD", "RT_90s_meanPI",
       "RT_90s_sdPI", "RT_90s_sdPIperc", "HR_90s_value", "SDHR_90s_stdm",
       "SDHR_90s_std", "SDHR_90s_perc", "SDHR_90s_percM", "SDHR_90s_slideMax",
       "SDHR_90s_slideMaxPerc", "featHRV_90s_LF", "featHRV_90s_LFnu",
       "featHRV_90s_LFnu2", "featHRV_90s_HF", "featHRV_90s_HFnu",
       "featHRV_90s_HFnu2", "featHRV_90s_conv_HF", "featHRV_90s_LFHF"];

## PCA
using Statistics
a=[1 1 1;
   2 4 6];
aMean=mean(a,dims=1)
print(aMean)
a.-aMean

## cov
aCov=cov(a)

## 
using LinearAlgebra
eigvals(aCov)
eigvecs(aCov)



