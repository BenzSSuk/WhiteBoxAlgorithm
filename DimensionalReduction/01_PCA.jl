## using and include
# Standard
using CSV
using DataFrames
using Statistics
using Plots
# custom
# find current dirctory
# if open folder in vscode, pwd() = open folder
currentDir=pwd()
projDir=currentDir;

print(["Current Dir: " projDir])

# this included file may be contain with multiple module 
include(joinpath(projDir,raw"DimensionalReduction\lib\DimensionalReduction.jl"))
import .DimReduce # import module in file that already included

# include(joinpath(projDir,raw"Tools\FileAndTable.jl"))
include(joinpath(projDir,raw"Tools\zTools.jl"))
import .zTools

## Load feature
pathFeat=raw"C:\Users\Panna\OneDrive\rawdata\20200502";
listFile=zTools.GetList(pathFeat)
featureTable,listFeat=zTools.LoadTable(pathFeat,".csv",0);

## select feature
headerFeature=names(featureTable[4]);
listFeatureSelect=["RT_90s_SD", "RT_90s_mean",
       "RT_90s_SDPP", "RT_90s_meanPP", "RT_90s_HRV_sm_RMSSD",
       "RT_90s_HRV_smnm_SD", "RT_90s_HRV_sm_nm_SD", "RT_90s_meanPI",
       "RT_90s_sdPI", "RT_90s_sdPIperc", "HR_90s_value", "SDHR_90s_stdm",
       "SDHR_90s_std", "SDHR_90s_perc", "SDHR_90s_percM", "SDHR_90s_slideMax",
       "SDHR_90s_slideMaxPerc", "featHRV_90s_LF", "featHRV_90s_LFnu",
       "featHRV_90s_LFnu2", "featHRV_90s_HF", "featHRV_90s_HFnu",
       "featHRV_90s_HFnu2", "featHRV_90s_conv_HF", "featHRV_90s_LFHF"];

# select only column in DF that is feature value and convert to matric 
featureMat=Matrix(featureTable[4][:,listFeatureSelect]);
# featureMat=convert(Matrix,featureTable[4][:,listFeatureSelect]);
featureMat,indexFilOut=zTools.FilterDF(featureMat,["nan"]);

## PCA
newdims=3;
featureMatNew,eigRatio,covMat_eigVal, covMat_eigVec=DimReduce.PCA(featureMat,newdims);

gr()
# plot(featureMatNew[:,1],featureMatNew[:,2],featureMatNew[:,3])
scatter(featureMatNew[:,1],featureMatNew[:,2],featureMatNew[:,3])




