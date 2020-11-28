## import module
include("C:\\Users\\Panna\\OneDrive\\Code\\4_Julia\\Module\\Tools.jl")
import .zt

include("C:\\Users\\Panna\\OneDrive\\Code\\4_Julia\\Module\\FeatureAnalysis.jl")
import .zfs
pathFeat="C:\\Users\\Panna\\OneDrive\\rawdata\\20200502"

using CUDA
## import
featureTable,listFeat=zt.LoadTable(pathFeat,".csv",0)

## select feature
headerFeature=names(featureTable[4])

listFeatureSelect=["RT_90s_SD", "RT_90s_mean",
       "RT_90s_SDPP", "RT_90s_meanPP", "RT_90s_HRV_sm_RMSSD",
       "RT_90s_HRV_smnm_SD", "RT_90s_HRV_sm_nm_SD", "RT_90s_meanPI",
       "RT_90s_sdPI", "RT_90s_sdPIperc", "HR_90s_value", "SDHR_90s_stdm",
       "SDHR_90s_std", "SDHR_90s_perc", "SDHR_90s_percM", "SDHR_90s_slideMax",
       "SDHR_90s_slideMaxPerc", "featHRV_90s_LF", "featHRV_90s_LFnu",
       "featHRV_90s_LFnu2", "featHRV_90s_HF", "featHRV_90s_HFnu",
       "featHRV_90s_HFnu2", "featHRV_90s_conv_HF", "featHRV_90s_LFHF"]

# output is dataframe
featureMat=featureTable[4][:,listFeatureSelect]

# output is array
# labelNum=featureTable[4][:,"anyAH"]
labelNum=featureTable[4][:,"anyApnea"]

## process feature table
zEntropy=zfs.Entropy([10 14])

# featureDFFil,indexRowRemove=zt.FilterDF(featureMat,["nan","zero"])
# labelNumFil=labelNum[.!indexRowRemove]
tIG=@elapsed featureMatSort=zfs.InfoGain(featureMat,labelNum,[0,1])
