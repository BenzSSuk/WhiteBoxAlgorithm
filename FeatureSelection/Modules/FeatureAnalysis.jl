module zfs
# standard lib
using DataFrames
using CUDA
# custom
include("C:\\Users\\Panna\\OneDrive\\Code\\4_Julia\\Module\\Tools.jl")
import .zt

function Entropy(nInstanceArr)
        # HIte = prob * math.log2(prob)
    # nClass = nInstanceArr.shape[0] # number of row
    nClass = length(nInstanceArr) # number of row

    # nInstanceAll = np.sum(nInstanceArr)
    nInstanceAll = sum(nInstanceArr)
    # global nClass,nInstanceAll

    if nInstanceAll > 0 && nClass > 1
        Hsum = 0
        # for iclass in range(nClass):
        for iclass=1:nClass
            # global nInstanceAll,nClass
            prob = nInstanceArr[iclass] / nInstanceAll
            if prob > 0
                HIte = prob * log2(prob)
            elseif prob == 0
                # if prob = 0 >> use lim P-->0 0log(0) = 0
                HIte = 0
            end
            Hsum = Hsum + HIte
        end
        H = -Hsum

    elseif nInstanceAll <= 0
        # there are invalid numer of class or no instance in each class.
        # sys.exit('Input must be positive integer and no. of sample must greater than 1')
        error("Input must be positive integer and no. of sample must greater than 1")

    elseif nClass == 1
        # if num class is 1 >> prob=1 >> entropy will always H=0
        H = 0

    else
        print("Error in Entropy: Invalid input")

    end

    return H
end # entropy


# def FindNumClassArr(LabelMat, LabelRange):
function FindNumClassArr(LabelMat, LabelRange)
    # input:
    #   LabelMat = array of label
    #   LabelRange
    # output:
    nClass = length(LabelRange)
    # nClassArr=np.zeros((1,nClass))
    nClassArr=zeros(1,nClass)

    # for inc in range(nClass):
    countClass=0
    for iclass=LabelRange
        countClass = countClass + 1
        indexFoundClass = (LabelMat .== iclass)
        nClassArr[countClass] = sum(indexFoundClass)
    end
    return nClassArr

end # Findnum

function testgpu!(IG,axisVec)
    for i=1:length(IG)
        IG[i]=IG[i]+axisVec[i]
    end
end

# def InfoGain(featureDF, label, LabelRange):
function InfoGain(featureDF, label, LabelRange)
    # Input:
    #       featureDF = dataframe of feature table
    #       label = columns array (ndarray)
    #       LabelRange = [1, 2, 3,....] group of class

    # Try to divided data with threshold in each feature and calculated information gain in each threshold step
    # InfoGain use to find position that clearly seperated data into each class(position that yields maximum InfoGain)

    # custom lib
    # include("C:\\Users\\Panna\\OneDrive\\Code\\4_Julia\\Module\\Tools.jl")
    # import .zt

    # remove nan, zero, missing
    featureDF,idxRowRemove=zt.FilterDF(featureDF,["nan","zero","missing"])
    label=label[.!idxRowRemove]

    axisVecType="CenterGap"
    # nFeat = featureDF.shape[1]  # o=row, 1=column
    # nInstanceAll = featureDF.shape[0]
    nInstanceAll,nFeat=size(featureDF)

    percIncrement = 0.1

    # warning label matricx not is numClassArr
    # -- 1. Calculate Entropy of matrix in
    # LabelRange = [1, 2]
    nClassArr = FindNumClassArr(label, LabelRange)

    HIn = Entropy(nClassArr)

    # IGmax = np.zeros((1, nFeat))
    IGmax=zeros(nFeat)
    # IGmaxIndex = np.zeros((1, nFeat))
    IGmaxIndex=zeros(nFeat)

    featureName=Dict()

    # GPU function, please remind function must not has output
    # function testgpu!(IG,axisVec)
    #     for i=1:length(IG)
    #         IG[i]=IG[i]+axisVec[i]
    #     end
    # end
    function gpu_IG!(IG,axisVec,featureIn)
        for i=1:length(axisVec)
            ix=axisVec[i]
            indexWinL = (featureIn .< ix)
            nInstanceL=sum(indexWinL)
            LabelL = label[indexWinL]
            nClassArrL = FindNumClassArr(LabelL, LabelRange)
            HwinL = Entropy(nClassArrL)

            indexWinR = (featureIn .> ix)
            nInstanceR = sum(indexWinR)
            LabelR = label[indexWinR]
            nClassArrR = FindNumClassArr(LabelR, LabelRange)
            HwinR = Entropy(nClassArrR)

            Hratio = (1 / nInstanceAll) * (nInstanceL * HwinL + nInstanceR * HwinR)
            # ig = ig + 1
            IG[i] = HIn - Hratio


        end # loop ix

        return nothing
    end


    # - consider feature one by one
    # for ifeat in range(nFeat):
    for ifeat=1:nFeat
    # Threads.@threads for ifeat=1:nFeat
        print("ifeat:$ifeat/$nFeat\n")
        # featureIn = featureDF.iloc[:, ifeat]
        # featureIn = np.array([featureIn])
        featureIn = featureDF[:,ifeat]
        # featureName[ifeat] = names(featureDF,ifeat)

        # -- 2. create x axis array and the step to find the Information gain
        # elif axisVecType == 'CenterGap':
        if axisVecType == "CenterGap"
            # step in the gap between data point
            # featureInSorted = np.sort(featureIn, axis=0) # axis=0 low to high, axis=1 high to low
            featureInSorted=sort(featureIn)
            featureInSorted=unique(featureInSorted)

            # n=featureInSorted.shape[1]
            # featureInSorted = 0.2  0.33  0.5  0.8  1.5
            # axisVec         =     1     1    1    1
            # axisVec = (featureInSorted[0,1:n - 1] + featureInSorted[0,2:n]) / 2
            axisVec = (featureInSorted[1:end-1] + featureInSorted[2:end]) / 2

        end # if axisType

        # -- 3.[CPU] find infomation gain in each x position
        ig = 0
        # IG = np.zeros((axisVec.shape))
        IG = zeros(size(axisVec))
        for ix in axisVec
            indexWinL = (featureIn .< ix)
            nInstanceL=sum(indexWinL)
            LabelL = label[indexWinL]
            nClassArrL = FindNumClassArr(LabelL, LabelRange)
            HwinL = Entropy(nClassArrL)

            indexWinR = (featureIn .> ix)
            nInstanceR = sum(indexWinR)
            LabelR = label[indexWinR]
            nClassArrR = FindNumClassArr(LabelR, LabelRange)
            HwinR = Entropy(nClassArrR)

            Hratio = (1 / nInstanceAll) * (nInstanceL * HwinL + nInstanceR * HwinR)
            ig = ig + 1
            IG[ig] = HIn - Hratio
        end # loop ix

        # -- 3.[CPU-Multi-threads] find infomation gain in each x position
        # ig = 0
        # # IG = np.zeros((axisVec.shape))
        # IG = zeros(size(axisVec))
        # Threads.@threads for it=1:length(axisVec)
        #     ix=axisVec[it]
        #     indexWinL = (featureIn .< ix)
        #     nInstanceL=sum(indexWinL)
        #     LabelL = label[indexWinL]
        #     nClassArrL = FindNumClassArr(LabelL, LabelRange)
        #     HwinL = Entropy(nClassArrL)
        #
        #     indexWinR = (featureIn .> ix)
        #     nInstanceR = sum(indexWinR)
        #     LabelR = label[indexWinR]
        #     nClassArrR = FindNumClassArr(LabelR, LabelRange)
        #     HwinR = Entropy(nClassArrR)
        #
        #     Hratio = (1 / nInstanceAll) * (nInstanceL * HwinL + nInstanceR * HwinR)
        #     # ig = ig + 1
        #     IG[it] = HIn - Hratio
        # end # loop ix



        # -- 3.[GPU] find infomation gain in each x position
        # IG=CUDA.zeros(length(axisVec))
        # axisVec=CuArray{Float32}(axisVec)
        # featureIn=CuArray{Float32}(featureIn)
        # # @cuda thread=256 gpu_IG!(IG,axisVec,featureIn)
        # @cuda thread=256 testgpu!(IG,axisVec)


        # -- 4. find the position that yields the maximum information gain
        # IGmax[ifeat] = max(IG)
        # IGmaxIndex[ifeat] = IG.index(IGmax[ifeat])
        IGmax[ifeat],posMax=findmax(IG)
        IGmaxIndex[ifeat]=axisVec[posMax]
    end # loop feature

    # listFeature=names(featureDF)
    InfoGainTable=DataFrame(indexCol=1:nFeat,FeattureName=names(featureDF),IGmax=IGmax,IGmaxPos=IGmaxIndex)
    # lenIG=length(IGmax)
    # lenIndex=length(IGmaxIndex)
    # print("IGmax:$lenIG\n")
    # print("Index:$lenIndex")
    # InfoGainTable=DataFrame(IGmax=IGmax,IGmaxPos=IGmaxIndex)
    # InfoGainTable=crossjoin(featureName,InfoGainTable)

    # sort
    InfoGainTable=sort(InfoGainTable,3,rev=true)

    return InfoGainTable
end # info Gain

end # module
