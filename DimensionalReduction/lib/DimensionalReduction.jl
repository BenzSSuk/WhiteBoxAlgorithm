# module DimensionalReduction
module DimReduce

using Statistics
using LinearAlgebra

export PCA

function PCA(featureMat,newDims)
# featureMat >> feature in column, sample in row

# 0. Fliter nan and zero

# 1. Centralize, subtract by mean  
#                      subtract each col with mean each column
featMatCentralized = featureMat .- mean(featureMat,dims=1);

# 2. Find Covariamce matric 
featMatCentralized_cov = cov(featMatCentralized,dims=1);

# 3. Eigen vector and eigen values
covMat_eigVec = eigvecs(featMatCentralized_cov) 
covMat_eigVal = eigvals(featMatCentralized_cov) # sorted low >> high

# 4. Sum eigen
sumEigval = sum(covMat_eigVal);
neigs=length(covMat_eigVal)
eigvalRatio=zeros(neigs)
for i = 1:neigs
    # global eigvalRatio
    eigvalRatio[i]=sum(covMat_eigVal[1:i])/sumEigval;
end

featureMatNew=featureMat*covMat_eigVec[:,1:newDims];

return featureMatNew, eigvalRatio, covMat_eigVal, covMat_eigVec

end


end