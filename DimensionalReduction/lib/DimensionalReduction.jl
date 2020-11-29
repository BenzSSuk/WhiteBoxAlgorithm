module DimensionalReduction

using statistics
using LinearAlgebra

function PCA(featureMat,dimNew)
# feature in column, sample in row

# 1. Centralize, subtract by mean  
#                      subtract each col with mean each column
featMatCentralized = featureMat .- mean(featureMat,dims=1);

# 2. Find Covariamce matric 
featMatCentralized_cov = cov(featMatCentralized,dims=1);

# 3. Eigen vector and eigen values
covMat_eigVec = eigvecs(featMatCentralized_cov)
covMat_eigVal = eigvals(featMatCentralized_cov)

# 4. Sum eigen
sumEigval = sum(covMat_eigVal);


end

end