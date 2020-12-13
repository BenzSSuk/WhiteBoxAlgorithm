import numpy as np
import math
import  matplotlib.pyplot as plt

import NeuralNet as skNN

'''
Try to use neural for curve fitting
Here we will set input as sine wave

'''
## input of network
p=np.linspace(-1,1,30)
ampl=5
f=0.4

p=np.mat(p)
indexSampleP=1 # 0=Sample in row, 1=Sample in column

# target of network
# target=np.matrix('0; 0; 0; 1')
amplNoise=1.5
noiseSig=amplNoise*(np.random.rand(1,p.shape[1]))

t=p
target=ampl*np.sin(2*math.pi*f*t)
target=target+noiseSig
target=np.mat(target)

indexSampleTarget=1 # 0=Sample in row, 1=Sample in column

# plot target
# plt.figure(1)
# plt.plot(p,target,'or')
# plt.show()

# network structure
# nHidLayer=1
nNodeInEachHid=np.array([[15,1]])
nHidLayer=len(nNodeInEachHid)
# if len(nNodeInEachHid) != nHidLayer:
# print('You must set the no. of neural in each hidden layer !')

# array of transfer function that related with no. of hidden layer
# dim=len(nNeuronArr)-2 (set transfer function only neuron in each hidden layer)
transfNetwork=['sigm','purelin']

# Build the network
nnfit=skNN.MLP()
nnfit.build(p,indexSampleP,target,indexSampleTarget,nNodeInEachHid,transfNetwork)

# training
# visualize='text','graph'
# epoch is no. of repeat the entire input set to train
# nIte is np. of repeat batch input
nnfit.train.BP(lr=0.0015,batchSize=3,epoch=1000,nIte=5,errorType='SE',TolBatch=0.1,TolAll=0.1,visualize='graph',visualizeStep=0.01)
# nnfit.train.BP(lr=0.00052,batchSize=5,epoch=5,nIte=20,errorType='SE',Tol=0.01,visualize='text',visualizeStep=0.001)
# nnfit.train.BP(lr=0.0015,batchSize=5,epoch=1000,nIte=1,errorType='SE',TolBatch=0.1,TolAll=0.1,visualize='none',visualizeStep=0.01)
#

