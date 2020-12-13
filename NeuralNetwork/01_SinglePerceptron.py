import numpy as np
import random
import matplotlib.pyplot as plt

#import sys

#import skTools as sk
#try:
#  temp=skNN.MLP(p,indexSampleP,target,indexSampleTarget,nNodeInEachHid,transfNetwork)
#except:

# import NeuralNet as skNN
import NeuralNet as skNN

def randGauss(nRand,mu,sigma):
    randArr=np.zeros((2,nRand)) # dim[0] = no. var, dim[1]=no. samp.
    for i in range(nRand):
        randArr[0,i]=random.gauss(mu=mu,sigma=sigma)
        randArr[1,i]=random.gauss(mu=mu,sigma=sigma)
    return randArr

## input of network
# p=[[0,0,1,1],
#    [0,1,0,1]]
# p=np.mat(p)
# indexSampleP=1 # 0=Sample in row, 1=Sample in column

# # target of network
# target=np.matrix('0; 0; 0; 1')
# indexSampleTarget=0 # Sample in row

# nRandom=50
# xc1=np.zeros((1,nRandom))
# yc1=np.zeros((1,nRandom))
# for i in range(nRandom):
#     xc1[0,i]=random.gauss(mu=5,sigma=2)
#     yc1[0,i]=random.gauss(mu=5,sigma=2)
#
# xc2=np.zeros((1,nRandom))
# yc2=np.zeros((1,nRandom))
# for i in range(nRandom):
#     xc2[0,i]=random.gauss(mu=9,sigma=2)
#     yc2[0,i]=random.gauss(mu=9,sigma=2)

nSamp=50
p=randGauss(nSamp,3,2)
target=np.zeros((nSamp,1))

nSamp=50
p=np.concatenate((p,randGauss(nSamp,11,2)),axis=1)
target=np.concatenate((target,np.ones((nSamp,1))),axis=0)

p=np.mat(p)
indexSampleP=1
target=np.mat(target)
indexSampleTarget=0

# network structure
nHidLayer=1
nNodeInEachHid=np.array([1])
if len(nNodeInEachHid) != nHidLayer:
    print('You must set the no. of neural in each hidden layer !')

# array of transfer function that related with no. of hidden layer
# dim=len(nNeuronArr)-2 (set transfer funciton only neuron in each hidden layer)
transfNetwork=['hardlim']

# print('\ninput=')
# print(p)
#
# print('\ntarget=')
# print(target)
# singleNN=nn()
# singleNN.inAndOut(p,1,target,0,[2,1,1],['hardlim'])

# Build the network
nnSP1=skNN.MLP()
nnSP1.build(p,indexSampleP,target,indexSampleTarget,nNodeInEachHid,transfNetwork)

nnSP1.setWeightAndBias(1,30)

percError=10
# nnSP1.train.unified(loopInInput=10,tol=tol,visualize='text',visualizeStep='none')
# nnSP1.train.unified(loopInInput=10,tol=tol,visualize='graph',visualizeStep='click')
nnSP1.train.unified(loopInInput=10,tol=percError,visualize='graph',visualizeStep=0.05)

# --------------------- Try another input -------------------------
# # input of network
# p2=[[0.2,-0.1,1.2,1.05],
#    [0.1,1.11,0.3,1.4]]
# p2=np.mat(p2)
# indexSampleP=1 # 0=Sample in row, 1=Sample in column
#
# # target of network
# target2=np.matrix('0; 0; 0; 1')
# indexSampleTarget=0 # Sample in row
# nnSP1.setInputAndTarget(p2,indexSampleP,target2,indexSampleTarget)
#
# nnSP1.train.unified(2,tol,'graph')

# nRandom=50
# xc1=np.zeros((1,nRandom))
# yc1=np.zeros((1,nRandom))
# for i in range(nRandom):
#     xc1[0,i]=random.gauss(mu=5,sigma=2)
#     yc1[0,i]=random.gauss(mu=5,sigma=2)
#
# xc2=np.zeros((1,nRandom))
# yc2=np.zeros((1,nRandom))
# for i in range(nRandom):
#     xc2[0,i]=random.gauss(mu=9,sigma=2)
#     yc2[0,i]=random.gauss(mu=9,sigma=2)

# plt.figure(1)
# plt.plot(xc1,yc1,'ro')
# plt.plot(xc2,yc2,'bo')
# plt.xlim([0,15])
# plt.ylim([0,15])
# plt.show()

