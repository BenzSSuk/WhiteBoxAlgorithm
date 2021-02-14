
import numpy as np
import math
import matplotlib.pyplot as plt

class nn(object):

    #def inAndOut(self,p,indexSampleIn,target,indexSampleTar,nNodeArr,transfArr):
    def __init__(self,p,indexSampleIn,target,indexSampleTar,nNodeArr,transfArr):
        '''
    nNodeArr=[nInput nNodeInHiddenLayer1 nNodeInHiddenLayer2 ... nNodeOutput]
    transferArr=[ActivationFuncInHiddenLayer1 ActivationFuncInHiddenLayer2 ... ActivationFuncInOutputLayer]

    *manual set nNode of input and outout is more convenient than get the input and label data to this
        '''
        # initial weight and bias
        self.nNodeArr=nNodeArr
        self.nLayer=len(self.nNodeArr)-2
        self.nNodeInput=nNodeArr[0]
        self.nNodeOutput=nNodeArr[-1]

        self.transfArr=transfArr

        self.w={}
        self.b={}
        self.a={}
        self.n={}
        for ilayer in range(self.nLayer):
            # nNodeArr start with num input,num node hid1,num node hid2,...
            self.w[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],self.nNodeArr[ilayer]),dtype=float))*0.1
            self.b[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],1),dtype=float))*0.1

            self.n[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer+1],1),dtype=float))
            self.a[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer+1],1),dtype=float))

        # check dimension of input and target
        dimInput=p.shape
        nSampleInput=dimInput[indexSampleIn]
        indexFeature=1-indexSampleIn
        nFeatureInput=dimInput[indexFeature]

        dimTarget=target.shape
        nSampleTarget=dimTarget[indexSampleTar]
        indexFeatureTarget=1-indexSampleTar
        nFeatureTarget=dimTarget[indexFeatureTarget]

        if nSampleInput==nSampleTarget and (nFeatureInput==self.nNodeInput and nFeatureTarget==self.nNodeOutput):
            self.p=p

            # make sure Sample is column
            if indexSampleTarget==0:
                target=target.transpose()
            self.target=target

        else:
            print('dimension of input and target mismatch !')
            print(f'SampleInput={nSampleInput}, SampleTarget={nSampleTarget} \n')
            print(f'nNodeInput={self.nNodeInput}, nFeatureInput={nFeatureInput} \n')
            print(f'nFeatureTarget={nFeatureTarget}, nNodeOutput={self.nNodeOutput}\n')
            # case 1 nSample of input and target mismatch
            # case 2 nFeature != nNodeInput
            # case 3 type of target != nNodeOutput

        self.nInput=nSampleInput

        ##self.train=nn.train(self.p,self.target,self.w,self.b,self.nInput,self.nLayer)
        #print(f'Print in set nn, train.w={self.train.w}')
#    def initialaWeight:

    def inputAndLabel(self,p,indexSampleIn,target,indexSampleTar):
        # check dimension of input and target
        dimInput=p.shape
        nSampleInput=dimInput[indexSampleIn]
        indexFeature=1-indexSampleIn
        nFeatureInput=dimInput[indexFeature]

        dimTarget=target.shape
        nSampleTarget=dimTarget[indexSampleTar]
        indexFeatureTarget=1-indexSampleTar
        nFeatureTarget=dimTarget[indexFeatureTarget]

        if nSampleInput==nSampleTarget and (nFeatureInput==self.nNodeInput and nFeatureTarget==self.nNodeOutput):
            self.p=p
            self.target=target
        else:
            print('dimension of input and target mismatch !')
            print(f'SampleInput={nSampleInput}, SampleTarget={nSampleTarget} \n')
            print(f'nNodeInput={self.nNodeInput}, nFeatureInput={nFeatureInput} \n')
            print(f'nFeatureTarget={nFeatureTarget}, nNodeOutput={self.nNodeOutput}\n')
            # case 1 nSample of input and target mismatch
            # case 2 nFeature != nNodeInput
            # case 3 type of target != nNodeOutput

        self.nInput=nSampleInput

    def f(self,net,transf):
        if transf=='purelin':
            aout=net
        elif transf=='hardlim':
            aout=np.mat(np.zeros((net.shape),dtype=float))
            therdhold=0
            aout[net>=therdhold]=1
            #aout[net<therdhold]=0
    #        net[net>=therdhold]=1
    #        net[net<therdhold]=0


    
        elif transf=='sigm':
            x=np.array(net) # dim=n*1
            aout=np.zeros((net.shape),dtype=float)
            for ix in range(x.shape[0]):
                aout[ix,0]=1/(1+math.exp(-x[ix,0]))
        elif transf=='tanh':
            x=np.array(net) # dim=n*1
            aout=np.zeros((net.shape),dtype=float)
            for ix in range(x.shape[0]):
                aout[ix,0]=(2/(1+math.exp(-2*x[ix,0])))-1
        return aout

    def F(self,net,transf):
        if transf=='purelin':
            aout=np.ones((net.shape),dtype=float)
        elif transf=='hardlim':
            aout=np.zeros((net.shape),dtype=float)
        elif transf=='sigm':
            x=net
            aout=nn.f(x,'sigm')*(1-nn.f(x,'sigm'))
        elif transf=='tanh':
            x=np.array(net) # dim=n*1
            x=x.transpose() # dim=1*n
            aout=1/(x*x+1)
            aout=np.mat(aout)
            aout=aout.transpose()
        return aout

    def feedforward(self,p):
        #-------------------- feed forward ---------------------
        #self.w=self.train.w
        #self.b=self.train.b

        for ilayer in range(self.nLayer):
            print(f'Feedforward w={self.w}, b={self.b}')
            if ilayer>0:
#                print(f'w={self.w[ilayer], b={self.b[ilayer]}')
                self.n[ilayer]=self.w[ilayer]*self.a[ilayer-1]+self.b[ilayer]
            elif ilayer==0:
                # warning if we assing W as matric with dim=nNode*nInput, n must be n=Wp+b (no transpose)
                # transpose W require only when w is column vector(normally in the update part)
                # self.n[iL]=np.transpose(self.w[iL])*p+self.b[iL]
#                print(f'w={self.w[ilayer], b={self.b[ilayer]}')
                self.n[ilayer]=self.w[ilayer]*p+self.b[ilayer]

            # self.a[iL]=nn.f(n[iL],'purelin'); # adaline use pure linear
#            print(f'a[{ilayer}]={self.a[ilayer]}')
            self.a[ilayer]=nn.f(self,self.n[ilayer],self.transfArr[ilayer])

        #print(f'Feed forward p={p} \n a={self.a}');

        # return only the output of network(output of the last layer)
        return self.a[self.nLayer-1]

    def getWeightAndBias(self):
        return self.w,self.b

    def showVal(self,ilayer):
        ilayer=ilayer-1
        print(f'the current weight layer {ilayer} = {self.w[ilayer]}')
        print(f'the current bias layer {ilayer} = {self.b[ilayer]}')       

    #class train:
    def train(self,lrRule):
    # According to each learning rule require different input argv,
    # use train as class and lrRule as def if convenient for independent input argv
    #print(f'Update the network using {learningRule}')
    #    def __init__(self,p,target,w,b,nInput,nLayer):
    #        self.w=w
    #        self.b=b
    #        self.p=p
    #        self.target=target
    #        self.nInput=nInput
    #        self.nLayer=nLayer
    #        self.nn=nn()
            #self.nn.showVal(1)

        #def Unified(self):
        if lrRule=='Unified':
#            plot=lrParam.plot
            
            icount=-1
            while True:
            #for ip in range(self.nInput):
                icount=icount+1
                # find the error all

                sumError=0
                for ip in range(self.nInput):
                    # get output of the network
                    # update
                    # out is a at last layer
                    output=nn.feedforward(self,self.p[:,ip])

                    # error
                    # target is already set sample into column
                    e=(self.target[:,ip]-output)

                    print(f'target[ip]={self.target[:,ip]}, output={output}, e={e}')
                    print(f'wold={self.w[0]}, p={self.p[:,ip]}')

                    if e != 0:
                        # update w
                        # Wnew = Wold + (t-a)p
                        wBuff=self.w[0].transpose()
                        #self.w[0].transpose()=self.w[0]+e[0,0]*self.p[:,ip];
                        wBuff=wBuff+e[0,0]*self.p[:,ip]
                        self.w[0]=wBuff.transpose()
                        
                        # biasNew = biasOld + e
                        self.b[0]=self.b[0]+e
                        
                    sumError=sumError+e

                    print(f'wnew={self.w[0]}')
                    print(f'sumError={sumError}\n')
                #print(f'')
                print(f'icount={icount}')
                if sumError==0 or icount==3:
                    break

                #if ip==(self.nInput-1):
                #    ip=-1

                #for ilayer in range(self.nLayer):
                    # nNodeArr start with num input,num node hid1,num node hid2,...
                    #self.w[ilayer]=self.w[ilayer]+e[0,0]*self.p[:,ip];
            
        #def BP():
        elif lrRule=='BP':
        # BackPropagation
#            self.w[0]=self.w[0]+100
#            print(f'weight layer {0} = {self.w[0]}')
            #lr, batchSize, funcError, 
            
            sen={}
            sumSen={}
            diagF={}
            for ilayer in range(self.nLayer):
                # nNodeArr start with num input,num node hid1,num node hid2,...
                # number of s is same as n node
                sen[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer],1),dtype=float))
                sumSen[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer],1),dtype=float))
                diagF[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer],self.nNodeArr[ilayer]),dtype=float))

            # find error
            icountBatch=0
            sumSen=0
            for ip in range(self.nInput):
                icountBatch=icountBatch+1
                pIn=self,p[:,ip]
                
                output=nn.feedforward(pIn)
                
                error=self.target[:,ip]-output
                errorArr[icountBatch]=error

                # loop from the last to the first layer
                #nLayerBack=np.arange(self.nLayer-1,0,-1)
                #for ilayer in nLayerBack:
                for ilayer in range(self.nLayer-1,0,-1):
                    
                    #diagF[ilayer]=np.diag(np.transpose(nn.F(self,self.n[ilayer])))
                    # diagflat can set input as row or vector
                    diagF[ilayer]=np.diagflat(nn.F(self,self.n[ilayer],self.transfArr[ilayer]))
                    if (ilayer < self.nLayer-1) and (ilayer >= 0):
                    # not the last layer
                        #sen[ilayer]=nn.F(self,self.n[ilayer],self.transfArr[ilayer])*self.w[ilayer+1]*sen[ilayer+1]
                        sen[ilayer]=diagF*self.w[ilayer+1]*sen[ilayer+1]
                        
                    elif ilayer == (self.nLayer-1):
                    # the Last layer 
                        #sen[ilayer]=-2*nn.F(self,self.n[ilayer],self.transfArr[ilayer])*error
                        sen[ilayer]=-2*diagF[ilayer]*error
                        
                    sumSen[ilayer]=sumSen[ilayer]+sen[ilayer]

                    if icountBatch==batchSize:
                    # update the weight and bias 
                        if ilayer==0:
                            aProvLayer=pIn
                        elif ilayer>0 and ilayer<=self.nLayer-1:
                            aProvLayer=self.a[ilayer-1]
                        self.w[ilayer]=self.w[ilayer]-lr*sumSen[ilayer]*self.a[ilayet]*aProvLayer
                        self.b[ilayer]=self.b[ilayer]-lr*sumSen[ilayer]
                        
                        # reset countBatch
                        icountBatch=0

                    

                # combine error of batch size 
                if errorType=='se':
                    sumError=sum(np.square)
                    
                # update weight and bias
                # curlE/a
                #wnew = wold + lr*s*am-1

            # find sen
            
            # 

        #def VLPBP():
        elif lrRule=='VLRBP':
        # Variable learning rate Back propagation
            print('Update the weight using Variable learning rate Back Prop')
            self.w[0]=self.w[0]+200
            print(f'weight layer {0} = {self.w[0]}')

        else:
            print('Learning rule was not recognite !')

    def test(self,testMat,indexSampleTestMat,targetTest,indexSampleTestTarget):
        if indexSampleTestMat==0:
            testMat=testMat.transpose()
        nSampleTest=testMat.shape[1]

        if indexSampleTestTarget==0:
            targetTest=targetTest.transpose()
        nSampleTargetTest=targetTest.shape[1]

        # check compatatble
        if nSampleTest==nSampleTargetTest:
            outputMat=np.zeros([self.nNodeOutput,nSampleTest])
            errorMat=np.zeros([nSampleTest,1])
            for ip in range(nSampleTest):
                # get output of the network
                # out is a at last layer
                output=nn.feedforward(self,testMat[:,ip])


                # error
                # target is already set sample into column
                error=(self.target[:,ip]-output)

                errorMat[ip,0]=error
                outputMat[:,ip]=output
        else:
            print('Error, size of testMat and targetTest not match !')

        return outputMat,errorMat
#[no. of input node, no. of hiddedn node1, no. of hiddedn node2,...,no. of output node ]
#nn1=nn(np.array([2,3,3]),['purelin',])

#print('value of weight is \n')
#nn1.showVal(0)
#nn1.update('BackProp')
#nn1.showVal(0)
#nn1.update('VLRBackProp')
#nn1.showVal(0)

p=[[0,0,1,1],
   [0,1,0,1]]
p=np.mat(p)
indexSampleP=1 # Sample in column
nDimP=p.shape[1-indexSampleP]

target=np.matrix('0; 0; 0; 1')
indexSampleTarget=0 # Sample in row
nDimTarget=target.shape[1-indexSampleTarget]

#nNeuronArr=[nInput, nNode hidden layer 1, nNode hidden layer 2, .... nOutput]
neuronArr=[nDimP,1,nDimTarget]

# array of transfer function that related with no. of hidden layer
# dim=len(nNeuronArr)-2 (set transfer funciton only neuron in each hidden layer)
transfArr=['hardlim']

print('\ninput=')
print(p)

print('\ntarget=')
print(target)
#singleNN=nn()
#singleNN.inAndOut(p,1,target,0,[2,1,1],['hardlim'])
singleNN=nn(p,indexSampleP,target,indexSampleTarget,neuronArr,transfArr)

#singleNN.inputAndLabel(p,1,target,0)
# show output of nn with single input
singleNN.feedforward(p[:,0])

# train network
#singleNN.train.Unified()
#lrParam.plot=0
singleNN.train('Unified')
[w,b]=singleNN.getWeightAndBias()

#singleNN.feedforward(p[:,0])

[outputTest,errorTest]=singleNN.test(p,indexSampleP,target,indexSampleTarget)
