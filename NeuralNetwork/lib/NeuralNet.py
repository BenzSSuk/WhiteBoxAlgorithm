import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# Architecture = Multiple layer perceptron
class MLP(object):
    def __init__(self):
        # class for training the network
        self.train=train()       
        
    def build(self,p,indexSampleIn,target,indexSampleTar,nNodeInEachHid,transfArr):
        '''Input:
            p = input matrix
            indexSampleIn = index of instances of input matrix
                          = 1
            nNodeArr=[nInput nNodeInHiddenLayer1 nNodeInHiddenLayer2 ... nNodeOutput]
            transferArr=[ActivationFuncInHiddenLayer1 ActivationFuncInHiddenLayer2 ... ActivationFuncInOutputLayer]

            manual set nNode of input and outout is more convenient than get the input and label data to this
        ''' 
        # initial weight and bias
        # nNeuronArr=[nInput, nNode in hidden layer 1, nNode in hidden layer 2, .... nOutput]
        # Create network with only 1 hidden layer, 1 node

        shapeInput=p.shape
        nDimP=shapeInput[1-indexSampleIn]
        nDimP=np.array([nDimP])
        
        nSampleInput=shapeInput[indexSampleIn]

        shapeTarget=target.shape
        nDimTarget=shapeTarget[1-indexSampleTar]
        nDimTarget=np.array([nDimTarget])
        
        nSampleTarget=shapeTarget[indexSampleTar]

        nNodeArr=np.append(nDimP,nNodeInEachHid)
        nNodeArr=np.append(nNodeArr,nDimTarget)
        
        print(f'Nerwork structure={nNodeArr}')
        
        # print(f'nInput={nSampleInput},nOutput={nSampleTarget}')
        self.nNodeArr=nNodeArr
        self.nLayer=len(self.nNodeArr)-2
        self.nNodeInput=nNodeArr[0]
        self.nNodeOutput=nNodeArr[-1]

        self.transfArr=transfArr

        # pre-allocate cell
        self.w={}
        self.b={}
        self.a={}
        self.n={}
        for ilayer in range(self.nLayer):
            # nNodeArr start with num input,num node hid1,num node hid2,...
            self.w[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],self.nNodeArr[ilayer]),dtype=float))*1
            self.b[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],1),dtype=float))*0

            self.n[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer+1],1),dtype=float))
            self.a[ilayer]=np.mat(np.zeros((self.nNodeArr[ilayer+1],1),dtype=float))

        # check dimension of input and target
        # dimInput=p.shape
        
        # indexFeature=1-indexSampleIn
        # nFeatureInput=dimInput[indexFeature]
        # dimTarget=target.shape
        # indexFeatureTarget=1-indexSampleTar
        # nFeatureTarget=dimTarget[indexFeatureTarget]
        
        # print(f"nInput={nSampleInput},nOutput={nSampleTarget}")
        if nSampleInput==nSampleTarget:
            # and (nFeatureInput==self.nNodeInput and nFeatureTarget==self.nNodeOutput)
            # print('Initial set is compatible !')
            
            if indexSampleIn==0:
                p=p.transpose()
            self.p=p
            
            # make sure Sample is column
            if indexSampleTar==0:
                target=target.transpose()
            self.target=target
            
            self.nInput=nSampleInput

            self.train.initialParameter(self.p,self.target,self.w,self.b,self.n,self.a,self.transfArr,self.nInput,self.nLayer,self.nNodeArr)
            
            print(f'Building the network: succeeded !')
            
        else:
            print(f'Building the network: fail !')
            print('Dimension of input and target mismatch !')
            print(f'SampleInput={nSampleInput}, SampleTarget={nSampleTarget} \n')
            # print(f'nNodeInput={self.nNodeInput}, nFeatureInput={nFeatureInput} \n')
            # print(f'nFeatureTarget={nFeatureTarget}, nNodeOutput={self.nNodeOutput}\n')
            # case 1 nSample of input and target mismatch
            # case 2 nFeature != nNodeInput
            # case 3 type of target != nNodeOutput

    def setInputAndTarget(self,p,indexSampleIn,target,indexSampleTar):
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
            if indexSampleIn==0:
                p=p.transpose()
            self.p=p
            
            if indexSampleTar==0:
                target=target.transpose()
            self.target=target
            
            self.nInput=nSampleInput
            self.train.setInputAndTarget(self.p,self.target)
          
        else:
            print('dimension of input and target mismatch !')
            print(f'SampleInput={nSampleInput}, SampleTarget={nSampleTarget} \n')
            print(f'nNodeInput={self.nNodeInput}, nFeatureInput={nFeatureInput} \n')
            print(f'nFeatureTarget={nFeatureTarget}, nNodeOutput={self.nNodeOutput}\n')
            # case 1 nSample of input and target mismatch
            # case 2 nFeature != nNodeInput
            # case 3 type of target != nNodeOutput
    def setWeightAndBias(self,w,b):
        # set w and b in every layer
        for ilayer in range(self.nLayer):
            # nNodeArr start with num input,num node hid1,num node hid2,...
            self.w[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],self.nNodeArr[ilayer]),dtype=float))*w
            self.b[ilayer]=np.mat(np.ones((self.nNodeArr[ilayer+1],1),dtype=float))*b
          
    def getWeightAndBias(self):
        self.w=self.train.w
        self.b=self.train.b
        return self.w,self.b

    def showVal(self,ilayer):
        self.w=self.train.w
        self.b=self.train.b
        ilayer=ilayer-1
        print(f'the current weight layer {ilayer} = {self.w[ilayer]}')
        print(f'the current bias layer {ilayer} = {self.b[ilayer]}')       

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
                output=self.feedforward(testMat[:,ip])
    
                # error
                # target is already set sample into column
                error=(self.target[:,ip]-output)
    
                errorMat[ip,0]=error
                outputMat[:,ip]=output
        else:
            print('Error, size of testMat and targetTest not match !')
    
        return outputMat,errorMat
    
class train(object):
    '''
    According to each learning rule require different input argv,
    use train as class and lrRule as def are convenient for independent input argv
    '''
    # def __init__(self):
    #     pass
        
    def initialParameter(self,p,target,w,b,n,a,transfArr,nInput,nLayer,nNodeArr):
        self.p=p
        self.target=target
        self.w=w
        self.b=b
        self.n=n
        self.a=a
        self.transfArr=transfArr
        self.nInput=nInput
        self.nLayer=nLayer
        self.nNodeArr=nNodeArr

        self.nSamp=target.shape[1]
        print('Preallocating initial parameter: succeeded !')
        #print(f'Test feedforward={feedforward(p[:,0])}')
        
    def setInputAndTarget(self,p,target):
        self.p=p
        self.target=target
        
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
            aout=self.f(net,'sigm')*(1-self.f(net,'sigm'))
        elif transf=='tanh':
            x=np.array(net) # dim=n*1
            x=x.transpose() # dim=1*n
            aout=1/(x*x+1)
            aout=np.mat(aout)
            aout=aout.transpose()
        return aout

    #def feedforward(transfArr,w,b,a,p):
    def feedforward(self,p):
        '''
        Input:
            w = weight of network, dataType = dict, w[0]=matrix weight of layer 1, ...
            b = weight of network, dataType = 
            p = input of network, dataType = 
        
        Output: 
            a = out of network (in each layer)
            
        '''
        #nLayer=len(w)
        # a={}
        for ilayer in range(self.nLayer):
            #print(f'Feedforward w={self.w}, b={self.b}')
            if ilayer>0:
    #                print(f'w={self.w[ilayer], b={self.b[ilayer]}')
                self.n[ilayer]=self.w[ilayer]*self.a[ilayer-1]+self.b[ilayer]
                #n=w[ilayer]*a[ilayer-1]+b[ilayer]
            elif ilayer==0:
                # warning if we assing W as matric with dim=nNode*nInput, n must be n=Wp+b (no transpose)
                # transpose W require only when w is column vector(normally in the update part)
                # self.n[iL]=np.transpose(self.w[iL])*p+self.b[iL]
    #                print(f'w={self.w[ilayer], b={self.b[ilayer]}')
                self.n[ilayer]=self.w[ilayer]*p+self.b[ilayer]
                #n=w[ilayer]*p+b[ilayer]
    
            # self.a[iL]=f(n[iL],'purelin'); # adaline use pure linear
    #            print(f'a[{ilayer}]={self.a[ilayer]}')
            
            self.a[ilayer]=self.f(self.n[ilayer],self.transfArr[ilayer])
            #a[ilayer]=f(n,transfArr[ilayer])
    
        #print(f'Feed forward p={p} \n a={self.a}');
    
        # return only the output of network(output of the last layer)
        #return self.a[self.nLayer-1]
        return self.a[self.nLayer-1],self.a

    # create output
    def calError(self,e,errorType):
        if errorType=='SAE':
            # sum of absolute error
            sumError=np.sum(abs(e))

        elif errorType=='SE':
            # sum of square error
            sumError=np.sum(np.power(e,2))

        elif errorType=='RMS':
            sumError=np.sum(np.power(e, 2)/len(e))

        elif errorType=='PBE':
            # in scenario, we has binary class
            sumError=(np.sum(abs(e))*100)/len(e)
        else:
            sys.exit('Error type is not recognite !')

        return sumError

    def test(self,errorType,indexInput):
        '''
        Input:
            errorType = Sum sq error, Root mean sq error, ....
            indexInput = index of desired input that want to test
                       = -1 is meant test all input

        Output:

        '''
        # print(f'Type arr={type(indexInput)}')
        # isint=type(indexInput)==int
        # print(f'Is int:{isint}')
        if type(indexInput)==int and indexInput==-1:
            # test all input
            e=np.zeros((self.target.shape))
            indexInput=np.arange(0,self.target.shape[1])
        elif type(indexInput)==int and indexInput>=0:
            # test only one input
            e=np.zeros((1,1))
        elif type(indexInput)!=int:
            # assume it is array of index input
            e = np.zeros((1, indexInput.shape[0]))
        else:
            sys.exit('index input is not recognite !')

        # for ip in range(self.nInput):
        countE=-1
        for ip in indexInput:
            [output,a]=self.feedforward(self.p[:,ip])
            countE = countE + 1
            e[:,countE]=(self.target[:,ip]-output)

        sumError=self.calError(e,errorType)

        return sumError

    def unified(self,loopInInput,tol,visualize,visualizeStep):
        # we have to train target 0, 1, 0, 1
        classArr = np.unique(self.target, axis=1)
        nClass = len(classArr)
        arrSeq=np.arange(0,self.nSamp)
        indexClass0 = (self.target == classArr[0, 0])
        indexClass0 = np.array(indexClass0)
        posClass0=arrSeq[indexClass0[0,:]]
        indexClass1 = (self.target == classArr[0, 1])
        indexClass1 = np.array(indexClass1)
        posClass1=arrSeq[indexClass1[0,:]]

        posRunTrain=np.zeros((1,self.nSamp),dtype=int)
        countSamp0=-1
        countSamp1=-1
        for i in range(self.nSamp):

            if i==0 or (not (i%2)) :
                countSamp0 = countSamp0 + 1
                posRunTrain[0,i]=posClass0[countSamp0]
            elif i>0 and (i%2)>0:
                countSamp1 = countSamp1 + 1
                posRunTrain[0,i]=posClass1[countSamp1]

        icount=0
        itrain=0
        #sumErrorPlot=[]
        stopTrain=0
        while (icount <= loopInInput) and not(stopTrain):
        #for ip in range(self.nInput):
            icount=icount+1
            # find the error all

            for ip in posRunTrain[0,:]:
                # print(ip)
                itrain=itrain+1
                # get output of the network
                # update
                # out is a at last layer
                [output,a]=self.feedforward(self.p[:,ip])
                # print(f'output={output}')
                # print(f'target={self.target}')
                # output=a[self.nLayer-1] # output of network in the last layer


                # error
                # target is already set sample into column
                e=(self.target[:,ip]-output)
                # sumError=self.test('SAE')
                sumError = self.test('PBE')
                # print(f'sumError={sumError}')
                if visualize == 'text':
                    # sum of absolute error
                    # SAE=sum(abs(self.a[-1]-self.target))
                    print(f'Ite:{icount}/{loopInInput}, Input:{ip + 1}/{self.nInput}, eInput={e[0,0]}, eAll={sumError}')
                    # print(f'target[ip]={self.target[:,ip]}, output={output}, e={e}')
                    # print(f'wold={self.w[0]}, p={self.p[:,ip]}')
                if visualize == 'graph':
                    # elif visualize == 'graph':
                    print(f'Ite:{icount}/{loopInInput}, Input:{ip + 1}/{self.nInput}, eInput={e[0, 0]}, eAll={sumError}')

                    # sumErrorPlot.append(sumError)
                    plt.figure(1)
                    # point in each class
                    plt.cla()

                    plt.plot(self.p[0, indexClass0[0, :]], self.p[1, indexClass0[0, :]], 'or')
                    plt.plot(self.p[0, indexClass1[0, :]], self.p[1, indexClass1[0, :]], 'ob')
                    plt.plot(self.p[0,ip],self.p[1,ip],'og')
                    # plt.legend('Class_1','Class_2')

                    # limit of axis
                    minPx = np.min(self.p[0, :])
                    maxPx = np.max(self.p[0, :])
                    minPy = np.min(self.p[1, :])
                    maxPy = np.max(self.p[1, :])
                    rangeX = maxPx - minPx
                    percMaxMin = 0.3
                    xlim = [minPx - percMaxMin * rangeX, maxPx + percMaxMin * rangeX]
                    rangeY = maxPy - minPy
                    ylim = [minPy - percMaxMin * rangeY, maxPy + percMaxMin * maxPy]

                    # weigth vector
                    xW=self.w[0][0, 0]
                    yW=self.w[0][0, 1]
                    plt.arrow(0, 0, xW, yW, head_width=1, head_length=1,color='blue')
                    #plt.text(self.w[0][0, 0], self.w[0][0, 1], 'Weight vector')

                    # dicision boundary(orthogonal with weight vector)
                    # decision line is  >> a=f(n)=hardlim(wp+b)
                    # w1*x1+w2*x2+b = 0
                    # x2 = (-b-w1*x1)/w2
                    minW = np.min(self.w[0])
                    maxW = np.max(self.w[0])

                    # minX = min(minPx, minW)
                    # maxX = max(maxPx, maxW)
                    minX=np.min(xlim)
                    maxX=np.max(xlim)

                    x = np.linspace(minX, maxX, 2)
                    yPlot = -(self.b[0][0, 0] + self.w[0][0, 0] * x) / self.w[0][0, 1]
                    # print(f'x={x}, yPlot={yPlot}')
                    plt.plot(x, yPlot,'-k') # Dicision boundary
                    # plt.text(np.min(yPlot), np.max(yPlot), 'Dicision Boundary')

                    # plt.xlim(xlim)
                    # plt.ylim(ylim)
                    plt.xlim([-5,20])
                    plt.ylim([-5,20])
                    # plt.axis('equal')
                    # plt.legend(['Data class_1','Data class_2','Vector Weight','Dicision Boundary'])
                    if type(visualizeStep)==str:
                        if visualizeStep=='click':
                            plt.waitforbuttonpress()
                            plt.pause(0.05) # "plt.waitforbuttonpress()" need delay, or the plot will crash
                        else:
                            print('Please set visualizeStep="click" for ')
                    elif type(visualizeStep)==float or type(visualizeStep)==int:
                        plt.pause(visualizeStep)
                    # plt.show()

                if sumError <= tol:
                    print('Stop training, error reach critiria !')
                    stopTrain=1

                    plt.show()
                    break

                elif icount == loopInInput:
                    print('Stop training, reach limit of iteration.')
                    stopTrain=1
                    plt.show()
                    break

                else:
                    if abs(e) > 0.1: # we can't set == 0, due to float number
                        # update w and b
                        # Wnew = Wold + (t-a)p
                        wBuff=self.w[0].transpose()
                        #self.w[0].transpose()=self.w[0]+e[0,0]*self.p[:,ip];
                        wBuff=wBuff+e[0,0]*self.p[:,ip]
                        self.w[0]=wBuff.transpose()

                        # biasNew = biasOld + e
                        self.b[0]=self.b[0]+e

    def BP(self,lr,batchSize,errorType,TolAll,TolBatch,epoch,nIte,visualize,visualizeStep):
        '''
        backpropagation learning

        Wnew = Wold - lr*(dE/dW)
        Bnew = Bold - lr*(dE/dW)

        sumError
        there we have 2 part to calculate error
        1. sum error of all input
        2. sum error of bath size
        '''
        # errorArr=np.zeros((1,batchSize))
        
        # preallocate training parameter
        # TolBatch=0.1

        sen={}
        sumSen={}
        sumSenOut={}
        diagF={}

        # Calculate the number of set after divided into input batchSize
        modnBatch=self.nInput%batchSize
        if modnBatch==0:
            nWinBatch = int(self.nInput / batchSize)
        else:
            nWinBatch=int(np.floor(self.nInput/batchSize))+1

        # How handle with residual
        stopTrain=False
        stopMessage='Somethine wrong, stop training with unknow reason !'

        # preallocate for visualize
        if visualize=='graph':
            countWinError=-1
            nWinError=100
            xWinError=np.arange(0,nWinError)
            xWinError=xWinError.reshape(1,nWinError)
            winSumError=np.zeros((1,nWinError))

        for iepoch in range(epoch):
            # find error
            countBatch=0
            countWinBatch=0

            for ibatch in range(nWinBatch):
                # Find index of current batch size
                #  0 :     0         -      batchSize-1
                #  1 : batchSize     -      2*batchSize-1
                #  2 : 2*batchSize   -      3*batchSize-1
                if modnBatch==0 or (modnBatch>0 and ibatch<nWinBatch-1):
                    idxStart=ibatch*batchSize
                    idxStop=idxStart+batchSize-1
                elif modnBatch>0 and ibatch==nWinBatch-1:
                    idxStart = ibatch * batchSize
                    idxStop = idxStart + (modnBatch - 1)

                indexInputInBatch=np.arange(idxStart,idxStop+1,1)


                for iter in range(nIte): # iterate in each batch size
                    # loop in indexing
                    # calculate error before start training
                    errorInBatch = self.test('SE', indexInputInBatch)
                    if errorInBatch <= TolBatch:
                        # move to the next batch set
                        print('Move to next Batch !')
                        break

                    # Preallocate and also use to clear before update the next batch
                    for ilayer in range(self.nLayer):
                        # nNodeArr start with num input,num node hid1,num node hid2,...
                        # number of s is same as n node
                        sen[ilayer] = np.mat(np.zeros((self.nNodeArr[ilayer + 1], 1), dtype=float))
                        sumSen[ilayer] = np.mat(np.zeros((self.nNodeArr[ilayer + 1], 1), dtype=float))
                        sumSenOut[ilayer] = np.mat(np.zeros((self.nNodeArr[ilayer + 1], 1), dtype=float))
                        diagF[ilayer] = np.mat(np.zeros((self.nNodeArr[ilayer + 1], self.nNodeArr[ilayer + 1]), dtype=float))

                    # for ip in range(self.nInput):
                    for ipb in range(len(indexInputInBatch)):
                        # batchSize and number of input may mismatch
                        ip=int(indexInputInBatch[ipb])

                        countBatch=countBatch+1


                        pIn=self.p[:,ip]

                        [output,a]=self.feedforward(pIn)

                        error=self.target[:,ip]-output
                        # errorArr[countBatch]=sum(np.abs(error))

                        # loop from the last to the first layer
                        #nLayerBack=np.arange(self.nLayer-1,0,-1)
                        #for ilayer in nLayerBack:
                        if self.nLayer>1:
                            listLayerBack=np.arange(self.nLayer-1,-1,-1)
                        elif self.nLayer==1:
                            listLayerBack=np.array([0])
                        else:
                            print('nLayer is out of range !')
                            sys.exit()

                        for ilayer in listLayerBack:
                            #diagF[ilayer]=np.diag(np.transpose(f(self,self.n[ilayer])))
                            # diagflat can set input as row or vector
                            # print(self.n)
                            # print(self.n[ilayer])
                            diagFCol=self.F(self.n[ilayer],self.transfArr[ilayer])

                            # print(f'diagFCol={diagFCol}')
                            # print(f'flat={diagFCol.flatten()}')
                            # np.diag need input as np.array >> input = [1 2 3]
                            # if input = [[1] [2] [3]] function not work
                            diagF[ilayer]=np.diag(diagFCol.flatten())
                            # print(f'diagF={diagF}')
                            if (ilayer < self.nLayer-1) and (ilayer >= 0):
                            # not the last layer
                                #sen[ilayer]=f(self,self.n[ilayer],self.transfArr[ilayer])*self.w[ilayer+1]*sen[ilayer+1]
                                sen[ilayer]=diagF[ilayer]*self.w[ilayer+1].transpose()*sen[ilayer+1]

                            elif ilayer == (self.nLayer-1):
                            # the Last layer
                                #sen[ilayer]=-2*f(self,self.n[ilayer],self.transfArr[ilayer])*error
                                sen[ilayer]=-2*diagF[ilayer]*error

                            if ilayer == 0:
                                aProvLayer = pIn
                            elif ilayer > 0 and ilayer <= self.nLayer - 1:
                                aProvLayer = a[ilayer]

                            sumSen[ilayer]=sumSen[ilayer] + sen[ilayer]
                            sumSenOut[ilayer]=sumSenOut[ilayer] + sen[ilayer]*aProvLayer


                            # if countBatch==batchSize or ip==self.nInput-1:
                            # if is last input of the current batchSize
                            # print(f'lastIndex={indexInputInBatch[-1]}')
                            if ip==indexInputInBatch[-1]:
                                # Update the weight and bias when finish collect error of each batch size
                                # But please remind, we not update it only one times

                                # self.w[ilayer]=self.w[ilayer]-lr*sumSenOut[ilayer]*self.a[ilayer]*aProvLayer
                                countWinBatch=countWinBatch+1
                                self.w[ilayer] = self.w[ilayer] - lr*sumSenOut[ilayer]
                                self.b[ilayer] = self.b[ilayer] - lr*sumSen[ilayer]

                                errorInBatch = self.test('SE', indexInputInBatch)
                                # error of all input
                                sumErrAll = self.test(errorType,-1)
                                # combine error of batch size
                                if visualize=='text':
                                    print(f'Ep:{iepoch+1}/{epoch}, ib={ibatch+1}/{nWinBatch}, eInBatch={errorInBatch}, eAll={sumErrAll}')

                                elif (visualize=='graph' and (iter%1 == 0)) or (visualize=='Last' and (sumErrAll<=TolAll or iepoch >= epoch-1)):
                                    print(f'Ep:{iepoch+1}/{epoch}, ib={ibatch+1}/{nWinBatch}, eInBatch={errorInBatch}, eAll={sumErrAll}')

                                    plt.figure(1)
                                    # point in each class

                                    # # target data
                                    # plt.subplot(2,1,1)
                                    plt.cla()
                                    # Ref signal
                                    plt.plot(self.p,self.target,'g-o')
                                    # Input in batch
                                    plt.plot(self.p[:,indexInputInBatch],self.target[:,indexInputInBatch],'r-o')

                                    # output of the network
                                    outArr=np.zeros((1,self.nInput))
                                    for ipa in range(self.nInput):
                                        [output,abuff]=self.feedforward(self.p[:,ipa])
                                        outArr[0,ipa]=output
                                    # it is only one output of the network
                                    # plt.plot(self.p,self.a[self.nLayer-1],'r-o')
                                    plt.plot(self.p, outArr, 'k-o')
                                    plt.ylim([np.min(self.target)-0.5,np.max(self.target)+0.5])
                                    # plt.subplot(2,1,2)
                                    # plt.cla()
                                    # countWinError=countWinError+1
                                    # if countWinError<nWinError:
                                    #     winSumError[0,countWinError]=sumErrAll
                                    # elif countWinError>=nWinError:
                                    #     # winSumError=np.concatenate(winSumError[0,1:nWinError-1],sumErrAll)
                                    #     winSumError[0,0:nWinError-3]=winSumError[0,1:nWinError-2]
                                    #     winSumError[0, nWinError-1] = sumErrAll
                                    # plt.plot(xWinError,winSumError,'k--o',linewidth=2)
                                    # # plt.plot(xWinError, winSumError)

                                    if type(visualizeStep) == str:
                                        if visualizeStep == 'click':
                                            plt.waitforbuttonpress()
                                            plt.pause(0.05)  # "plt.waitforbuttonpress()" need delay, or the plot will crash
                                        else:
                                            print('Please set visualizeStep="click" for ')
                                    elif type(visualizeStep) == float or type(visualizeStep) == int:
                                        plt.pause(visualizeStep)


                                # else:
                                #     sys.exit('Visualize is out of range !')

                                if sumErrAll<=TolAll:
                                    stopTrain=True
                                    stopMessage='Stop training, error reaching criteria'
                                    plt.show()
                                    sys.exit(f'Stop training, error reaching Tol(={Tol})')
                                if iepoch >= epoch-1:
                                    stopTrain=True
                                    plt.show()
                                    print(f'Epoch:{iepoch + 1}/{epoch}, ibatch={ibatch + 1}/{nWinBatch},  SE={sumErrAll}')
                                    sys.exit(f'Stop training, Reaching epoch limitation !(MaxIte={nIte})')

                                if stopTrain:
                                    print(stopMessage)
                                    if visualize=='graph':
                                        plot.show()
                                    sys.exit()
                                else:
                                    # reset countBatch when finish update the weight in the first layer
                                    if ilayer==0:
                                        countBatch = 0


    def VLPBP(self):
#        elif lrRule=='VLRBP':
    # Variable learning rate Back propagation
        print('Update the weight using Variable learning rate Back Prop')
        self.w[0]=self.w[0]+200
        print(f'weight layer {0} = {self.w[0]}')

#        else:
#            print('Learning rule was not recognite !')

    # method in class train

