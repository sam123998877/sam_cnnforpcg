#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test definition embodying all components of a learning experiment:

    Data (files as well as segmentation and features specifications),
    Machine learning model
    Test specifications (batch and eopch sizes, etc.)

Created on May 15th 2017

@author: Baris Bozkurt
"""
import pickle
import time
import numpy as np
from keras.models import load_model
from Result import Result
from models import loadModel
from keras.layers import Conv2D, MaxPooling2D

Strides = [(1,1) , (2,2) , (3,3)]
Activation_functions = ['softsign' , 'tanh' , 'relu']
Pooling = [ MaxPooling2D(pool_size=(1, 1))  , MaxPooling2D(pool_size=(2, 2))  , MaxPooling2D(pool_size=(3, 3)) ]
Dropout_Rate = [0.1 , 0.2 , 0.3]


class Test(object):
    '''Design of a complete learning test/設計完整的學習測試

    '''
    def __init__(self,modelName,data,resultsFolder,batch_size=128,num_epochs=100,mergeType='majorityVoting'):
        '''Constructor

        Args:
            modelName (str): Name of the NN model to be applied on data.
                Model loaded using the loadModel() function defined in models.py
            data (Data): Data object (embodies data specifications as well
                features, segmentation, etc)
            resultsFolder (str): Path where results will be saved
            batch_size (int): Batch size (default: 128)
            num_epochs (int): Number of epochs for the learning tests (default: 100)
            mergeType (str): If more than a single feature is used, the final
                outputs obtained for each feature can be merged. mergeType
                defines the strategy for merging this information.
                Ex:"majorityVoting", etc. If not specified, default
                method ('majorityVoting') will be applied.

        '''
        self.modelName=modelName
        self.data=data
        self.mergeType=mergeType
        self.batch_size=batch_size
        self.num_epochs=num_epochs
        self.resultsFolder=resultsFolder
        #opening a report file for printing logs
        localtime = time.asctime( time.localtime(time.time()))
        self.logFile=open(resultsFolder+'TestRep_'+localtime.replace(':','').replace(' ','_')[4:-7]+'.txt','w')
        self.logFile.write(self.data.info+'\n')
        self.logFile.write('Model name: '+modelName+', dba: '+data.name+'\n')
        self.logFile.write('Test started at: '+localtime+'\n')


    def run(self):
        '''Runs the test for each feature in the feature list/對要素列表中的每個要素運行測試

        Each test on a separate feature is performed in isolation due to memory
        concerns. Outputs are written to files
        由於內存的原因，對單獨功能的每個測試都是獨立執行的關注。 輸出將寫入文件
        '''
        #Load and perform test for each feature/Load並對每個功能執行測試
        #For each feature a Keras model needs to be separately loaded since the input size/對於每個功能，自輸入大小以來需要單獨加載Keras模型
        #is defined by the feature data size/因為輸入大小由特徵數據大小定義
        counter=0
        for counter in range(9):
            for feature in self.data.features:
                #new file and variables for outputs of this test for the specific feature/特定功能的此測試輸出的新文件和變量
                dimsStr=str(feature.dimensions[0])+'by'+str(feature.dimensions[1])
                if feature.involveDelta:
                    deltaStr='del'
                else:
                    deltaStr=''
                self.modelFile=self.resultsFolder+'M'+'_'+self.modelName+feature.name+dimsStr+deltaStr+'_'+feature.segmentation.abbr+feature.window4frame[0]+'.h5'
                self.logFile.write('-------------------------------------------------\n')
                self.logFile.write('New test initiated: '+self.modelFile.replace('.h5','')+'\n')
                self.logFile.write('Feature: '+feature.name+dimsStr+' using segmentation: '+feature.segmentation.sourceType+', size: '+feature.segmentation.sizeType+'\n')
                self.logFile.write('\tperiodSync:'+str(feature.segmentation.periodSync)+', with window: '+feature.window4frame+'\n')
                print('-------------------------------------------------')
                print('New test initiated: '+self.modelFile.replace('.h5',''))
                #initiating a new Result object for computing and writing results/启动一个新的Result对象来计算和编写结果
                self.result=Result(self.resultsFolder,self.modelName,feature.name,feature.dimensions)
    
                #Loading train, validation, test samples
                # data.train_x, .train_y, .valid_x, .valid_y, .test_x, .test_y
                dataIsLoaded=self.data.loadSplittedData(feature)
                #If data could be loaded, performing the test/如果可以加载数据，则执行测试
                if dataIsLoaded:
                    self.result.num_classes=self.data.num_classes
                    #Load NN model
                    
                    if counter==0:
                        sn=0;   an=0;  pn=0;  dn=0;
                    elif counter==1:
                        sn=0;   an=1;  pn=1;  dn=1;
                    elif counter==2:
                        sn=0;   an=2;  pn=2;  dn=2;
                    elif counter==3:
                        sn=1;   an=0;  pn=1;  dn=2;
                    elif counter==4:
                        sn=1;   an=1;  pn=2;  dn=0;
                    elif counter==5:
                        sn=1;   an=2;  pn=0;  dn=1;
                    elif counter==6:
                        sn=2;   an=0;  pn=2;  dn=1;
                    elif counter==7:
                        sn=2;   an=1;  pn=0;  dn=2;
                    elif counter==8:
                        sn=2;   an=2;  pn=1;  dn=0;
                        
                    self.model=loadModel(self.modelName,self.data.shape,self.data.num_classes,Strides[sn],Activation_functions[an],Pooling[pn],Dropout_Rate[dn])
                    #perform training
                    self._train()
                    #perform testing
                    self._test()
                    #printing to log
                    self.logFile.write('---Experiment '+ str(counter) +'---'+'\n')
                    self.logFile.write('Results saved for '+self.testName+'\n')
                    self.logFile.write(' '+self.testName+'\n')
                    self.logFile.write(time.asctime( time.localtime(time.time()))+'\n')
                    print('Results saved for '+self.testName)
                #printing to log
                self.logFile.write('-------------------------------------------------\n')
                print('-------------------------------------------------')
            #printing to log
            localtime = time.asctime( time.localtime(time.time()))
            self.logFile.write('Test finished at: '+localtime+'\n')
            self.logFile.close()
            
            counter = counter + 1

    def _train(self):
        '''Training step

        Saves model that gives highest accuracy on validation(on frame level)
        in an h5 file. Plots change of loss and accuracy versus epochs
        保存模型，在验证时提供最高精度（在帧级别）在h5文件中。 绘图损失和准确性与时期的变化
        '''
        #Creating a unique name for the test/為測試創建唯一名稱
        self.testName=self.modelFile.replace('.h5','')+'_'+self.mergeType[0:3]
        #printing to log/打印到日誌
        self.logFile.write('Training: '+self.testName+'\n')
        self.logFile.write('Number of frames/features: (train,validation,test): '+str([len(self.data.train_x),len(self.data.valid_x),len(self.data.test_x)])+'\n')
        #count data in category for/計算類別中的數據
        uniqueLabels, counts = np.unique(np.argmax(self.data.train_y,axis=1), return_counts=True)
        self.logFile.write('Train categories and number of samples: '+str(uniqueLabels)+', '+str(counts)+'\n')
        print('Number of frames/features: (train,validation,test): ',len(self.data.train_x),len(self.data.valid_x),len(self.data.test_x))
        print('Train categories and number of samples: '+str(uniqueLabels)+', '+str(counts))

        highestTestScore=0
        #-----TRAINING-----
        #Creating a flag to track if there is any learning or not. It is possible in some cases that accuracy values 
        #   do not change at all for all the learning process. Then, this 
        #创建一个标记以跟踪是否有任何学习。
        #在某些情况下，所有学习过程的准确度值都不会发生变化。 然后，这个
        validScoreNeverIncreased=True
        for epoch in range(self.num_epochs):
            if epoch%1==0:#print position each 20 epochs
                print('Epoch:',epoch)
            #One epoch of learning
            self.model.fit(self.data.train_x, self.data.train_y,batch_size=self.batch_size,epochs=1,verbose=0,
                  validation_data=(self.data.valid_x, self.data.valid_y))
            scoreValid = self.model.evaluate(self.data.valid_x, self.data.valid_y, verbose=0)
            scoreTrain = self.model.evaluate(self.data.train_x, self.data.train_y, verbose=0)
            #Storing the model in a h5 file if accuracy on validation is improved/如果验证的准确性得到改善，则将模型存储在h5文件中
            if scoreValid[1]>highestTestScore:#scoreValid[0]代表loss,scoreValid[1]代表accuary,highestTestScore=0
                validScoreNeverIncreased=False
                highestTestScore=scoreValid[1]
                #saving model with highest score
                self.model.save(self.modelFile)
                self.logFile.write('Model saved: epoch %d, accuracyOnValidation: %f\n' % (epoch,highestTestScore))
                print('Model saved: epoch %d, accuracyOnValidation: %f' % (epoch,highestTestScore))
            if scoreTrain[1]>0.99999 and False:#comment part after 'and' if you like to stop training when an accuracy of 1 is achieved for train
                break
            self.result.appendEpoch(scoreTrain,scoreValid)        
        #Report training results/报告训練结果
        localtime = time.asctime( time.localtime(time.time()))
        self.result.plotLossVsEpochs(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_loss.png')
        self.result.plotAccVsEpochs(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_acc.png')  
        
        if validScoreNeverIncreased:
            self.result.learningFailed=True
        else:#this flag will be checked while computing average of success measures, if no learning happened it should not be used in average comp./在計算成功度量的平均值時將檢查此標誌，如果沒有學習，則不應在平均值中使用。
            self.result.learningFailed=False

    def _test(self):
        '''Testing the learned model(using train and validation sets)
            on the test data set
            測試學習模型（使用訓練集和驗證集)在測試數據集上

        Assigns:
            self.result.test_fileDecisions (dict {str->[#,#]}): File level decisions/文件級別決策
                {file-name, [true_class,predicted_class]}
            self.result.test_fileProbs (dict {str->[#,#]}): Frame classification probabilities for a file/文件的幀分類概率
                {file-name, [probabilities_of_frames]}
        '''
        #Load best model saved in modelFile for this test/加載保存在modelFile中的最佳模型以進行此測試
        model = load_model(self.modelFile)#load_model is imported from Keras/load_model是從Keras導入的
        #Perform estimation on the test data set/對測試數據集執行估計
        #y_probs=model.predict(self.data.test_x, batch_size=self.data.test_x.shape[0], verbose=0)#原始
        y_probs=model.predict(self.data.test_x, batch_size=self.data.test_x.shape[0], verbose=1)#泓翔新增
        y_probs_valid=model.predict(self.data.valid_x, batch_size=self.data.valid_x.shape[0], verbose=1)#泓翔新增
        print("y_probs:",y_probs[:10])#泓翔新增
        text_file_predict = open("predict.txt", "w")#泓翔新增
        text_file_predict.write('test result(predict)\n')#泓翔新增
        text_file_predict.write('\n')#泓翔新增
        text_file_predict.write('\nresult_preduct_y_probs =\t'+str(y_probs[:15]))#泓翔新增
        text_file_predict.write('\nresult_preduct_y_probs_valid=\t'+str(y_probs_valid[:15]))#泓翔新增
        text_file_predict.write('\nresult_preduct_y_probs_valid_feature =\t'+str(self.data.valid_x[:10])+'\nresult_preduct_y_probs_valid_label =\t'+str(y_probs_valid[:10]))#泓翔新增
        text_file_predict.write('\n------------------------------------------\n')#泓翔新增
        text_file_predict.close()#泓翔新增
        #Ground truth is in test_y in a vector form, convert to 1 dimensional label and store as test_y_trueFrame
        #基本事實在矢量形式的test_y中，轉換為1維標籤並存儲為test_y_trueFrame
        self.result.test_y_trueFrame=np.argmax(self.data.test_y,axis=1)
        self.result.test_y_probsFrame=y_probs
        self.result.test_patSegMaps=self.data.test_patSegMaps
        #converting from activation output value to class-index by picking the largest value/ 通過選擇最大值從激活輸出值轉換為類索引
        self.result.test_y_predsFrame=np.argmax(y_probs,axis=1)

        for curFile in self.data.test_patSegMaps:#for each file in the test set/對於測試集中的每個文件
            segInds=self.data.test_patSegMaps[curFile]#get indexes of frames for the file/獲取文件的幀索引
            #Get true class for the file from frame labels(all should be the same: checked within the function)
            #從幀標籤獲取文件的真正類（所有應該相同：在函數內檢查）
            true_class=self._patientTrueClass(self.result.test_y_trueFrame[segInds])
            #Get estimated probabilities for frames of a file/獲取文件幀的估計概率
            filesFrameProbs=list(y_probs[segInds])
            self.result.test_fileProbs[curFile]=filesFrameProbs
            #Using the 'mergeType', make a file level prediction from frame level probabilities
            #使用'mergeType'，從幀級概率進行文件級預測
            pred_class=self._patientPredClass(filesFrameProbs)
            self.result.test_fileDecisions[curFile]=[true_class,pred_class]#store true and predicted classes/存儲真實和預測的類

        localtime = time.asctime( time.localtime(time.time()))
        self.result.report(self.testName+localtime.replace(':','').replace(' ','_')[4:-7]+'_res.txt')

        #saving results
        pickleProtocol=1#choosen for backward compatibility/選擇向後兼容
        with open(self.modelFile+localtime.replace(':','').replace(' ','_')[4:-7]+self.mergeType[0:3]+'.pkl' , 'wb') as f:
            pickle.dump(self.result, f, pickleProtocol)

    def _patientPredClass(self,pat_y_probs):
        '''File level estimation from file's estimated frame probabilities

        To be implemented: mergeType='meanProbs': computing mean probability of
            frames after removing outliers and taking decision from mean-probability
            根據文件的估計幀概率估計文件級別

         要實現：mergeType ='meanProbs'：計算平均概率
             刪除異常值並從均值概率中作出決定後的幀

        '''
        if self.mergeType=="majorityVoting":
            pat_y_preds=np.argmax(pat_y_probs,axis=1)
            pat_y_preds=list(pat_y_preds)
            return max(pat_y_preds,key=pat_y_preds.count)
        else:
            print('Unknown merge type from frame level to file level')
            return -1

    def _patientTrueClass(self,trueFrameClasses):
        '''Given all frame level decisions for a specific file/patient,
        checks if they all are the same (they should be!)
        and returns that common value as the true class else -1
        給定特定文件/患者的所有幀級決策，
         檢查它們是否都相同（它們應該是！）
         並將該公共值作為真正的類返回-1        
        '''
        trueClass=trueFrameClasses[0]
        for cat in trueFrameClasses:
            if trueClass!=cat:
                self.logFile.write('Error: frame level true classes do not match for a patient\n')#錯誤：幀級true類與患者不匹配
                print('Error: frame level true classes do not match for a patient')#錯誤：幀級true類與患者不匹配
                print(trueFrameClasses)
                return -1
        return trueClass
    
  
        
        
