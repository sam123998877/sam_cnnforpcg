#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np

class Result(object):
    """Results for a single classification test

    lossTrain,accTrain,lossValid,accValid are lists that contain results at
    each learning epoch
    
    處理單個分類測試的結果的Function
顯示的內容包含訓練與驗證的損失函數(Loss)與準確度(Acc)

    For definitions, refer to:
        https://en.wikipedia.org/wiki/Precision_and_recall
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient
    """
    def __init__(self, saveFolder='', modelName='',featureName='', featureDims=[]):
        '''Constructor

        Args:
            saveFolder (str): Path to where results will be saved 
                              保存結果的路徑
            modelName (str): Name of the machine learning model ('uocSeq0', 'uocSeq1', etc.) 
                             機器學習模型的名稱  如: 'uocSeq0' 'uocSeq1' …等
            featureName (str): Name of the feature ('mfcc', 'subenv', etc.) 
                               特徵的名稱  如: ‘mfcc’ -梅爾頻率倒譜Mel-Frequency Cepstrum ,‘subenv’ - 子帶包絡sub-band envelopes …等
            featureDims (list): Dimensions of the feature [timeDimension,frequencyDimension] 
                                特徵的維度 [timeDimension，frequencyDimension]

        '''
        self.saveFolder=saveFolder
        #if folder not exists, create it /如果文件夾不存在，請創建它
        self._createFolder()

        self.modelName=modelName
        self.featureName=featureName
        self.featureDims=featureDims

        self.accuracy=-1
        self.sensitivity=-1
        self.specificity=-1
        self.F1=-1
        self.matthewsCorCoeff=-1
        self.ROC=[]
        self.confMat=None
        #results gathered during training at each epoch /在每個時代的訓練期間收集的結果
        self.lossTrain=[]
        self.accTrain=[]
        self.lossValid=[]
        self.accValid=[]
        self.tp=0 #true positive
        self.fp=0 #false positive
        self.tn=0 #true negative
        self.fn=0 #false negative
        #
        self.num_classes=0 #number of labels/categories in the data /數據中的標籤/類別數量
        self.test_y_probsFrame=[] #estimated probabilities for the test set on the frame level /在幀級別上測試集的估計概率
        self.test_y_predsFrame=[] #estimated labels for the test set on the frame level /幀級別上測試集的估計標籤
        self.test_y_trueFrame=[] #true labels(ground truth) for test frames /測試幀的真實標籤（基本事實）
        #
        #dictionary of file names versus true category and predicted category /文件名字典與真實類別和預測類別
        self.test_fileDecisions={}#example '0001.wav' -> [4 5] (file's true category was 4, predicted as 5) /例如'0001.wav' - > [4 5]（文件的真實類別為4，預測為5）
        #dictionary of file names versus predicted frame probabilities /文件名字典與預測幀概率
        self.test_fileProbs={}
        #dictionary of misclassified files versus their frame probabilities /錯誤分類文件的字典與其幀概率
        self.misLabeledFiles={}


    def _computeMeasures(self):
        ''''Main function for computing rates and the confusion matrix /計算Rate與混淆矩陣'''
        self.confMat=np.zeros((self.num_classes,self.num_classes))
        for (fileName,labels) in self.test_fileDecisions.items():
            true_class=int(labels[0])
            pred_class=int(labels[1])
            self.confMat[true_class,pred_class]+=1
            if true_class==pred_class: #correct classification /正確的分類
                if true_class==1:
                    self.tp+=1
                else:
                    self.tn+=1
            else: #false classification /錯誤的分類
                self.misLabeledFiles[fileName]=self.test_fileProbs[fileName]
                if pred_class==1:
                    self.fp+=1
                else:
                    self.fn+=1
        #tp,tn,fp,fn counted, compute measures from these counts /tp，tn，fp，fn計數，從這些計數中計算度量
        self._computeSensitivity()
        self._computeSpecificity()
        self._computeAccuracy()
        self._computeF1()
        self._computeMCC()

    def _computeSensitivity(self): #計算靈敏度(真陽性率)
        if (self.tp+self.fn)>0:#avoid zero-division error /避免零分割錯誤
            self.sensitivity=(self.tp)/(self.tp+self.fn)
        else:
            self.sensitivity=-1

    def _computeSpecificity(self): #計算特異度(真陰性率)
        if (self.tn+self.fp)>0:#avoid zero-division error /避免零分割錯誤
            self.specificity=(self.tn)/(self.tn+self.fp)
        else:
            self.specificity=-1

    def _computePrecision(self): #計算精確度
        if (self.tp+self.fp)>0:#avoid zero-division error /避免零分割錯誤
            self.precision=self.tp/(self.tp+self.fp)
        else:
            self.precision=-1

    def _computeAccuracy(self): #計算準確度
        if (self.tp+self.tn+self.fp+self.fn)>0:#avoid zero-division error /避免零分割錯誤
            self.accuracy=(self.tp+self.tn)/(self.tp+self.tn+self.fp+self.fn)
        else:
            self.accuracy=-1

    def _computeF1(self): #計算F1分數
        if (2*self.tp+self.fp+self.fn)>0:#avoid zero-division error /避免零分割錯誤
            self.F1=2*self.tp/(2*self.tp+self.fp+self.fn)
        else:
            self.F1=-1

    def _computeMCC(self): #計算 馬修斯相關係數(Matthews correlation coefficient , MCC)

        tp=self.tp
        tn=self.tn
        fp=self.fp
        fn=self.fn
        if ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))>0:#avoid zero-division error /避免零分割錯誤
            self.matthewsCorCoeff=(tp*tn-fp*fn)/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
        else:
            self.matthewsCorCoeff=-1


    '''Incremental updates on results /結果的增量更新'''
    def appendEpoch(self,scoreTrain,scoreValid):
        '''Appends new epoch results to existing list of result values /將新的結果 附加到原結果列表中

        Args:
            scoreTrain [list]: contains [lossTrain,accTrain]
            scoreValid [list]: contains [lossValid,accValid]

        '''
        self.lossTrain.append(scoreTrain[0])
        self.lossValid.append(scoreValid[0])
        self.accTrain.append(scoreTrain[1])
        self.accValid.append(scoreValid[1])

    '''Plotting functions /繪圖功能'''
    def plotLossVsEpochs(self,outFileName=str()):
        '''Plotting loss versus epoch number for train and validation /繪製損失、訓練和驗證的代數圖

        Args:
            outFileName (str): [optional] name of the file where plot will be saved /(選)將保存繪圖的文件的名稱
        '''
        plt.plot(np.array(range(1,len(self.lossTrain)+1)),np.array(self.lossTrain), color='black', label='Train')
        plt.plot(np.array(range(1,len(self.lossTrain)+1)),np.array(self.lossValid), color='red', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        if not outFileName==str():#save to file if outFileName specified /如果指定了outFileName，則保存到文件
            plt.savefig(outFileName,dpi=300)
            plt.clf()

    def plotAccVsEpochs(self,outFileName=str()):
        '''Plotting accuracy versus epoch number for train and validation /繪製準確度、訓練和驗證的代數圖

        Args:
            outFileName (str): [optional] name of the file where plot will be saved /(選)將保存繪圖的文件的名稱
        '''
        plt.plot(np.array(range(1,len(self.accTrain)+1)),np.array(self.accTrain), color='black', label='Train')
        plt.plot(np.array(range(1,len(self.accTrain)+1)),np.array(self.accValid), color='red', label='Validation')
        plt.xlabel('Number of epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='upper right')
        if not outFileName==str():#save to file if outFileName specified /如果指定了outFileName，則保存到文件
            plt.savefig(outFileName,dpi=300)
            plt.clf()

    def report(self,filename):
        '''Writing measures and misclassified files to report/results file /將評估(Measure)和錯誤分類的文件 寫入報告/結果檔案內''' 
        self._computeMeasures()
        with open(filename, "w") as text_file:
            text_file.write('File-level measures\n')
            for rowInd in range(self.num_classes):
                for colInd in range(self.num_classes):
                    text_file.write(str(self.confMat[rowInd,colInd])+'\t|')
                text_file.write('\n')
            text_file.write('\nSensitivity =\t'+str(self.sensitivity))
            text_file.write('\nSpecificity =\t'+str(self.specificity))
            text_file.write('\nAccuracy    =\t'+str(self.accuracy))
            text_file.write('\nF1          =\t'+str(self.F1))
            text_file.write('\nMatthews CC =\t'+str(self.matthewsCorCoeff))
            text_file.write('\n------------------------------------------\n')
            text_file.write('Frame-level measures\n')
            (frmConfMat,frmSens,frmSpec)=self._computeFrameMeasures()
            for rowInd in range(self.num_classes):
                for colInd in range(self.num_classes):
                    text_file.write(str(frmConfMat[rowInd,colInd])+'\t|')
                text_file.write('\n')
            text_file.write('\nSensitivity =\t'+str(frmSens))
            text_file.write('\nSpecificity =\t'+str(frmSpec)+'\n')
            self.frameConfMat=frmConfMat
            self.frameSensitivity=frmSens
            self.frameSpecificity=frmSpec
            text_file.write('\n------------------------------------------\n')

        ROCfile=filename.replace('.txt','_ROC.png')
        self._plotROC(ROCfile)

    def _computeFrameMeasures(self):
        ''''Computation of measures on frame level /計算各個框架水平的評估(Measure)結果'''
        confMat=np.zeros((self.num_classes,self.num_classes))
        (tp,tn,fp,fn)=(0,0,0,0)
        for ind in range(len(self.test_y_trueFrame)):
            true_class=self.test_y_trueFrame[ind]
            pred_class=self.test_y_predsFrame[ind]
            confMat[true_class,pred_class]+=1
            if true_class==pred_class:#correct classification /正確的分類
                if true_class==1:
                    tp+=1
                else:
                    tn+=1
            else:#false classification /錯誤的分類
                if pred_class==1:
                    fp+=1
                else:
                    fn+=1
        if (tp+fn)>0:#avoid zero-division error /避免零分割錯誤
            sensitivity=(tp)/(tp+fn)
        else:
            sensitivity=-1
        
        if (tn+fp)>0:#avoid zero-division error /避免零分割錯誤
            specificity=(tn)/(tn+fp)
        else:
            specificity=-1
            
        return (confMat,sensitivity,specificity)
        
    def _plotROC(self,ROCfile,title=''):
        '''Re-performing file level decision from frame level probabilities
        Implemented for binary-classification tasks. For more than 2 labels, 
        consider implementing a new version
        
        For computing mean frame probabilities, the probabilities are sorted
        and the values at two ends are dropped out first. The number of frames 
        to drop is controlled by the portionLeaveOutFrms variable which is 
        specified in percentage. Ex: if portionLeaveOutFrms=40, that means
        20% lowest values and 20% highest will be left out and the mean will 
        be computed afterwards 

        重新執行文件級別決策
        來自幀級概率為二進制分類任務實現。
        對於2個以上的標籤，請考慮實施新版本
                
        為了計算平均幀概率，對概率進行排序，並且首先丟棄兩端的值。
        要丟棄的幀數由partialLeaveOutFrms變量控制，該變量以百分比指定。
        例如：如果partLeaveOutFrms = 40，那意味著20％的最低值和20％的最高值將被省略，平均值將在之後計算

        '''
        portionLeaveOutFrms=30#portion of frames to be left out as extreme values of probability to be able to compute a realiable mean-prob. /要將剩餘的幀的一部分作為能夠計算可實現的平均概率的概率的極值。
        #creating a dictionary: file -> average frame-level pathology probability /創建字典：文件 - >平均幀級病理概率
        meanPathologyProbs={}
        for curFile in self.test_fileProbs:
            frameProbs=self.test_fileProbs[curFile]
            #sumPathProb=0#sum of probability of pathology /sumPathProb = 0＃病理概率之和
            allPathProb=[]#we will collect all frame pathology probabilities in a list /我們將在列表中收集所有框架病理概率
            for frmProb in frameProbs:#each frame's probability vector /每幀的概率向量
                #sumPathProb+=frmProb[1]
                allPathProb.append(frmProb[1])
            #for leaving out extreme prob. values, values will be sorted and the mid-values will be used /遺漏極端的概率。 將對值進行排序，並使用中間值
            sortedProb=np.sort(allPathProb)
            numFrms2leaveOnEnds=int(np.round(len(sortedProb)*(portionLeaveOutFrms/2)/100))
            if len(sortedProb[numFrms2leaveOnEnds:-numFrms2leaveOnEnds])>3:#if at least 3 frames left after removing extremes /如果在消除極端情況後至少剩下3幀
                meanVal=np.mean(sortedProb[numFrms2leaveOnEnds:-numFrms2leaveOnEnds])
            else:
                meanVal=np.mean(sortedProb)
            #meanPathologyProbs[curFile]=(sumPathProb/len(frameProbs))
            meanPathologyProbs[curFile]=meanVal

        #trying different threshold values to compute a ROC curve /嘗試不同的閾值來計算ROC曲線
        allTpr=[];allFpr=[];#ROC curve y and x points, Tpr: true positive rate, Fpr: false positive rate /ROC曲線y和x點，Tpr：真陽性率，Fpr：假陽性率
        for threshold in np.linspace(0.0, 1.0, num=100):#assiging new threshold values in range 0-1 /在0-1範圍內分配新的閾值
            fileDecDict={}
            for curFile in self.test_fileProbs:
                (true_class,pred_class)=self.test_fileDecisions[curFile]
                prob=meanPathologyProbs[curFile]
                #Making a new decision via thresholding /通過閾值處理做出新的決定
                if prob>=threshold:
                    pred_class=1
                else:
                    pred_class=0
                fileDecDict[curFile]=[true_class,pred_class]
            (tpr,fpr)=self._computeROCpoint(fileDecDict)
            #add point to ROC curve /將點添加到ROC曲線
            allTpr.append(tpr)
            allFpr.append(fpr)
        #plotting the ROC curve /繪製ROC曲線
        plt.clf()
        plt.plot(np.array(allFpr),np.array(allTpr), color='black')
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(title, loc='left')
        plt.savefig(ROCfile,dpi=300)
        plt.clf()

        #storing ROC curve data to be able to plot various ROC curves in one figure /存儲ROC曲線數據，以便能夠在一個圖中繪製各種ROC曲線
        self.ROC=(np.array(allTpr),np.array(allFpr))

    def _computeROCpoint(self,fileDecDict):
        '''Given a file-decision dictionary, computes
        True positive rate(tpr) and
        False positive rate(fpr)

        It is assumed that the dictionary has the following structure:
            file -> [true_class,predicted_class]

        (This function is used for computing ROC curves)
        
        給定文件決策字典，計算
         真陽性率（tpr）和
         誤報率（fpr）

         假設字典具有以下結構：
             file - > [true_class，predict_class]

         （此函數用於計算ROC曲線）
        '''
        tp=0; fp=0; tn=0; fn=0
        for (fileName,labels) in fileDecDict.items():
            true_class=int(labels[0])
            pred_class=int(labels[1])
            if true_class==pred_class:#correct classification /正確的分類
                if true_class==1:
                    tp+=1
                else:
                    tn+=1
            else:#false classification /錯誤的分類
                if pred_class==1:
                    fp+=1
                else:
                    fn+=1
        tpr=tp/(tp+fn)
        fpr=fp/(fp+tn)
        return (tpr,fpr)

    def _createFolder(self):
        '''Checks existence of the folder and creates if it does not exist /檢查資料夾是否存在 不存在就創建'''
        if not os.path.exists(self.saveFolder):
            os.makedirs(self.saveFolder)
