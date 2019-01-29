#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import os
import pickle
import random
from random import randint
import numpy as np
from keras.utils import to_categorical
import soundfile as sf
import resampy

class Data(object):
    '''Data specification, loading and processing functions
       規劃資料規格、讀取資料和處理的Function '''
    def __init__(self,name,wavFolder,featureFolder,features,useBalanced=True,splitRatios=[0.65,0.15,0.2],info="",dataAugRatio=1):
        '''Constructor

        Args:
            name (str): Name of the database. This will define how the
                data will be read. Ex: "Uoc","Physionet","UrbanSound", etc

            wavFolder (str): Path to folder containing wave files

            featureFolder (str): Path to folder containing feature files.
                If the feature files exist, they will be directly read. If they
                are missing, they will be computed and saved in subdirectories
                in this folder

            feature (list of Feature objects): Different features to be computed
                and used for this data(base).

            useBalanced (bool): Flag for using balanced data. If set, balancing
                will be performed by augmenting data with up-sampling

            splitRatios (list of floats): ratios of train, validation and test
                to applied as split ratios to the whole dba. ex: [0.65,0.15,0.20]
                If only two splits are given, it is assumed to refer to
                train-validation split and test data is separately available

            info (str): Additional info to be saved in report files

            dataAugRatio (float): Data augmentation ratio. If dataAugRatio>1,
                then this will indicate an augmentation request for the train
                data set. Augmentation will be carried via down-sampling of
                existing samples
                
        Args:
            name（str）：
                數據庫的名稱。這將定義如何讀取數據。
                例如：“Uoc”，“Physionet”，“UrbanSound” ...等

            wavFolder（str）：
                包含波形文件的文件夾的路徑

            featureFolder（str）：
                包含要素文件的文件夾的路徑。
                如果存在要素文件，則將直接讀取它們。
                如果它們丟失，它們將被計算並保存在此文件夾的子目錄中

            feature（list of Feature objects）：
                要計算並用於此數據的不同功能（基礎）。

            useBalanced（bool）：
                使用平衡數據的標誌。
                如果設置，將通過使用上採樣來擴充數據來執行平衡

            splitRatios（float）：
                列車，驗證和測試的比率，作為整體dba的分流比率。例如：[0.65,0.15,0.20]
                如果僅給出兩個分裂，則假設參考列車驗證分裂並且測試數據是單獨可用的

            info（str）：
                要在報告文件中保存的其他信息

            dataAugRatio（float）：
                數據增加率。
                如果dataAugRatio> 1，那麼這將指示對列車數據集的增強請求。
                增強將通過現有樣品的下採樣進行

        '''
        self.name=name
        self.wavFolder=wavFolder
        self._initiateLists()           #Call 初始化Function

        self.featureFolder=featureFolder
        self.features=features
        self.balancingPerformed=False
        self.splitRatios=splitRatios
        self.info=info           
        self.dataAugRatio=dataAugRatio  #disabled for this implementation
        self.useBalanced=useBalanced
        self.trainSetAugmented=False

        self.numTrainSamp=0             #number of train samples
        self.numValidSamp=0             #number of valid samples
        self.numTestSamp=0              #number of test samples
        self.numTestSampPerCat=[]       #number of test samples for each category

        self.preDefFileLabelMap={}      #pre-defined file-label map (used in Physionet2016 dba)


    def _initiateLists(self):
        '''Initializing lists in this object 初始化物件中的清單 '''
        self.trainFiles=[]#list of wave files for the train set
        self.validFiles=[]#list of wave files for the validation set
        self.testFiles=[]#list of wave files for the test set

        self.trainFileLabels=[]#labels for wave files for the train set
        self.validFileLabels=[]#labels for wave files for the validation set
        self.testFileLabels=[]#labels for wave files for the test set

        #x is used for inputs/feature-samples and y for outputs/labels
        self.train_x = []
        self.train_y = []
        self.valid_x = []
        self.valid_y = []
        self.test_x = []
        self.test_y = []

    def loadSplittedData(self,feature):
        '''Load feature data

        First checks if the pickle files are available in the featureFolder for the feature
        (in a subdirectory with name of the feauture). If the files are available, they are loaded and the data
        is returned. If the files are not available, feature estimation is performed, saved to file and reloaded.

        Args:
            featureName (str): name of the feature for which data will be loaded

        Returns:
            dataRead (bool): flag to indicate success of feature reading

        Assigns:
            self.train_x (numpy array): Input data(feature) for train
            self.train_y (numpy array): Outputs(labels) for train
            self.valid_x (numpy array): Input data(feature) for validation
            self.valid_y (numpy array): Outputs(labels) for validation
            self.test_x (numpy array) : Input data(feature) for test
            self.test_y (numpy array) : Outputs(labels) for test
            self.num_classes (int): number of classes
            
        首先檢查功能的featureFolder中是否有可用的pickle文件（在名稱為feauture的子目錄中）。
        如果文件可用，則加載它們並返回數據。
        如果文件不可用，則執行特徵估計，保存到文件並重新加載。

        ARGS：
             featureName（str）：要為其加載數據的特徵名稱

        返回：
             dataRead（bool）：表示功能讀取成功的標誌

        分配：
             self.train_x（numpy array）：用於訓練的輸入數據（特徵）
             self.train_y（numpy array）：用於訓練的輸出（標籤）
             self.valid_x（numpy array）：用於驗證的輸入數據（特徵）
             self.valid_y（numpy array）：用於驗證的輸出（標籤）
             self.test_x（numpy array）：用於測試的輸入數據（特徵）
             self.test_y（numpy array）：用於測試的輸出（標籤）
             self.num_classes（int）：分類的數量
            
        '''
        #Start by cleaning the lists, if not called, new data will be appended on previous/首先清理列表，如果沒有調用，新數據將附加在之前
        self._initiateLists()

        #Checking if feature files exist for train and test. Check是否存在用於訓練和測試的特徵文件。
        #   If validation does not exist, it will be obtained by applying random split of the train data
        #   如果不存在驗證集，則透過 隨機分割訓練集 來作為 驗證集

        #Creating path for feature files 創建特徵文件的資料夾路徑
        
        #先指定資料的維度
        dimsStr=str(feature.dimensions[0])+'by'+str(feature.dimensions[1])
        if feature.involveDelta:
            deltaStr='del'
        else:
            deltaStr=''
        #再設定資料夾位置
        subFolder=feature.name+dimsStr+deltaStr+'_'+feature.segmentation.abbr+feature.window4frame[0]+'/'
        self.featureSubFolder=subFolder
        dirName=self.featureFolder+subFolder
        
        #creates file names and checks their availability 創建文件名稱 並 檢查其可用性
        feature.checkFeatureFiles(dirName)

        if (not feature.trainFeatFilesExist):
            #Feature files missing, running feature extraction
            #特徵檔不存在，執行 特徵提取

            #If feature files folder does not exist, create it
            #如果 特徵檔資料夾不存在，創建它
            if not os.path.exists(dirName):
                os.mkdir(dirName)

            print('Computing feature: '+subFolder)
            #Get list of files in the wavFolder subDirectories: train, validation and test
            #   Data augmentation, balancing and splitting performed in this step
            # 獲取 音訊目錄中的文件列表： (即:訓練，驗證和測試集)
            # 並在此步驟中執行 數據增加，平衡和拆分 (如果需要的話)
            self.getWaveFilesAndLabels()

            #Running feature extraction for each file in a given file list
            #   Feature computation produces 'input' data and labels as 'true outputs'
            #   All are saved in pickle files. To be able to compute the features,
            #   segmentation is needed. Hence, segmentation files are read. If they are not
            #   available, segmentation is performed and saved into files
            #   Inexistence of a segmentation file indicates either some missing
            #   source(for example ecg-channel may be missing and ecg-based segmentation may have
            #   been targeted). This will result in skipping that file

            #   為給定文件列表中的每個文件運行 特徵提取
            #   特徵計算生成“輸入”數據並將標籤標記為“真實輸出”
            #   所有都保存在pickle文件中。
            #   為了能夠計算特徵，需要進行分割。
            #   因此，讀取分段文件。
            #   如果它們不存在，則執行分段Function 並將其保存
            #   分段文件的不存在 表示某些缺少的來源
            #   （例如，可能缺少ecg-channel並且可能已經針對基於ecg的分段）。
            #   這將導致跳過該文件

            #開始計算訓練集特徵
            feature.compute('train',self.trainFiles,self.trainFileLabels)
            #if source files for segmentation could not be found, feature will not be computed 
            #如果找不到用於分段的源文件，則不會計算特徵
            if len(feature.segmentation.startsMs)==0:
                return False
            #開始計算訓練集特徵
            feature.compute('valid',self.validFiles,self.validFileLabels)
            #開始計算訓練集特徵
            feature.compute('test',self.testFiles,self.testFileLabels)
            #檢查特徵檔案
            feature.checkFeatureFiles(dirName)

        #Reading train-feature/label files
        #讀取訓練集的 特徵與標籤文件(pickle形式)
        with open(feature.trainFeatureFile, 'rb') as f:
            trainExamples=pickle.load(f)
        with open(feature.trainLabelFile, 'rb') as f:
            trainLabels=pickle.load(f)
        with open(feature.trainMapFile, 'rb') as f:
            train_patSegMaps=pickle.load(f)
            
        #feature dimensions 設定特徵的維度
        dims=trainExamples[0].shape[0:2]
        if feature.involveDelta:
            featDims=(feature.dimensions[0],feature.dimensions[1]*2)
        else:
            featDims=(feature.dimensions[0],feature.dimensions[1])
            
        #Check match with feature dimension 檢查特徵的維度
        if featDims!=dims:
            print('Error: data dimensions ',dims,', does not match feature dimensions ', featDims)
        self.shape = (dims[0], dims[1], 1)

        #Number of distinct classes in data 數據中不同分類的數量(2類)
        self.num_classes = len(np.unique(trainLabels))

        #if validation feature,label files exist, read them
        #如果驗證集的特徵與標籤存在 ， 將它們讀取進來
        if(feature.validFeatFilesExist):
            with open(feature.validFeatureFile, 'rb') as f:
                self.valid_x=pickle.load(f)
            with open(feature.validLabelFile, 'rb') as f:
                self.valid_y=pickle.load(f)
            with open(feature.validMapFile, 'rb') as f:
                self.valid_patSegMaps=pickle.load(f)

        #if test feature,label files exist, read them
        #如果測試集的特徵與標籤存在 ， 將它們讀取進來
        if(feature.testFeatFilesExist):
            with open(feature.testFeatureFile, 'rb') as f:
                self.test_x=pickle.load(f)
            with open(feature.testLabelFile, 'rb') as f:
                self.test_y=pickle.load(f)
            with open(feature.testMapFile, 'rb') as f:
                self.test_patSegMaps=pickle.load(f)

        #if valid and test files do not exist, perform splitting of the trainSamples into three
        #如果驗證集和測試集不存在，則執行trainSamples 將資料集拆分為三個
        if ( not feature.testFeatFilesExist) and (not feature.validFeatFilesExist):
            #to be implemented if a database contains just the train features (no validation or test)/如果數據庫僅包含列車功能（無驗證或測試），則實施
            pass

        #if only train and test files exist, split train in two to get validation
        #如果只存在訓練集和測試集 ， 則將訓練集分成兩部分 (真.訓練集 與 真.驗證集)
        if (feature.testFeatFilesExist) and (not feature.validFeatFilesExist):
            print("Train data split into train - validation")
            #splitting data
            indexes=list(range(trainExamples.shape[0]))#phyton 3.5: range creates an object, in python 2.7 a list (P3.5 建立物件 / P2.7 建立清單)
            random.shuffle(indexes)
            splitRatio=self.splitRatios[0]# ratio of the train samples within the whole (分隔比例)
            self.numTrainSamp=int(trainExamples.shape[0]*splitRatio)
            self.numValidSamp=trainExamples.shape[0]-self.numTrainSamp
            trainIndexes=indexes[:self.numTrainSamp]
            validIndexes=indexes[self.numTrainSamp:]

            #type conversions applied to fit to Keras input specifications
            #將資料的類型做轉換 才能符合Keras的輸入格式
            self.train_x=trainExamples[trainIndexes];self.train_x = self.train_x.astype('float32')
            self.valid_x=trainExamples[validIndexes];self.valid_x = self.valid_x.astype('float32')

            self.train_y=trainLabels[trainIndexes];self.train_y=self.train_y.astype('uint8')
            self.valid_y=trainLabels[validIndexes];self.valid_y=self.valid_y.astype('uint8')

            # convert class vectors to binary class matrices 將類向量 轉換為 二進制類矩陣
            self.train_y = to_categorical(self.train_y, self.num_classes)
            self.valid_y = to_categorical(self.valid_y, self.num_classes)

        #if all exist, just copy trainSamples to train data without split
        #如果全部存在(訓練/驗證/測試)，只需複制trainSamples即可在不拆分的情況下訓練數據
        if (feature.testFeatFilesExist) and (feature.validFeatFilesExist):
            self.train_x=trainExamples
            self.train_y=trainLabels
            self.train_patSegMaps=train_patSegMaps

        return True#features computed, files created, features loaded
        #   return value is False if segmentation was not available and feature
        #   extraction could not be performed
        #如果分段不可用且無法執行特徵提取，則返回值為False

    def getWaveFilesAndLabels(self):
        '''Gathers file lists for train, valid and test
           讀取訓練、驗證、測試集的文件列表

        This function assumes the following presentation for the database:
            File lists for wave files are presented in "train.txt","validation.txt", "test.txt"
            Each list file contains a row which is composed of the filename and the label separated by a tab
            If the files do not exist, they will be created

            此函數假定數據庫的以下格式(檔案名稱)：
             音訊檔的文件列表顯示在“train.txt”，“validation.txt”，“test.txt”中
             每個列表文件都包含一行資料，是由'文件名'和'標籤'所組成
             如果文件不存在，將會透過Function來創建

        Assigns:
            self.trainFiles (list): List of wave files for the train set
            self.validFiles (list): List of wave files for the validation set
            self.testFiles (list): List of wave files for the test set

            self.trainFileLabels (list):labels for wave files for the train set
            self.validFileLabels (list):labels for wave files for the validation set
            self.testFileLabels (list):labels for wave files for the test set

        '''
        #先依照資料夾路徑 設定好個個資料集的檔案位置
        trainListFile=self.wavFolder+"train.txt"
        validListFile=self.wavFolder+"validation.txt"
        testListFile=self.wavFolder+"test.txt"
        #if list-files do not exist, attempt to create 如果檔案不存在，嘗試創建
        if ((not os.path.exists(trainListFile)) and (not os.path.exists(validListFile)) and (not os.path.exists(testListFile)) ):
            self._createFileLists(self.wavFolder)

        #If balanced data will be used, perform data augmentation using upsampling
        #如果將使用平衡數據，則使用上採樣執行數據增強
        if self.useBalanced and (not self.balancingPerformed):
            self._augmentForBalancing(trainListFile)#augmentation of train for making it balanced 增加訓練集 使其平衡
            self._augmentForBalancing(validListFile)#augmentation of validation for making it balanced 增加驗證集 使其平衡

        #Reading file-lists from files 從文件中讀取文件列表
        #讀取訓練集 文件列表
        if os.path.exists(trainListFile):
            fid_listFile=open(trainListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.trainFiles.append(os.path.join(self.wavFolder, name))
                self.trainFileLabels.append(label)
            fid_listFile.close()

        #讀取驗證集 文件列表
        if os.path.exists(validListFile):
            fid_listFile=open(validListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.validFiles.append(os.path.join(self.wavFolder, name))
                self.validFileLabels.append(label)
            fid_listFile.close()

        #讀取測試集 文件列表
        if os.path.exists(testListFile):
            fid_listFile=open(testListFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                self.testFiles.append(os.path.join(self.wavFolder, name))
                self.testFileLabels.append(label)
            fid_listFile.close()
            
        #找不到檔案
        if(len(self.trainFiles)==0):
            print("Error: train files could not be found in expected format. The database should either file list files or stored in specific subfolders: train, validation and test")
            
        #列出 各個資料集的資料量
#        text_file_predict = open("Number of files for all.txt", "w")#泓翔新增
#        text_file_predict.write('test result(predict)\n')#泓翔新增
#        text_file_predict.write('\n')#泓翔新增
#        text_file_predict.write('\nNumber of files for train(augmented), validation(augmented) and test: =\t'+str([len(self.trainFiles),len(self.validFiles),len(self.testFiles)]))#泓翔新增
#        text_file_predict.write('\n------------------------------------------\n')#泓翔新增
#        text_file_predict.close()#泓翔新增
        print('Number of files for train(augmented), validation(augmented) and test: ',[len(self.trainFiles),len(self.validFiles),len(self.testFiles)])

    def _createFileLists(self,dirName):
        '''File list creation

        1)If the directory(dirName) includes 'train', 'validation','test' directories
        file-lists will be created for files in these folders
        
        1)If the directory(dirName) includes 'test' and not 'validation' directories
        train will be splitted into train and validation using their relative sizes defined in split sizes

        3)If those subdirectories do not exist, a single voulume of data without splits will be assumed
        and random splitting will be performed to gather file-lists

        For each of the cases, file lists (together with class information) is
        written to train.txt, validation.txt and test.txt
        It would be a good practice to check their content to make sure splitting operation is correctly handled

        如果訓練、驗證、測試集的資料夾路徑是否存在，若無則進行資料夾的生成
        接著會依照你所設定之資料集，對不同的Data進行資料清單的設置

        進而生成train.txt，validation.txt和test.txt三個檔案
過程中會call _getLabel Function

        '''
        if self.name=='Physionet':
            '''For Physionet-2016 data, labels are stored in csv files
            First these label files are read and this information is further used in _getLabel
            對於Physionet-2016數據，標籤存儲在csv文件中
            首先讀取這些標籤文件，並在_getLabel中進一步使用此信息
            '''
            for root, dirs, files in os.walk(dirName):
                for file in files:
                    if file.lower().endswith('.csv'):
                        fileFullName=os.path.join(root, file)
                        with open(fileFullName) as f:
                            lines = f.readlines()
                        for line in lines:
                            tokens=line.strip().split(',')
                            val=int(tokens[1])
                            val=int((val+1)/2) #conversion from [-1,1](normal,abnormal) to [0,1] /從[-1,1]（正常，異常）轉換為[0,1]
                            self.preDefFileLabelMap[tokens[0]+'.wav']=val

        #Checking options/检查选项
        splitNames=['train','validation','test']
        folderExists={}
        for subFolder in splitNames:
            if os.path.exists(dirName+subFolder):
                folderExists[subFolder]=True
            else:
                folderExists[subFolder]=False
        
        #Testing option 1 (三個資料集皆存在)
        if(folderExists['train'] and folderExists['validation'] and folderExists['test']):
            for subFolder in splitNames:
                if os.path.exists(dirName+subFolder):
                    listFile=open(dirName+subFolder+'.txt','w')
                    for root, dirs, files in os.walk(dirName+subFolder):
                        for file in files:
                            if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                                fileFullName=os.path.join(root, file)
                                label=self._getLabel(root,file)
                                listFile.write('%s\t%s\n' % (fileFullName,label))
                    listFile.close()

        #Testing option 2 (Test資料集 存在)
        if(folderExists['test']):
            #Creating test set list file
            subFolder='test'
            if os.path.exists(dirName+subFolder):
                listFile=open(dirName+subFolder+'.txt','w')
                for root, dirs, files in os.walk(dirName+subFolder):
                    for file in files:
                        if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                            fileFullName=os.path.join(root, file)
                            label=self._getLabel(root,file)
                            listFile.write('%s\t%s\n' % (fileFullName,label))
                listFile.close()
            #Split ratio modified not to include a split for test when 'test' folder exists/當'test'文件夾存在時，拆分比率被修改為不包括用於測試的拆分
            self.splitRatios[0]=self.splitRatios[0]/(self.splitRatios[0]+self.splitRatios[1])
            self.splitRatios[1]=self.splitRatios[1]/(self.splitRatios[0]+self.splitRatios[1])
            self.splitRatios[2]=0

        #Testing option 3: (驗證集 不存在)
        if (not folderExists['validation']):
            #Gathering file-label list and applying random split such that 
            #   each set has similar distribution for each category
            #   ex: if split ratio for train is 0.7, 70 percent random selection
            #   will be performed from each label(normal, pathological)
            # 收集文件標籤列表並應用隨機拆分，使每個集合對每個類別具有相似的分佈
            # 例如：如果訓練集的比例為0.7，則會對每個標籤的70％進行隨機選擇（正常，病理）
            fileLabelDict={}
            labels=[]
            if(folderExists['train']):#if train sub-folder exists use that, if not use the whole directory 如果train子文件夾存在則使用，如果不使用整個目錄
                subDir='train'
            else:
                subDir=''
            for root, dirs, files in os.walk(dirName+subDir):
                for file in files:
                    if file.lower().endswith('.wav') and (not 'ecgChn' in file):
                        fileFullName=os.path.join(root, file)
                        label=self._getLabel(root,file)
                        fileLabelDict[fileFullName]=label
                        labels.append(label)
            #set targeted number of samples per label for train, validation and test sets 為訓練，驗證和測試集設置每個標籤的目標樣本數
            uniqueLabels, counts = np.unique(labels, return_counts=True)
            uniqueLabels=uniqueLabels.tolist()
            if len(self.splitRatios)>2:
                numTrain=[int(x*self.splitRatios[0]) for x in counts]
                numValid=[int(x*self.splitRatios[1]) for x in counts]
                numTest=[]
                for ind in range(len(numTrain)):
                    numTest.append(counts[ind]-(numTrain[ind]+numValid[ind]))
            else:
                print('!!! Random split for ratios with sizes other than 3 is not implemented yet')
            
            #APPLY SPLITS ON FILE LEVEL 在文件級別應用分割
            trainFilesDict={}
            while max(numTrain)>0:#add files until num2add becomes a zero-list /添加文件，直到num2add成為零列表
                #pick a random file, check if a copy is needed for that category /選擇一個隨機文件，檢查該類別是否需要副本
                randFile=random.sample(list(fileLabelDict),1)[0]
                #check if sample needed for that category /檢查該類別是否需要樣品
                label=fileLabelDict[randFile]
                if numTrain[uniqueLabels.index(label)]>0:#if a new file is needed for that category /如果該類別需要新文件
                    trainFilesDict[randFile]=label#add to train set /添加到訓練集
                    del fileLabelDict[randFile]#remove that file from all-files dictionary /從所有文件字典中刪除該文件
                    numTrain[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category /減少該類別所需的新文件數量
            
            validFilesDict={}
            while max(numValid)>0:#add files until num2add becomes a zero-list /添加文件，直到num2add成為零列表
                #pick a random file, check if a copy is needed for that category /選擇一個隨機文件，檢查該類別是否需要副本
                randFile=random.sample(list(fileLabelDict),1)[0]
                #check if sample needed for that category /檢查該類別是否需要樣品
                label=fileLabelDict[randFile]
                if numValid[uniqueLabels.index(label)]>0:#if a new file is needed for that category /如果該類別需要新文件
                    validFilesDict[randFile]=label#add to validation set /添加到驗證集
                    del fileLabelDict[randFile]#remove that file from all-files dictionary /從所有文件字典中刪除該文件
                    numValid[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category /減少該類別所需的新文件數量
            

            #creating train.txt file /創建train.txt文件
            listFile=open(dirName+'train.txt','w')
            for (fileName,label) in trainFilesDict.items():
                listFile.write('%s\t%s\n' % (fileName,label))
            listFile.close()

            #creating validation.txt file /創建validation.txt文件
            listFile=open(dirName+'validation.txt','w')
            for (fileName,label) in validFilesDict.items():
                listFile.write('%s\t%s\n' % (fileName,label))
            listFile.close()

            #creating test.txt file /創建test.txt文件
            if(self.splitRatios[2]>0):
                testFilesDict=fileLabelDict#rest is left to test set /休息留給測試集(?)
                listFile=open(dirName+'test.txt','w')
                for (fileName,label) in testFilesDict.items():
                    listFile.write('%s\t%s\n' % (fileName,label))
                listFile.close()


    def _getLabel(self,root,file):
        '''Finding label/class for a specific wave file
        Database specific:
            Uoc: the label/class information is available in the sub-folder name
                normal: 0, pathological: 1
                nomur:0, murmus:1, murpat:2
            Physionet: labeles are stored in csv files
            UrbanSound: the label/class information is available in the filename
            
         查找特定波形文件的標籤/類
         數據庫特色：
             Uoc：標籤/類信息在子文件夾名稱中可用
                  正常：0，有病：1
                  nomur：0，murmus：1，murpat：2
             Physionet：labeles存儲在csv文件中
             UrbanSound：標籤/類信息在文件名中可用
        '''
        label=''
        #get label (dba-specific) /獲取標籤（特定於dba）
        if self.name=='UrbanSound':
            #In UrbanSound dba, the label is coded in the file-name /在UrbanSound dba中，標籤以文件名編碼
            label = file.split('/')[-1].split('-')[1]
        elif self.name=='Physionet':
            return self.preDefFileLabelMap[file]
        else:
            print('Error: unknown dba-name:', self.name)
        return label

    def _augmentForBalancing(self,listFile):
        '''Augmenting the set defined by the listFile by upsampling wave files (%10-%x percent)

        Modifies the file list and creates new wav files with names *aug_*
        
        通過上採樣波形文件擴充listFile定義的集合（％10-％x ％）
        修改文件列表並創建名為* aug_ *的新wav文件

        '''
        #reading the file names and labels /讀取文件名和標籤
        allFiles=[]
        allFileLabels=[]
        if os.path.exists(listFile):
            fid_listFile=open(listFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                allFiles.append(os.path.join(self.wavFolder, name))
                allFileLabels.append(label)
            fid_listFile.close()

        #deduce number of files to be added to each category /推斷要添加到每個類別的文件數
        uniqueLabels, counts = np.unique(allFileLabels, return_counts=True)
        uniqueLabels=uniqueLabels.tolist()
        num2add=[max(counts)-x for x in counts]
        maxAugRatio=max(num2add)/min(counts)
        print('Data augmentation for balancing via upsampling: ',listFile.split('/')[-1])

        #creating list of new files and their labels /創建新文件及其標籤的列表
        files2add=[]
        files2addLabel=[]
        print('Number of files to add for each category:',num2add)
        while max(num2add)>0:#add files until num2add becomes a zero-list /添加文件，直到num2add成為零列表
            #pick a random file, check if a copy is needed for that category /選擇一個隨機文件，檢查該類別是否需要副本
            randInd=randint(0,len(allFiles)-1)
            #check if duplicate needed for that category /檢查該類別是否需要重複
            label=allFileLabels[randInd]
            if num2add[uniqueLabels.index(label)]>0:#if a new file is needed for that category /如果該類別需要新文件
                curFile=allFiles[randInd]
                #Decide random-modification percentage [should be two digits!] /決定隨機修改百分比[應該是兩位數！]
                #   change amount will be coded in filename as #aug /更改金額將以文件名編碼為#aug
                #   if the set will be augmented by 3, random values will be taken in range 10-16 /如果該組將增加3，則隨機值將在10-16範圍內
                modifPerc=randint(10,10+int(maxAugRatio*2))
                curFile=curFile.replace('.wav','_'+str(modifPerc)+'aug_.wav')
                #if this new file has not been added yet and it is not a file already
                #   created via resampling in a previous data augmentation[see _augmentTrain()], add
                #如果尚未添加此新文件 並且 它不是通過先前數據擴充中的重新採樣創建的文件 [請參閱_augmentTrain（）]，請添加
                if (not curFile in files2add) and ('resampAdd' not in curFile):
                    files2add.append(curFile)
                    files2addLabel.append(label)
                    num2add[uniqueLabels.index(label)]-=1#reduce number of new files needed for that category /減少該類別所需的新文件數量

        print(len(files2add),' files selected for up-sampling')

        #creation of new files and adding them in the file lists /創建新文件並將其添加到文件列表中
        for ind in range(len(files2add)):
            newPcgFile=files2add[ind]
            label=files2addLabel[ind]
            modifPerc=int(newPcgFile.split('aug_.wav')[0][-2:])#change amount coded in filename as #aug /將文件名中的數量更改為#aug
            orgPcgFile=newPcgFile.replace('_'+str(modifPerc)+'aug_.wav','.wav')
            #reading, resampling and writing to new files /讀取，重新採樣和寫入新文件
            data, samplerate = sf.read(orgPcgFile)
            data=resampy.resample(data, samplerate, int(samplerate*(1+modifPerc/100)), axis=-1)
            sf.write(newPcgFile, data, samplerate)#write as if sample rate has not been modified /寫，好像未修改採樣率

            #add files to the list /將文件添加到列表中
            allFiles.append(newPcgFile)
            allFileLabels.append(label)

        #over-write final list /覆蓋最終名單
        listFileFID=open(listFile,'w')
        for ind in range(len(allFiles)):
            fileFullName=allFiles[ind]
            label=allFileLabels[ind]
            listFileFID.write('%s\t%s\n' % (fileFullName,label))
        listFileFID.close()

        #re-check and report number of file for each category
        #reading the file names and labels
        #重新檢查並報告讀取文件名和標籤的每個類別的文件數
        allFiles=[]
        allFileLabels=[]
        if os.path.exists(listFile):
            fid_listFile=open(listFile,'r')
            for infile in fid_listFile:
                nameAndLabel=infile.split()
                name=nameAndLabel[0]
                label=nameAndLabel[1]
                allFiles.append(os.path.join(self.wavFolder, name))
                allFileLabels.append(label)
            fid_listFile.close()
        uniqueLabels, counts = np.unique(allFileLabels, return_counts=True)
        print('Number of files for each category in ',listFile.split('/')[-1],counts, ' after augmentation to balance')

