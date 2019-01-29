#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running repeated experiments for the system with settings:
    Sub-band envelopes computed from async. frames extracted with 
    length of 2 seconds or 3 seconds, with a hop size of 1 second.
    Feature dimensions:
        Time: 32 or 64
        Frequency: 16
    Models:
        'uocSeq1' or 'uocSeq2' (see models.py)

    This script tests a total of 8 systems settings, 
    repeats the tests 5 times and writes results in the data/results folder

The data balancing operation involves creation of new samples. This will add
    new files in your database folder. Files named *aug_ are created that way. 
    You can see a complete list of files created by checking the train.txt and 
    validation.txt file-list files. New files' names are placed at the end of 
    each list. cleanFilesOfPrevSess function can be used to delete these files
    at the end of the tests
    
This script has been tested on Physionet 2016 data accessible from:
https://www.physionet.org/physiobank/database/challenge/2016/training.zip 
Example resulting files are available in the 'exampleResults' folder
      
The script first creates a 'data' folder, downloads Physionet data and unzips 
into that folder. The original validation package of Physionet is used as
test data and the train package is split into train and validation.

To replicate the tests, the user simply needs to run this script. The test 
results are written to data/results


"""
import os #处理文件和目录
import urllib.request #urllib可以存取網頁、下載資料、剖析資料、修改表頭(header)、執行GET與POST的請求…。
import zipfile#用来做zip格式编码的压缩和解压缩的，由于是很常见的zip格式，所以这个模块使用频率也是比较高。
#ZipFile是主要的类，用来创建和读取zip文件
#os.chdir(os.path.dirname(__file__))原本就存在的   /os.chdir(path)改变当前工作目录 os.path.dirname(__file__)返回脚本的路径
localRepoDir=os.getcwd() #返回当前工作目录

#%%數據庫下載
dataFolder=localRepoDir+'/PhysionetData/'
if not os.path.exists(dataFolder):#os.path.exists(路徑)如果只要檢查輸入的路徑是否存在, 不論是檔案或目錄都會回傳 TRUE
    os.mkdir(dataFolder);#os.mkdir(path[, mode])以数字mode的mode创建一个名为path的文件夹./创建一个名为dataFolder的文件夹
dbaName='Physionet'
urls={}
urls['train']='https://www.physionet.org/physiobank/database/challenge/2016/training.zip'
urls['test']='https://www.physionet.org/physiobank/database/challenge/2016/validation.zip'

#你下載過Physionet列車數據
print('Downloading Physionet data (200Mb) as zip files ... ')
print('Validation dataset will be used as the Test set and train dataset will be splitted to obtain the Train and Validation sets')
print('After the download, possible duplicates(in test and train) are checked and removed')
for dataCategory in urls.keys():
    targetDir=dataFolder+'/'+dataCategory
    if not os.path.exists(targetDir):
        os.mkdir(targetDir);
        url=urls[dataCategory]
        filename=url.split('/')[-1]
        #Downloading the zip file from the url從網址下載zip文件
        print('Downloading ',filename)
        urllib.request.urlretrieve(url,filename)
        #Unzipping to a specific folder解壓縮到特定文件夾
        zip_ref = zipfile.ZipFile(filename, 'r')
        zip_ref.extractall(targetDir)
        zip_ref.close()
        os.remove(filename)#Removing the zip file/ os.remove(filename)删除路径为filename的文件
        print('Data downloaded and unzipped to: ',targetDir)
    else:
        print('Folder ',targetDir,' already exists, delete it if you want to re-download data')

#I have observed later that validation data of Physionet is in fact a subset of the training set
# so, here is the code to remove duplicates 
# !! we will assume files having the same name and size are identical

#Collect test file list together with size information/收集測試文件列表以及大小信息
testDataFolder=dataFolder+'/test/'
testFileSizes={}
for root, dirs, files in os.walk(testDataFolder):#os.walk输出在文件夹中的文件名通过在树中游走，向上或者向下
    for file in files:
        if file.endswith('.wav') or file.endswith('.WAV'):#Python endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False
            testFileSizes[file]=os.stat(os.path.join(root, file)).st_size
            #os.stat(path)获取path指定的路径的信息/os.path.join(path1[,path2[,......]])返回值：将多个路径组合后返回
#Remove duplicates in the train folder checking file name and size
#刪除列車文件夾中的重複項，檢查文件名和大小            
trainDataFolder=dataFolder+'/train/'
for root, dirs, files in os.walk(trainDataFolder):  
    for file in files:
        if file in testFileSizes:
            fileSize=os.stat(os.path.join(root, file)).st_size
            if(fileSize == testFileSizes[file]):
                #print('File to be deleted: ', file, ' size: ',fileSize)
                os.remove(os.path.join(root, file))#os.remove(filename)删除路径为filename的文件
                #If exists, remove also the label file如果存在，也刪除標籤文件
                labelFile=os.path.join(root, file.replace('.wav','.hea'))
                if os.path.exists(labelFile):#os.path.exists()如果只要檢查輸入的路徑是否存在, 不論是檔案或目錄都會回傳 TRUE
                    os.remove(labelFile)

#The results will be put in 'results' folder in the database folder
#結果將放在數據庫文件夾中的'results'文件夾中
resultsFolder=dataFolder+"/results/"
if not os.path.exists(resultsFolder):#create folder if not exists /create folder如果不存在
    os.mkdir(resultsFolder)#os.mkdir(path)他的功能是一级一级的创建目录，前提是前面的目录已存在，如果不存在会报异常
#The features will be saved in 'features' folder in the database folder這些功能將保存在數據庫文件夾的“features”文件夾中
featureFolder=dataFolder+"/features/"
if not os.path.exists(featureFolder):#create folder if not exists
    os.mkdir(featureFolder)

#%%TEST specification TEST規範
from Segmentation import Segmentation #是副程式
from Feature import Feature#是副程式
from Data import Data#是副程式
from Test import Test#是副程式
from miscFuncs import cleanFilesOfPrevSess#是副程式


info="Single-test repeated experiment"#any info about the test/有關測試的任何信息
#Setting train-validation-test split ratios, 設置列車驗證 - 測試分流比
# if 'test' exists as a static folder, the first two values are used 
# and rest of the data is split taking into account their ratio
#如果'test'作為靜態文件夾存在，則使用前兩個值，並在考慮其比率的情況下拆分其餘數據
splitRatios=[0.65,0.15,0.20]

#This flag defines if the same number of samples will be used for each class in training此標誌定義在訓練中是否將為每個類使用相同數量的樣本
#   If True, data augmentation(via up-sampling some existing files) will be carried
#   to have balanced set for training and validation. Not applicable to test files
#如果為True，則將進行數據增強（通過對一些現有文件進行上採樣）以獲得用於訓練和驗證的平衡集。不適用於測試文件

useBalancedData=True

#Define segmentation strategy /定義細分策略
async2secSegments=Segmentation("None",periodSync=False,sizeType="fixed",frameSizeMs=2000.0,hopSizeMs=1000.0)
async3secSegments=Segmentation("None",periodSync=False,sizeType="fixed",frameSizeMs=3000.0,hopSizeMs=1000.0)
segStrategies=[async2secSegments,async3secSegments]

#Define features to be used /定義要使用的功能
features=[]
for featName in ['SubEnv']:#other options: 'MFCC','MelSpec'/其他選項：'MFCC'，'MelSpec'
    for segType in segStrategies:
        for timeDim in [32,64]:
            for freqDim in [16]:
                features.append(Feature(featName,[timeDim,freqDim],"frame",segType,involveDelta=False))


#Define data specifications for this database/定義此數據庫的數據規範  
data=Data(dbaName,dataFolder,featureFolder,features,useBalancedData,splitRatios,info)
#Defining NN model with a name. /用名稱定義NN模型。
#   Implementation is in models.py. Feel free to add your own models and /實現在models.py中。
#   test by just changing the name here/您可以在此處更改名稱，隨意添加自己的模型並進行測試
modelNames=['uocSeq1','uocSeq2']

#Running random split and testing several times (1/testSetPercentage)/運行隨機拆分和測試幾次（1 / testSetPercentage）
# ex: if test set is 20%, tests will be repeated 5 times/例如：如果測試組為20％，測試將被重複5次
numExperiments=int(1/splitRatios[-1])
for i in range(numExperiments):
    for modelName in modelNames:
        #Define test specifications and run/定義測試規範並運行
        singleTest=Test(modelName,data,resultsFolder,batch_size=128,num_epochs=50)
        #Run the tests: outputs will be put in the results folder/運行測試：輸出將被放入結果文件夾中
        singleTest.run()
        #Cleaning this test sessions' intermediate files/清理此測試會話的中間文件
    cleanFilesOfPrevSess([dataFolder])
