#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running repeated experiments for the system with settings:
    Sub-band envelopes computed from async. frames extracted with 
    length of 2 seconds and hop size of 1 second. 

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

Created on May 15th 2017

@author: Baris Bozkurt
"""
import os#处理文件和目录
import urllib.request #urllib可以存取網頁、下載資料、剖析資料、修改表頭(header)、執行GET與POST的請求…。
import zipfile#用来做zip格式编码的压缩和解压缩的，由于是很常见的zip格式，所以这个模块使用频率也是比较高。
#ZipFile是主要的类，用来创建和读取zip文件

#os.chdir(os.path.dirname(__file__)) 原本就存在的   /os.chdir(path)改变当前工作目录 os.path.dirname(__file__)返回脚本的路径
localRepoDir=os.getcwd()#返回当前工作目录

#%%Database download
dataFolder=localRepoDir+'/data/'
if not os.path.exists(dataFolder):#os.path.exists(路徑)如果只要檢查輸入的路徑是否存在, 不論是檔案或目錄都會回傳 TRUE
    os.mkdir(dataFolder);#os.mkdir(path[, mode])以数字mode的mode创建一个名为path的文件夹.
dbaName='Physionet'
urls={}
urls['train']='https://www.physionet.org/physiobank/database/challenge/2016/training.zip'
urls['test']='https://www.physionet.org/physiobank/database/challenge/2016/validation.zip'

#Dowloading Physionet train data /你下載過Physionet列車數據
print('Downloading Physionet data (200Mb) as zip files ... ')#下載Physionet數據（200Mb）作為zip文件...
for dataCategory in urls.keys():#空字典.keys() 取得以建元素的組合
    targetDir=dataFolder+'/'+dataCategory
    if not os.path.exists(targetDir):#os.path.exists(路徑)如果只要檢查輸入的路徑是否存在, 不論是檔案或目錄都會回傳 TRUE
        os.mkdir(targetDir);#os.mkdir(targetDir)创建一个名为targetDir的文件夹.
        url=urls[dataCategory]
        filename=url.split('/')[-1]#split()指分割 '/'符號,並取序列-1的項 
        #Downloading the zip file from the url /從網址下載zip文件
        print('Downloading ',filename)
        urllib.request.urlretrieve(url,filename)#将URL表示的网络对象复制到本地文件,url：外部或者本地url,filename：指定了保存到本地的路径（如果未指定该参数，urllib会生成一个临时文件来保存数据）
        #Unzipping to a specific folder/解壓縮到特定文件夾
        zip_ref = zipfile.ZipFile(filename, 'r')#用来创建和读取zip文件,'r'表示打开一个存在的只读ZIP文件
        zip_ref.extractall(targetDir)#zipfile.extractall(path) path解压缩目录，
        zip_ref.close()#写入的任何文件在关闭之前不会真正写入磁盘。 
        os.remove(filename)#Removing the zip file/os.remove(filename)删除路径为filename的文件
        print('Data downloaded and unzipped to: ',targetDir)
    else:
        print('Folder ',targetDir,' already exists, delete it if you want to re-download data')

#The results will be put in 'results' folder in the database folder/结果将放在数据库文件夹中的“results”文件夹中
resultsFolder=dataFolder+"/results/"
if not os.path.exists(resultsFolder):#create folder if not exists/如果不存在则创建文件夹
    os.mkdir(resultsFolder)#创建一个名为resultsFolder的文件夹.
#The features will be saved in 'features' folder in the database folder/这些功能将保存在数据库文件夹的“features”文件夹中
featureFolder=dataFolder+"/features/"
if not os.path.exists(featureFolder):#create folder if not exists
    os.mkdir(featureFolder)

#%%TEST specification/测试规范
from Segmentation import Segmentation#packege
from Feature import Feature#packege
from Data import Data#packege
from Test import Test#packege
#from miscFuncs import cleanFilesOfPrevSess#packege#泓翔刪除

#Single-test repeated experiment/單次測試重複實驗
info="Single-test repeated experiment"#any info about the test/有關測試的任何信息
#Setting train-validation-test split ratios, /設置訓練驗證 - 測試分流比，
# if 'test' exists as a static folder, the first two values are used 
# and rest of the data is split taking into account their ratio/如果'test'作为静态文件夹存在，则使用前两个值，并在考虑其比率的情况下拆分其余数据
splitRatios=[0.65,0.15,0.20]#splitRatios/分数比率

#This flag defines if the same number of samples will be used for each class in training/此標誌定義在訓練中是否將為每個類使用相同數量的樣本
#   If True, data augmentation(via up-sampling some existing files) will be carried/如果為True，將進行數據增強（通過對一些現有文件進行上採樣）
#   to have balanced set for training and validation. Not applicable to test files/具有平衡的訓練和驗證設置。不適用於測試文件
useBalancedData=True #使用平衡数据

#Define segmentation strategy/定義細分策略
async2secSegments=Segmentation("None",periodSync=False,sizeType="fixed",frameSizeMs=2000.0,hopSizeMs=1000.0)
#async 2sec Segments/異步2秒段,Segmentation分割,period-synchronous 週期同步,fixed固定,hop size跳躍大小
#Define features to be used/定義要使用的功能
features=[]#use of multiple features is possible, that's why we use a list here/可以使用多個功能，這就是我們在這裡使用列表的原因
timeDim=32#猜測time dimension/時間分辨率
freqDim=16#猜測frequency dimensions/频段
#This implementation uses only the sub-band envelopes feature
# other features can be appended here/此實現僅使用子帶包絡功能，此處可附加其他功能
features.append(Feature('SubEnv',[timeDim,freqDim],"frame",async2secSegments))

#Define data specifications for this database/定義此數據庫的數據規範
data=Data(dbaName,dataFolder,featureFolder,features,useBalancedData,splitRatios,info)
#Defining NN model with a name. /用名稱定義NN模型。
#   Implementation is in models.py./實現在models.py中。 Feel free to add your own models and 
#   test by just changing the name here/ 您可以在此處更改名稱，隨意添加自己的模型並進行測試
modelName='uocSeq1'

#Running random split and testing several times (1/testSetPercentage)/運行隨機拆分和測試幾次（1 / testSetPercentage）
# ex: if test set is 20%, tests will be repeated 5 times/例如：如果測試組為20％，測試將被重複5次
#numExperiments=int(1/splitRatios[-1])#泓翔刪除
#for i in range(numExperiments):#泓翔刪除
    #Define test specifications and run/定義測試規範並運行
singleTest=Test(modelName,data,resultsFolder,batch_size=128,num_epochs=50)
    #Run the tests: outputs will be put in the results folder/運行測試：輸出將被放入結果文件夾中
singleTest.run()
    #Cleaning this test sessions' intermediate files/清理此測試會話的中間文件
#cleanFilesOfPrevSess([dataFolder])#泓翔刪除
