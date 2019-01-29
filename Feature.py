#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017
@author: Baris Bozkurt
"""
import os
import pickle
import soundfile as sf
import numpy as np
from scipy.signal import hilbert
from spectrum import create_window
from gammatone.filters import make_erb_filters,erb_filterbank,centre_freqs
from keras.utils import to_categorical

class Feature(object):
    ''' '''
    def __init__(self,name,dimensions,signalLevel,segmentation,involveDelta=False,window4frame='tukey'):
        '''Constructor

        Args:
            name (str): Name of the feature, ex: "mfcc", "pncc",...
                        所使用之特徵的名稱。 這將定義了根據不同的特徵 如何去計算。
                        例如："mfcc", "pncc" ...等

            dimensions (list): Time and frequency dimensions of the feature,
                ex: [48 16] refers to 48 time-segments and 16 freq.-bands
                    [48] refers to a single dimensional array
                      儲存時域與頻域的維度。
                例如：[48 16]指的是48個時段和16個頻段
                      [48]指的是單維陣列

            signalLevel (str): level for which feature is computed,
                should be "file" or "frame"
                計算特徵的級別，應為“文件 file”或“框架 frame”
                
            segmentation (Segmentation object): type of segmentation applied on wave
                files. Ex: Segmentation("ecg",periodSynch=True,sizeType="1period")
                                                應用於wave文件的分段類型。
                     例如：Segmentation("ecg",periodSynch=True,sizeType="1period")

            involveDelta (bool): involving delta features or not
                                 是否涉及delta特徵。

            window4frame (str): Window function to be applied to each frame before feature extraction
                                在特徵提取之前 應用於每個幀的窗口函數(Window function)。

        '''
        self.name=name
        self.signalLevel=signalLevel
        self.segmentation=segmentation
        self.dimensions=dimensions
        if not(name=='MFCC' or name=='MelSpec') and involveDelta:
            involveDelta=False
            print('Only MFFC and MelSpec can include delta coeffs... involveDelta disabled for '+name)
        self.involveDelta=involveDelta
        #flags for availability of feature files /功能文件可用性的標誌
        self.trainFeatFilesExist=False
        self.testFeatFilesExist=False
        self.validFeatFilesExist=False
        self.window4frame=window4frame#window function that will be applied to the frame before feature extraction /在特徵提取之前將應用於幀的窗口函數

    def checkFeatureFiles(self,folderName):
        '''Checking availability of feature files
        If the files are available, the features will not be re-computed

        Args:
            folderName (str): Folder name where feature files will be searched
        Assigns:
            self.trainFeatFilesExist, self.validFeatFilesExist, self.testFeatFilesExist
            
        檢查功能文件的可用性 -> 如果特徵檔案存在，則不會重新計算特徵
   
         - 輸入變數 : folderName (str)
         - 輸出變數 : self.trainFeatFilesExist, self.validFeatFilesExist, self.testFeatFilesExist

        '''
        self.trainFeatureFile=folderName+'trainFeatures_'+self.name+'.pkl'
        self.trainLabelFile=folderName+'trainLabels_'+self.name+'.pkl'
        self.trainMapFile=folderName+'trainFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.trainFeatureFile) and os.path.exists(self.trainLabelFile) and os.path.exists(self.trainMapFile):
            self.trainFeatFilesExist=True
        else:
            self.trainFeatFilesExist=False

        self.validFeatureFile=folderName+'validFeatures_'+self.name+'.pkl'
        self.validLabelFile=folderName+'validLabels_'+self.name+'.pkl'
        self.validMapFile=folderName+'validFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.validFeatureFile) and os.path.exists(self.validLabelFile) and os.path.exists(self.validMapFile):
            self.validFeatFilesExist=True
        else:
            self.validFeatFilesExist=False

        self.testFeatureFile=folderName+'testFeatures_'+self.name+'.pkl'
        self.testLabelFile=folderName+'testLabels_'+self.name+'.pkl'
        self.testMapFile=folderName+'testFrmFileMaps_'+self.name+'.pkl'
        if os.path.exists(self.testFeatureFile) and os.path.exists(self.testLabelFile) and os.path.exists(self.testMapFile):
            self.testFeatFilesExist=True
        else:
            self.testFeatFilesExist=False

    def compute(self,splitName,filelist,fileLabels):
        '''Feature computation for files in the filelist

        Args:
            splitName (str): 'train', 'valid' or 'test'
            filelist (list): List containing file names(fullpath)
            fileLabels (list): List of labels(of categories) for each file

        Writes the outputs to files:
            self.trainFeatureFile, self.trainLabelFile, self.trainMapFile, etc.

        文件列表中文件的特徵計算
         - 輸入變數 : splitName,filelist,fileLabels
         - 輸出變數 : splitName (str) -> ‘train’, ‘valid’ 或 'test'
                      filelist (list) -> 包含文件名的列表(fullpath)
                      fileLabels (list) -> 每個文件的標籤(或類別)列表
         - 輸出寫入檔案 :  self.trainFeatureFile, self.trainLabelFile, self.trainMapFile …等 
        '''

        timeDim=self.dimensions[0]
        freqDim=self.dimensions[1]
        if self.involveDelta:
            freqDim=freqDim*2

        #initializing feature vectors /初始化特徵向量
        allFeatures=np.zeros((1,timeDim,freqDim))
        allLabels=[]
        fileSegmentMap={}#map containing filename versus indexes of segments/features within all samples in this set /包含文件名的映射與此集合中所有樣本中的段/要素的索引
        globalInd = 0
        for ind in range(len(filelist)):#for each file /對於每個文件
            if(ind!=0 and ind%50==0):
                print('Number of files processed: ',ind)
            file=filelist[ind]
            label=fileLabels[ind]

            #segmentation: if file does not exist, create it by running segmentation /分段：如果文件不存在，請通過運行分段創建它
            segFile=file.replace('.wav','_'+self.segmentation.abbr+'_seg.pkl')
            if not os.path.exists(segFile):
                self.segmentation.extract(file)
                if len(self.segmentation.startsMs)==0:#if segmentation could not be performed skip the process /如果無法執行分段，則跳過該過程
                    continue
                #writing segmentation result as a Segmentation object /將分割結果寫為分段對象
                pickleProtocol=1#choosen for backward compatibility /選擇向後兼容
                with open(segFile, 'wb') as f:
                    pickle.dump(self.segmentation, f, pickleProtocol)

            #loading segmentation info from file /從文件加載分段信息
            with open(segFile, 'rb') as f:
                self.segmentation=pickle.load(f)

            #Frame-level feature extraction /幀級特徵提取
            # Extract signal segments and apply feature extraction on each /提取信號段並對每個段應用特徵提取
            # Gather the result in allFeatures /在allFeatures中收集結果

            if (not self.segmentation.sizeType=='wholefile'):
                sig, samplerate = sf.read(file)#read wave signal
                self.samplerate=samplerate

                #windowing using segmentation info and performing feature extraction /使用分段信息進行窗口化並執行特徵提取
                starts=[int(round(x*samplerate/1000)) for x in self.segmentation.startsMs]
                stops=[int(round(x*samplerate/1000)) for x in self.segmentation.stopsMs]
                for ind in range(len(starts)):
                    segment=sig[starts[ind]:stops[ind]]
                    #applying windowing function to the segment /將窗口函數應用於段
                    if self.window4frame=='tukey':#windowing with Tukey window, r=0.08 /用Tukey窗口加窗，r = 0.08
                        segment=segment*create_window(stops[ind]-starts[ind],'tukey',r=0.08)
                    #windowing with Hanning window: to suppress S1 in 1-period frame length cases /使用漢寧窗口進行窗口化：在1週期幀長度情況下抑制S1
                    elif self.window4frame=='hanning':
                        segment=segment*create_window(stops[ind]-starts[ind],'hanning')
                    if(np.max(segment)>0):#normalization /正規化
                        segment=segment/np.max(segment)
                    feature=self._computeSingleFrameFeature(segment)
                    #adding computed feature /添加計算特徵
                    if globalInd==0:#if this is the first feature assign it directly /如果這是第一個功能直接分配它
                        allFeatures[0]=feature
                    else:#add one more element in the feature vector and then assign /在特徵向量中添加一個元素，然後分配
                        allFeatures=np.vstack([allFeatures,np.zeros((1,timeDim,freqDim))])
                        allFeatures[globalInd]=feature
                    #adding segment to file-segment map /將段添加到文件段映射
                    if file in fileSegmentMap:#if file already exists, append segment /如果文件已存在，則追加段
                        val=fileSegmentMap[file]
                        val.append(globalInd)
                        fileSegmentMap[file]=val
                    else:#file does not exist in map, add the first file-segment map /文件在地圖中不存在，添加第一個文件段映射
                        fileSegmentMap[file]=[globalInd]
                    allLabels.append(label)
                    globalInd+=1
            else:#File-level feature extraction /文件級特徵提取
                pass

        #If no data is read/computed at this point skip the rest [this happens when ecg signal is not available and segmentation on ecg was targeted for a test]
        #如果此時沒有讀取/計算數據，則跳過其餘的[當ecg信號不可用時會發生這種情況，並且ecg上的分段是測試的目標]
        if len(allLabels)==0:
            return

        #re-formatting feature vectors /重新格式化特徵向量
        allFeatures=allFeatures.reshape(allFeatures.shape[0],timeDim,freqDim,1)
        allLabels=np.array(allLabels,dtype = np.int)
        allLabels = to_categorical(allLabels)

        #Writing to files /寫入文件
        if splitName=='train':
            featureFile=self.trainFeatureFile
            labelFile=self.trainLabelFile
            mapFile=self.trainMapFile
        elif splitName=='valid':
            featureFile=self.validFeatureFile
            labelFile=self.validLabelFile
            mapFile=self.validMapFile
        elif splitName=='test':
            featureFile=self.testFeatureFile
            labelFile=self.testLabelFile
            mapFile=self.testMapFile
        else:
            print('Error: split-name should be train, valid or test')
        #saving features, labels and maps to pickle files /將功能，標籤和地圖保存到pickle文件
        pickleProtocol=1#choosen for backward compatibility /選擇向後兼容
        with open(featureFile, 'wb') as f:
            pickle.dump(allFeatures, f, pickleProtocol)
        with open(labelFile , 'wb') as f:
            pickle.dump(allLabels, f, pickleProtocol)
        with open(mapFile, 'wb') as f:
            pickle.dump(fileSegmentMap, f, pickleProtocol)
        print('--- ',splitName,' features computed')

    def _computeSingleFrameFeature(self,sig):
        '''Feature computation for a single time-series frame/segment

        Args:
            sig (numpy array): The signal segment for which feature will be computed
        Returns:
            feature (numpy array): Computed feature vector
            
        單個時間序列幀/段的特徵計算 
(只限 ”SubEnv” 子帶包絡(Sub-band envelopes)特徵計算)

         - 輸入變數 :  sig (numpy array)
         - 輸出變數 : feature (numpy array)
        '''


        if self.name=='SubEnv':
            '''Sub-band envelopes feature computation 子帶包絡特徵計算'''
            #Computing sub-band signals /計算子帶信號
            timeRes=self.dimensions[0]
            numBands=self.dimensions[1]
            low_cut_off=2#lower cut off frequency = 2Hz /較低的截止頻率= 2Hz
            centre_freqVals = centre_freqs(self.samplerate,numBands,low_cut_off)
            fcoefs = make_erb_filters(self.samplerate, centre_freqVals, width=1.0)
            y = erb_filterbank(sig, fcoefs)

            subenv = np.array([]).reshape(timeRes,0)
            for i in range(numBands):
                subBandSig=y[i,:]
                analytic_signal = hilbert(subBandSig)
                amp_env = np.abs(analytic_signal)
                np.nan_to_num(amp_env)
                #amp_env=resampy.resample(amp_env, len(amp_env), timeRes, axis=-1)#resampy library used resampling /resampy庫使用重新取樣
                #resampling may lead to unexpected computation errors, /重新採樣可能會導致意外的計算錯誤，
                #I prefered average amplitudes for short-time windows /我更喜歡短時間窗口的平均幅度
                downSampEnv=np.zeros((timeRes,1))
                winSize=int(len(amp_env)/timeRes)
                for ind in range(timeRes):
                    downSampEnv[ind]=np.log2(np.mean(amp_env[ind*winSize:(ind+1)*winSize]))
                subenv=np.hstack([subenv,downSampEnv])
            #removing mean and normalizing /刪除均值和正常化
            subenv=subenv-np.mean(subenv)
            subenv=subenv/(np.max(np.abs(subenv)))
            feature=subenv
        else:
            print('Error: feature '+self.name+' is not recognized')
            feature=[]

        return feature
