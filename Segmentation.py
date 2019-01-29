#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 15th 2017

@author: Baris Bozkurt
"""
import soundfile as sf#PySoundFile可以打開libsndfile支持的所有文件格式，例如WAV，FLAC，OGG和MAT文件。
import numpy as np

class Segmentation(object):#分割
    def __init__(self,sourceType="pcg",periodSync=True,sizeType="1period",frameSizeMs=1000.0,hopSizeMs=500.0):
        '''Constructor
         egmentation分割,period-synchronous 週期同步,fixed固定,hop size跳躍大小
        Args:
            source (str): source on which segmentation is computed from, /計算分段的源，
                ex: "ecg", "pcg"
            periodSync (bool): specifies if windows are period sync /指定窗口是否為周期同步
            sizeType (str): "1period","2periods", "3periods", "fixed"
            frameSizeMs (float): size of frame/window in milliseconds /幀/窗口的大小（以毫秒為單位） [not needed for "1period","2periods", "3periods"]
            hopSizeMs (float): hop-size/window-shift in milliseconds /以毫秒為單位的hop-size / window-shift [not needed for "1period","2periods", "3periods"]

        An abbreviation for segmentation is created and used in file names to be
            able to store results of different tests in the same folder
            /創建分段的縮寫並在文件名中使用，以便能夠在同一文件夾中存儲不同測試的結果
        '''
        self.sourceType=sourceType.lower()#輸入值
        self.periodSync=periodSync#輸入值
        self.frameSizeMs=frameSizeMs#輸入值
        self.frameSizeSamp=-1#not assigned/未分配  
        self.hopSizeMs=hopSizeMs#輸入值
        self.sizeType=sizeType.lower()#輸入值  lower()代表大寫會變小寫
        self.wavfile=""
        self._createAbbreviation()#abbreviation for the segmentation will be used in file/folder-naming/ abbreviation for segmentation將用於文件/文件夾命名
        self.sourceFilesExists=True#initiate with assumption that source for segmentation(i.e Ecg channel) exists, modified if not found
                                   #在假設存在分段源（即Ecg通道）的情況下啟動，如果未找到則修改
    def _createAbbreviation(self):#創建縮寫
        '''Creating the abbreviation string for the specific segmentation
        which will be used in file naming /為將在文件命名中使用的特定分段創建縮寫字符串
        '''
        self.abbr=""
        self.abbr+=self.sourceType[0]
        if self.periodSync:#猜測同步
            self.abbr+='Syn'
        else:
            self.abbr+='ASyn'
        if self.sizeType=="fixed":#猜測固定
            self.abbr+=str(int(self.frameSizeMs))+'len'
            if not self.periodSync:
                self.abbr+=('_'+str(int(self.hopSizeMs)))+'hop'
        else:
            self.abbr+=self.sizeType[0:4]


    def extract(self,wavfile):#提取
        '''Segmentation extraction /分段提取

        Args: /wavfile（str）：將執行分段的源信號
            wavfile (str): source signal from which segmentation will be performed

        Assigns: /分派
            self.startsMs, self.stopsMs, self.startsSamp, self.stopsSamp

        Note: /在Feature.compute中執行分段保存。細分對象直接保存在pickle文件中
            Saving of segmentation is performed in Feature.compute. The segmentation
            object is directly saved in a pickle file
        '''
        self.startsMs=[]
        self.stopsMs=[]
        self.periodMarksSec=[]

        self.wavfile=wavfile
        signal, samplerate = sf.read(wavfile)
        lenSigSamp=len(signal)
        lenSigMs=1000*lenSigSamp/samplerate
        self.lenSigMs=lenSigMs
        self.frameSizeSamp=int(samplerate*self.frameSizeMs/1000)#framesizesample=(取樣頻率*每一個音框所對應的時間)/10000

        if self.periodSync:#period-sync segmentation--------------------/週期同步分割
            print('Period synchronous segmentation disabled for this implementation')#為此實現禁用周期同步分段
            return
        else:#period-async segmentation--------------------/期間異步分割
            if self.sizeType=='fixed':
                '''Constant frame size windowing不變的幀大小窗口'''
                self.startsMs=list(np.arange(0,lenSigMs-self.frameSizeMs,self.hopSizeMs))
                self.stopsMs=[x+self.frameSizeMs for x in self.startsMs]


