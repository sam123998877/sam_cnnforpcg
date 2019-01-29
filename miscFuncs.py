#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper function collection/波處理函數定義

Created on May 15th 2017

@author: Baris Bozkurt
"""
import os#os 模块提供了非常丰富的方法用来处理文件和目录
import numpy as np
import soundfile as sf#PySoundFile可以打開libsndfile支持的所有文件格式，例如WAV，FLAC，OGG和MAT文件。
from spectrum import create_window#SPECTRUM：Python中的光譜or頻譜分析

#%%

def cleanFilesOfPrevSess(dbaFolders):#只刪除具有特定擴展名的文件，並且接受的最短dba路徑為20個字符
    '''CLEANING intermediate files from previous test session

    !!!Use with precaution, you may end up deleting important files un-expectedly
    To avoid unexpected cases, only files with specific extensions are deleted and the
    shortest dba-path accepted is 20 characters (so that by error it does not get
    a root path and delete)

    Deletes all files with extension: h5, pkl, png and those
    end with results.txt

    Args:
        wavFolders (list of paths): Folders containing databases
    '''
    count=0 #count計數
    if type(dbaFolders) is list:
        for dbaFolder in dbaFolders:#for each database folder/為每個數據庫文件夾
            if os.path.exists(dbaFolder) and len(dbaFolder)>25:
                #walking through subfolders of the folder/瀏覽文件夾的子文件夾
                for root, dirs, files in os.walk(dbaFolder):#os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。
                    for file in files:#os.path.join(path1[,path2[,......]])返回值：将多个路径组合后返回
                        fileName=os.path.join(root, file)#creating filename with path/使用路徑創建文件名
                        if (file.endswith('_seg.pkl') or#endswith() 方法用于判断字符串是否以指定后缀结尾，如果以指定后缀结尾返回True，否则返回False。
                            file=='test.txt' or
                            file=='train.txt' or
                            file=='validation.txt' or
                            (file.endswith('.wav') and ('aug_' in file)) or
                            (file.endswith('.wav') and ('resampAdd_' in file)) or
                            (file.endswith('.pkl') and ('train' in file) and ('features' in root)) or
                            (file.endswith('.pkl') and ('valid' in file) and ('features' in root)) or
                            (file.endswith('.pkl') and ('test' in file) and ('features' in root))):
                            #print('Deleting: ',fileName)
                            os.remove(fileName)#删除路径为fileName的文件。
                            count+=1
            else:
                print('Path does not exist or too short(min expected len: 25 chars): ',dbaFolder)#路徑不存在或太短（最小預期len：25個字符
                print('!!! Check out !!! this process is destructive')# !!!退房！這個過程是破壞性的
        print('Number of files deleted by cleanFilesOfPrevSess: ',count)#清理文件流程刪除的文件數：
    else:
        print('Input should be a list of paths!...')#輸入應該是路徑列表

def findAllSubStrIndexes(mainStr,subStr):
    """Finding all occurences of a substring in a string(mainStr)/查找字符串中子字符串的所有出現（mainStr）
    """
    indexes=[];
    index=mainStr.find(subStr);#find() 方法检测字符串中是否包含子字符串 str ，检查是否包含在指定范围内，如果包含子字符串返回开始的索引值，否则返回-1。
    if index != -1:
        indexes.append(index)
    else:
        return []
    while index < len(mainStr):
        index = mainStr.find(subStr, index+1)
        if index == -1:
            break
        indexes.append(index)
    return indexes

def stringNotContainsListElems(mainStr,excludeList):
    """Given a string and a list /給定一個字符串，列表檢查字符串中是否包含列表元素
    checks if none of the list elements are included in the string"""
    for elem in excludeList:
        if elem in mainStr:
            return False
    return True

def findIndOfMatchElemInStr(strList,mainStr):
    """
    Find the index of the element in the string-list that has a match in the 
    mainStr/ 在字符串列表中查找具有匹配項的元素的索引mainStr
    """
    for ind in range(len(strList)):
        if strList[ind] in mainStr:
            return ind
    return -1#refers to element not found/指未找到的元素


def getTimeSegsFromWavFile(fileName,winSizeSample,hopSizeSample,FS):
    '''Windowing function to gather time segments from a wavefile
    in a numpy array.
    winSizeSample: window size in number of samples
    hopSizeSample: hop size in number of samples
    FS: expected sampling frequency for the wave files
    
    Reason for using sample sizes and a fixed sampling rate:
        It would be preferable to use seconds for sizes and set window sizes
        using the sampling frequency of each file. However, all time frames 
        from all wave files in the database is stacked in a single array which
        necessitates the same size for all time segments. Hence, number of
        samples is prefered and sampling frequency of each file is checked for
        making sure all files have the same sampling frequency
        
    Returns a numpy array containing all time segments
    '''
    timeSegArr=np.array([])
    tukeyWinR=0.2#a value in range [0,1] that specifies fade in-out region portion of the window/＃範圍[0,1]中的值，指定窗口的淡入區域部分
    winFunc=create_window(winSizeSample,'tukey',r=tukeyWinR)
    
    data, samplerate = sf.read(fileName)
    if(samplerate!=FS):
        print('Error: Sampling frequency mismatch:'+fileName)#/錯誤：採樣頻率不匹配：
        return timeSegArr
    
    for ind in range(0,len(data),hopSizeSample):
        if(ind+winSizeSample>len(data)):
            break
        segSig=data[ind:ind+winSizeSample]
        segSig=segSig*winFunc
        #adding the segment to the arrays to be returned/將段添加到要返回的數組中
        if ind==0:
            timeSegArr=np.hstack((timeSegArr, segSig))
        else:
            timeSegArr=np.vstack((timeSegArr, segSig))
    return timeSegArr
   


