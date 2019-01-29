# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 13:43:15 2018

@author: LabSam
"""
#from Feature import *
from spectrum import create_window
from gammatone.filters import make_erb_filters,erb_filterbank,centre_freqs
from scipy.signal import hilbert
#from keras.utils import to_categorical
#import pickle
import numpy as np
import soundfile as sf
import keras
#import os

#def ListFilesToTxt(dir,file,wildcard,recursion):
#    exts = wildcard.split(" ")
#    files = os.listdir(dir)
#    for name in files:
#        fullname=os.path.join(dir,name)
#        if(os.path.isdir(fullname) & recursion):
#            ListFilesToTxt(fullname,file,wildcard,recursion)
#        else:
#            for ext in exts:
#                if(name.endswith(ext)):
#                    file.write(name + "\n")
#                    break

def compute(filepath,file):
    modelpath='C:/Users/user/Desktop/cnn/data/model/M_uocSeq1SubEnv32by16_nASyn2000len_1000hopt.h5'
    #dir='C:/Users/Lab606B/Desktop/result/'#txt 儲存路徑
    #wildcard="txt"
#    fileLabels=['1']
    timeDim=32
    freqDim=16
    frameSizeMs=2000
    hopSizeMs=1000
    signal, samplerate = sf.read(filepath+file)
    lenSigSamp=len(signal)
    lenSigMs=1000*lenSigSamp/samplerate
    lenSigMs=lenSigMs

    startsMs=list(np.arange(0,lenSigMs-frameSizeMs,hopSizeMs))
    stopsMs=[x+frameSizeMs for x in startsMs]

    #windowing using segmentation info and performing feature extraction /使用分段信息進行窗口化並執行特徵提取
    starts=[int(round(x*samplerate/1000)) for x in startsMs]
    stops=[int(round(x*samplerate/1000)) for x in stopsMs]
    
    globalInd = 0
    allFeatures=np.zeros((1,timeDim,freqDim))
#    allLabels=[]
    #(無用) fileSegmentMap={}#map containing filename versus indexes of segments/features within all samples in this set /包含文件名的映射與此集合中所有樣本中的段/要素的索引
    for ind in range(len(starts)):
        segment=signal[starts[ind]:stops[ind]]
        #applying windowing function to the segment /將窗口函數應用於段
        segment=segment*create_window(stops[ind]-starts[ind],'tukey',r=0.08)
    
        if(np.max(segment)>0):#normalization /正規化
            segment=segment/np.max(segment)
            
            #feature=Feature._computeSingleFrameFeature(segment)
            '''Sub-band envelopes feature computation 子帶包絡特徵計算'''
            #Computing sub-band signals /計算子帶信號
            low_cut_off=2#lower cut off frequency = 2Hz /較低的截止頻率= 2Hz
            centre_freqVals = centre_freqs(samplerate,freqDim,low_cut_off)
            fcoefs = make_erb_filters(samplerate, centre_freqVals, width=1.0)
            y = erb_filterbank(segment, fcoefs)
    
            subenv = np.array([]).reshape(timeDim,0)
            for i in range(freqDim):
                subBandSig=y[i,:]
                analytic_signal = hilbert(subBandSig)
                amp_env = np.abs(analytic_signal)
                np.nan_to_num(amp_env)
                #amp_env=resampy.resample(amp_env, len(amp_env), timeRes(timeDim), axis=-1)#resampy library used resampling /resampy庫使用重新取樣
                #resampling may lead to unexpected computation errors, /重新採樣可能會導致意外的計算錯誤，
                #I prefered average amplitudes for short-time windows /我更喜歡短時間窗口的平均幅度
                downSampEnv=np.zeros((timeDim,1))
                winSize=int(len(amp_env)/timeDim)
                for ind in range(timeDim):
                    downSampEnv[ind]=np.log2(np.mean(amp_env[ind*winSize:(ind+1)*winSize]))
                subenv=np.hstack([subenv,downSampEnv])
            #removing mean and normalizing /刪除均值和正常化
            subenv=subenv-np.mean(subenv)
            subenv=subenv/(np.max(np.abs(subenv)))
            feature=subenv
            
        #adding computed feature /添加計算特徵
        if globalInd==0:#if this is the first feature assign it directly /如果這是第一個功能直接分配它
            allFeatures[0]=feature
        else:#add one more element in the feature vector and then assign /在特徵向量中添加一個元素，然後分配
            allFeatures=np.vstack([allFeatures,np.zeros((1,timeDim,freqDim))])
            allFeatures[globalInd]=feature
        
    #(無用)    #adding segment to file-segment map /將段添加到文件段映射
    #(無用)    if file in fileSegmentMap:#if file already exists, append segment /如果文件已存在，則追加段
    #(無用)        val=fileSegmentMap[file]
    #(無用)        val.append(globalInd)
    #(無用)        fileSegmentMap[file]=val
    #(無用)    else:#file does not exist in map, add the first file-segment map /文件在地圖中不存在，添加第一個文件段映射
    #(無用)        fileSegmentMap[file]=[globalInd]
    #(無用)    allLabels.append(fileLabels)
        globalInd+=1
        
    #(無用) allFeatures=allFeatures.reshape(allFeatures.shape[0],timeRes,numBands,1)
    #(無用) allLabels=np.array(allLabels,dtype = np.int)
    #(無用) allLabels = to_categorical(allLabels)
    allFeatures = np.reshape(allFeatures,[len(allFeatures),timeDim,freqDim,1])
        
    #(無用) with open(filepath+'Test_Features.pkl', 'wb') as f:
    #(無用)    pickle.dump(allFeatures, f, 1)
    #(無用) with open(filepath+'Test_Labels.pkl' , 'wb') as f:
    #(無用)    pickle.dump(allLabels, f, 1)
    #(無用) with open(filepath+'Test_Map.pkl', 'wb') as f:
    #(無用)    pickle.dump(fileSegmentMap, f, 1)
    
    model = keras.models.load_model(modelpath)
    
    y_probs=model.predict(allFeatures, batch_size=allFeatures.shape[0], verbose=0)
    
    #normal = -1 = 0 ; abnormal = 1
    normal=0
    abnormal=0
    for i in range(len(y_probs)):
        if(y_probs[i,0]>y_probs[i,1]):
            normal = normal + 1
        else:
            abnormal = abnormal + 1
        
    if(normal > abnormal):
        result = 'normal' 
        resultRate = normal / len(y_probs) * 100
    elif(normal < abnormal):    
        result = 'abnormal'
        resultRate = abnormal / len(y_probs) * 100
    else:
        result = 'not sure'
        resultRate = 50


    #建立txt檔
    text_file_predict = open('C:/Users/user/Desktop/cnn/DataSpaceFoeFTP/Predict_Result/nxp/' + file.replace('.wav','')+".txt", "w",encoding='utf-8')
    #text_file_predict.write('test result(predict)\n')
    text_file_predict.write('檔案:'+str(file))
    text_file_predict.write('\n')
    text_file_predict.write('\n診斷結果 =\t'+str(result))
    text_file_predict.write('\n概率為 =\t'+str(resultRate)+'%')
    text_file_predict.write('\n------------------------------------------\n')
 #   ListFilesToTxt(dir,file,wildcard, 1)
    text_file_predict.close()
    
    print('診斷結果為 : ',result)
    print('機率為 : ',resultRate , '%')
    
    