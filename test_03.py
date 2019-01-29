from testcompute import compute
import time
import sched
import os

old_files=[]

while True:
#    schedule.run()        

    
    
    schedule = sched.scheduler(time.time, time.sleep) 
        
        
    # 指定要列出所有檔案的目錄           
    #mypath = 'C:/Users/Lab606B/Desktop/pcg_test/'
    mypath = 'C:/Users/user/Desktop/cnn/DataSpaceFoeFTP/Heart_Record/nxp/'
    # 取得所有檔案與子目錄名稱
    files = os.listdir(mypath)
    
    #old_files=old_files
    #new_files=interlist(old_files, files)
    
    
    
    # 以迴圈處理
    for f in files:
        if(f.endswith('.wav')):#塞選附檔名.wav檔
            # 產生檔案的絕對路徑
            fullpath = os.path.join(mypath, f)        
            # 判斷 fullpath 是檔案
            if os.path.isfile(fullpath):
                print("檔案：", f)
                print("soundfile exist")
                
                
                try:
                    old_files.index(f)
                    print("重複",f)
                except ValueError:
                    schedule.enter(1,0,compute,(mypath,f))#每隔1秒執行function
                    #schedule.every(10).seconds.do(compute,"mypath")#每隔1執行function
                    #schedule.run_pending()
                    schedule.run()
                    old_files.append(f)
                    
                   
            
        else:
           schedule.run()   
           
    time.sleep(10)#間隔10秒
    
