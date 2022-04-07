from pnn import *
from sklearn.model_selection import StratifiedKFold as KF
from sklearn.metrics import log_loss
from sklearn.preprocessing import LabelBinarizer
from math import log
import numpy as np
import pandas as pd
import math
import copy
import datetime
from sklearn import metrics
from SSA import SSA

labels = [1,2,3]

def pnn_ssa(Filename,acclevel):
    global trainss,validationss,validationrange,testss

    trainss=75  #81
    validationss=15 #9
    trainset=trainss-validationss
    testss=15

    Lambada_overfit=0.1
    acclevel=acclevel
    Lambada_overfitpunch=0.0999999999999
    bound=10
    pop_size = 10

    def problemm(x):
    
        x=pd.DataFrame(x)
        alist=[]
        costlist=[]
        mset=[]
        msev=[]
        fclist=[]
        list=[]
        
        for l in range(pop_size):     
            c = pd.read_csv(Filename,header=None)
            c.iloc[:,0] = c.iloc[:,0]*x.iat[l,0]
            c.iloc[:,1] = c.iloc[:,1]*x.iat[l,1]
            c.iloc[:,2] = c.iloc[:,2]*x.iat[l,2]
            c.iloc[:,3] = c.iloc[:,3]*x.iat[l,3]
    
            XX=c.iloc[:,0:-1].values
            yy=c.iloc[:,-1].values
            kf=KF(n_splits=5,shuffle=True)
            
            ##训练模型      
            def runpnn():
                trainX, labelX = load_data("name.txt")
                Nor_trainX = Normalization(trainX[0:trainset,:])
                Nor_testX = Normalization(trainX[0:trainset,:]) ###trainss
                Euclidean_D = distance_mat(Nor_trainX,Nor_testX)
                Gauss_mat = Gauss(Euclidean_D,m)
                Prob,label_class = Prob_mat(Gauss_mat,labelX)
                predict_results = calss_results(Prob,label_class)
                Pa=predict_results
                Pb=labelX[0:trainset]  ###trainss
                pa=pd.DataFrame(Pa)
                pb=pd.DataFrame(Pb)
                scoreee=pa-pb
                scoreee=scoreee.T
                scoreee=scoreee.values.tolist()
                scoreb = [ i for item in scoreee for i in item]
                asss=0
                
                squaredError = []
                for val in scoreb:
                    squaredError.append(val * val)#target-prediction之差平方 
                    
                MSE=sum(squaredError) / len(squaredError)
                for i in range(len(scoreb)):
                    if scoreb[i] == 0:
                        asss=asss+1
                finsc=(trainset-asss)/trainset
               
                #验证集准确率
                Nor_testX = Normalization(trainX[trainset:trainss,:]) 

                Euclidean_D = distance_mat(Nor_trainX,Nor_testX)
                Gauss_mat = Gauss(Euclidean_D,m)
                Prob,label_class = Prob_mat(Gauss_mat,labelX)
                predict_results = calss_results(Prob,label_class)
                scoreeeb=(pd.DataFrame(predict_results)-pd.DataFrame(labelX[trainset:trainss])).T.values.tolist()

                scorebb = [ i for item in scoreeeb for i in item]
                asssb=0
                
                squaredErrorv = []
                for valv in scorebb:
                    squaredErrorv.append(valv * valv)#target-prediction之差平方 
                    
                MSEv=sum(squaredErrorv)/len(squaredErrorv)
                #print (Pbb, Prob)
                where_are_null = np.isnan(Prob)
                Prob[where_are_null] = 0.1
                
                sk_log_lossv = log_loss(Pbb, Prob, labels=labels)

                for i in range(len(scorebb)):
                    if scorebb[i] == 0:
                        asssb=asssb+1
                finscb=(validationss-asssb)/validationss

                judge=finscb-finsc
                worngn=validationss-asssb
                return (worngn,m,finsc,finscb,MSE,MSEv,sk_log_lossv)

            scorev=0
            for traina,testa in kf.split(XX,yy):                
                X_train, X_test, y_train,y_test=XX[traina],XX[testa],yy[traina],yy[testa]                
                X_train=pd.DataFrame(X_train)                
                X_test=pd.DataFrame(X_test)                
                ccc=pd.concat([X_train,X_test],axis=0)                
                #np.savetxt('name.txt',ccc,fmt='%d', delimiter='\t')
                worngn,m,finsc,finscb,MSE,MSEv,sk_log_lossv=runpnn()

                worngn=worngn**3
                if m<Lambada_overfitpunch:
                    im=100
                elif m>=Lambada_overfitpunch:
                    im=1

                finalscore=sk_log_lossv*im     
                scorev=scorev+finalscore

            alist.append(scorev)
            costlist.append(finalscore)

        alist=pd.DataFrame(alist)
        alist=alist.values
        
        sla=np.sum(alist, 1) #鹅对此2
        return sla
    #开始疯狂循环
    cr1=SSA(problemm).run(True)
    
    m0=cr1.iat[4,0]
    m=(np.abs(m0)+0.1)/bound
    
    c = pd.read_csv(Filename,header=None)
    c.iloc[:,0] = c.iloc[:,0]*cr1.iat[l,0]
    c.iloc[:,1] = c.iloc[:,1]*cr1.iat[l,1]
    c.iloc[:,2] = c.iloc[:,2]*cr1.iat[l,2]
    c.iloc[:,3] = c.iloc[:,3]*cr1.iat[l,3]
    cci=pd.concat([a0s,a1xs,b1xs,c1xs,a6s],axis=1)        
    #保存加权后的文件
    np.savetxt('nameu.txt',c,fmt='%d', delimiter='\t')

    def runpnnr():
    # 1、导入数据
        trainX, labelX = load_data("nameu.txt")   ##########nameuc
    
    # 2、样本数据归一化 
        #模型的训练集
        Nor_trainX = Normalization(trainX[0:trainss,:])

        #计算训练集准确率
        Nor_testX = Normalization(trainX[0:trainss,:]) #trainX[70:,:]

        Euclidean_D = distance_mat(Nor_trainX,Nor_testX)
        Gauss_mat = Gauss(Euclidean_D,m)
        Prob,label_class = Prob_mat(Gauss_mat,labelX)
        predict_results = calss_results(Prob,label_class)

        scoreee=(pd.DataFrame(predict_results)-pd.DataFrame(labelX[0:trainss])).T.values.tolist()
        score = [ i for item in scoreee for i in item] #去中括号
        asss=0
        for i in range(len(score)):
            if score[i] == 0:
                asss=asss+1

        #训练集正确率
        finsc=asss/trainss

        #计算测试集准确率
        Nor_testX = Normalization(trainX[trainss:,:]) #trainX[70:,:]

        Euclidean_D = distance_mat(Nor_trainX,Nor_testX)
        Gauss_mat = Gauss(Euclidean_D,m)######m
        Prob,label_class = Prob_mat(Gauss_mat,labelX)
        predict_results = calss_results(Prob,label_class)
        
        Predict_r=predict_results
        True_r=labelX[trainss:]
        Prob=pd.DataFrame(Prob)
        Prob=Prob.values
        #各种指标  string, [None, ‘binary’ (default), ‘micro’, ‘macro’, ‘weighted’, ‘samples’]
        recall1=metrics.recall_score(Predict_r,True_r, average='macro')
        recall2=metrics.recall_score(Predict_r,True_r, average='weighted')
        mf1=metrics.f1_score(Predict_r,True_r, average='weighted') #'weighted'
        #kappa1=metrics.cohen_kappa_score(Predict_r,True_r)
        prec=metrics.precision_score(Predict_r,True_r,average='weighted') # average='micro'  'weighted'
        accu=metrics.accuracy_score(Predict_r,True_r)

        return (Predict_r,Prob,True_r,finsc,accu,prec,recall1,recall2,mf1)

    return runpnnr()
    ####################ok
