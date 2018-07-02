# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 03:19:50 2018

@author: Kartik
"""

import numpy as np
import nltk
import torch

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)
 
def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))    

def cos_sim(x,y):
     z=np.multiply(x,y)
     X,Y=np.square(x),np.square(y)
     x1,y1,z1=0,0,0
     for i in range(x.shape[0]):
         x1+=X[i]
         y1+=Y[i]
         z1+=z[i]
     x1,y1=np.sqrt(x1),np.sqrt(y1)
     return(z1/(x1*y1))

def preprocess(corpus):
    from nltk.tokenize import sent_tokenize,word_tokenize    
    sent=[]
    for i in corpus:
        for j in sent_tokenize(i):
            sent.append(j)
                             
    x=sent[:]#did bcoz changes in x got reflected in sentence 
    #while pos tagging in nltk never lowercase text    
    import re
    pattern=re.compile('\W')#have doubt see ho it works
    for i,sent in enumerate(x):
        x[i]=re.sub(pattern,' ',sent) 
    #above cleaned the data
    #below is pos tagging and its stemming    
    temp,tag=[],set()
    for i in x:
        temp.append(nltk.pos_tag(word_tokenize(i)))
    for i in temp:
        for j in i:
            if(j[1]=='NNP' or j[1]=='NNPS'):
                tag.add(j[0]) 
    tag=list(tag) 
    for i,tg in enumerate(tag):
        tag[i]=tg.lower()    
        tag[i]=nltk.PorterStemmer().stem(tg)
    #sentence's stopword filtering and stemming
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))    
    temp=[]
    for lst in x:
        temp.append(word_tokenize(lst))
    for i,lst in enumerate(temp):
        temp[i]=[w for w in lst if not w in stop_words]   
    for i,lst in enumerate(temp):
        for j,word in enumerate(lst):
            temp[i][j]=nltk.PorterStemmer().stem(word)
    x=[' '.join(i) for i in temp]    
    
    sentence=[]
    for i in corpus:
        for j in sent_tokenize(i):
            sentence.append(j)
    lst=[]
    for i in corpus:
        lst.append(sent_tokenize(i))
    para_dict={}
    for i,para in enumerate(lst):
        for j,sent in enumerate(para):
           para_dict[(i,j)]=sent        
    
    return(tag,sentence,x,para_dict)     
  
def feature_mat(tag,sentence,train,para_dict):
    sent=sentence[:]
    import pandas as pd  
    feature=pd.DataFrame() 
    #10 frequent word
    from collections import Counter   
    temp=[]
    for i in train:
        for j in i.split():
            temp.append(j)
    Counter = Counter(temp)        
    frequent = Counter.most_common(10)
    thematic=[]
    for i in frequent:
        thematic.append(i[0])
    temp=[]
    for i,line in enumerate(train):
        denom=train[i].split().__len__()  
        count=0
        for j,word in enumerate(line.split()):
            if word in thematic:
                count+=1
        temp.append(count/(denom+1e-10)) 
    se = pd.Series(temp)       
    feature['thematic'] = se.values
    #sentence pos
    temp=[]
    min,max=0.2*((train.__len__())**2),0.2*((train.__len__())**2)*2
    for i in range((train.__len__())):
        if i==0 or i==((train.__len__())-1):
           temp.append(1)
           continue
        else:
            val=np.cos((i+1-min)*((1/max)-min))
            temp.append(val)
    se=pd.Series(temp)        
    feature['sen_pos']=se.values 
    #sen_length
    temp=[]
    for i in train:
        k=(i.split().__len__())
        if k<3:
            temp.append(0)
        else:
            temp.append(k)         
    se=pd.Series(temp)        
    feature['sen_length']=se.values             
    #no of proper noun
    temp=[]
    for i in train:
        count=0
        for j in i.split():
            if j in tag:
                count+=1
        temp.append(count)    
    se=pd.Series(temp)        
    feature['proper_noun']=se.values  
    #numeral
    temp=[]
    for i in train:
        count=0
        for j in i.split():
            if j.isnumeric():
                count+=1
        temp.append(count) 
    se=pd.Series(temp)        
    feature['numeral']=se.values 
    #named entity recog
    temp=[]
    for i in sent:
        temp.append(nltk.word_tokenize(i))
    ne=set()    
    for i in temp:    
        for chunk in nltk.ne_chunk(nltk.pos_tag(i)):
            if hasattr(chunk, 'label'):#have doubt see ho it works
               ne.add(chunk[0][0])
    ne=list(ne)           
    sne=[]
    for i in ne:
        sne.append(nltk.PorterStemmer().stem(i))
    temp=[]
    for i in train:
        count=0
        for j in i.split():
            if j in sne:
               count+=1
        temp.append(count)    
    se=pd.Series(temp)        
    feature['name_entity']=se.values 
    #tf_idf
    isf={}
    word=[]
    for i in train:
        for j in i.split():
            word.append(j)
    from collections import Counter     
    Counter = Counter(word)        
    freq = Counter.most_common()
    for tup in freq:
        isf[tup[0]]=tup[1]        
    tf_isf=[]
    for i in train:
        l=i.split().__len__()
        temp=[]
        for j in i.split():
            temp.append(j)
        from collections import Counter     
        Counter = Counter(temp)        
        freq = Counter.most_common()
        tf={}
        for i in freq:
            tf[i[0]]=i[1]
        sum=0
        for i,j in tf.items():
            sum+=tf[i]*isf[i]
        tf_isf.append(np.log(sum)/(l+1e-5))  
    se=pd.Series(tf_isf)        
    feature['tf_isf']=se.values     
    #centroid similarity
    big=0
    for i in tf_isf:
        if big<i:
            big=i
    index=tf_isf.index(big)        
    lst=train[index].split()       
    sent_sim=[]
    for i in train:
        l=i.split().__len__()
        sim=0
        for j in i.split():
            if j in lst:
                sim+=1
        sent_sim.append(sim/(l+1e-10))#    
    se=pd.Series(sent_sim)        
    feature['cent_sim']=se.values     
    ##para pos
    par_pos=[]
    lst=[]
    for j,k in para_dict.items():
        lst.append([j[0],j[1],k])
    for h,i in enumerate(lst):
        if i[1]==0:
            par_pos.append(1)
            continue
        if h!=lst.__len__()-1 and lst[h+1][0]>i[0]:
           par_pos.append(1)
           continue
        else:
            par_pos.append(0)
    par_pos[lst.__len__()-1]=1        
    se=pd.Series(par_pos)        
    feature['par_pos']=se.values 
    return(feature)
    
def feat_enhance(feature):
    X=feature.iloc[:,:].values   
    #normalized between 0 and 1
    def sigmoid(x, derivative=False):
      return x*(1-x) if derivative else 1/(1+np.exp(-x))    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j]=sigmoid(X[i][j])
    lst=[]
    for i in range(X.shape[0]):
        n=[]
        for j in range(X.shape[1]):
            n.append(X[i,j])
        lst.append(n)            
    lst=torch.FloatTensor(lst) 
    rbm=RBM(9,7)                   
    nb_epoch,batch_size=5,4
    for epoch in range(1, nb_epoch + 1):
        train_loss = 0
        s = 0.
        for id_user in range(0, X.shape[0] - batch_size, batch_size):
            vk = lst[id_user:id_user+batch_size]
            v0 = lst[id_user:id_user+batch_size]
            ph0,_ = rbm.sample_h(v0)
            for k in range(4):
                _,hk = rbm.sample_h(vk)    
                _,vk = rbm.sample_v(hk)
            phk,_ = rbm.sample_h(vk)
            rbm.train(v0, vk, ph0, phk)
            train_loss += torch.mean(torch.abs(v0 - vk))
            s += 1.
        print('epoch: '+str(epoch)+' loss: '+str(train_loss/s))
    y = np.array([[]])
    for x in range(X.shape[0]):
        v=lst[x:x+1]
        m,_=rbm.sample_h(v)
        m=m.numpy()
        if x==0:
            y = np.hstack((y, m))
        else:
            y = np.vstack((y, m))   
  
    z=list()
    for i in range(y.shape[0]):
        z.append(0)
    
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            z[i]+=y[i][j]
    ind_enhance,temp,ind_normal=[],z[:],[]
    temp.sort()
    temp.reverse()
    for i in temp:
        ind_enhance.append(z.index(i)) 
    
    z=list()
    for i in range(y.shape[0]):
        z.append(0)
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            z[i]+=X[i][j]
    temp=z[:]        
    temp.sort()
    temp.reverse()
    for i in temp:
        ind_normal.append(z.index(i))    
    return(ind_enhance,ind_normal)     

def to_word(summar):
    import re
    pattern=re.compile('\W')#have doubt see ho it works
    for i,sent in enumerate(summar):
        summar[i]=re.sub(pattern,' ',sent) 
    temp=[]
    for line in summar:
        temp.append(nltk.word_tokenize(line))    
    
    from nltk.corpus import stopwords
    stop_words=set(stopwords.words('english'))    
    for i,lst in enumerate(temp):
        temp[i]=[w for w in lst if not w in stop_words]      
    sum_word=set()
    for i in temp:
        for j in i:
            sum_word.add(j)
    sum_word=list(sum_word) 
    for i,word in enumerate(sum_word):
        sum_word[i]=nltk.PorterStemmer().stem(word)
    return(sum_word,summar.__len__()) 

len,prec,num,i,recall=[],[],18,1,[]

while num:
    
    k=i
    data_path='data/articles/article'+str(k)
    
    with open(data_path,'r',encoding="utf8") as f:#'r' means read
        corpus=f.read().split('\n\n')
        
    tag,sentence,train,para_dict=preprocess(corpus)   
    
    feature=feature_mat(tag,sentence,train,para_dict)
    
    ind_enhance,ind_normal=feat_enhance(feature)   
        
    data_path1='data/outputs/article'+str(k)
    
    with open(data_path1,'r',encoding="utf8") as f:#'r' means read
        summar=f.read().split('\n')
    sum_word,l=to_word(summar)      
    
    X=feature.iloc[:,:].values
    
             
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i][j]=sigmoid(X[i][j])
    X[ind_enhance[0],:]
    temp1=[]    
    for i in range(0,int((ind_enhance.__len__()))):
        temp1.append(cos_sim(X[ind_enhance[0],:],X[i,:]))
    
    created_sum=[]
    for i in range(l):
        created_sum.append(sentence[ind_enhance[i]])
#    c_sum=' ' 
#    for i in created_sum:
#        c_sum+=i  
    
    csum_word,_=to_word(created_sum)  
    
    count=0
    for i in csum_word:
        if i in sum_word:
            count+=1
    sent=summar    
    
    len.append(sent.__len__())
    prec.append(count/csum_word.__len__()) 
    recall.append(count/sum_word.__len__())
    num-=1
    i=k
    i=i+1

print(prec)
print('************')
print(recall)

import matplotlib.pyplot as plt
fig=plt.figure()
len1, prec1 = zip(*sorted(zip(len, prec)))#important
ax1=fig.add_subplot(211)
ax1.plot(len1,prec)
ax1.set_ylim([0,1])
plt.xlabel('no of summary sentence')
plt.ylabel('precision')
plt.legend()

len1, recall1 = zip(*sorted(zip(len, recall)))
ax2=fig.add_subplot(212)
ax2.plot(len1,recall1)
ax2.set_ylim([0,1])
plt.xlabel('no of summary sentence')
plt.ylabel('recall')
plt.legend()

plt.show()


    


    
