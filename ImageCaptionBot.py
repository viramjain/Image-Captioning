#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np


# # Image Captioning
# ## Generating Captions for images

# # Steps

# ### Data Collection
# ### Understanding the data
# ### Data Cleaning
# ### Loading the training set
# ### Data Preprocessing -images
# ### Data Preprocessing -Captions
# ### Data Preparation using Generator Function
# ### Word Embeddings
# ### Model Architecture
# ### Inference 
# ### Evaluation

# In[59]:


# Read text Captions
def readTextFile(path):
    with open(path) as f:
        captions=f.read()
    return captions


# In[60]:


captions=readTextFile('Flickr_Data/Flickr_TextData/Flickr8k.token.txt') 


# In[61]:


captions=captions.split("\n")[:-1]


# In[62]:


captions[0]


# In[63]:


len(captions)


# In[64]:


# Dictionary to Map each image with the list of captions it has
first,second=captions[0].split("\t")
print(first.split(".")[0])


# In[65]:


descriptions={}


# In[66]:


for x in captions:
    first,second=x.split("\t")
    img_name=first.split(".")[0]
    # if the image id is already present or not
    if descriptions.get(img_name) is None:
        descriptions[img_name]=[]
    descriptions[img_name].append(second)


# In[67]:


descriptions['1000268201_693b08cb0e']


# In[68]:


IMG_PATH='Flickr_Data/Images/'


# In[69]:


import cv2
import matplotlib.pyplot as plt


# In[70]:


img=cv2.imread(IMG_PATH+'1000268201_693b08cb0e.jpg')
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis("off")
plt.show()


# # Data Cleaning

# In[71]:


import keras
import re
import nltk
from nltk.corpus import stopwords
import string 
import json
from time import time
import pickle
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions
from keras.preprocessing import image
from keras.models import Model,load_model
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input,Dense,Dropout,Embedding,LSTM
from keras.layers.merge import add


# In[72]:


def clean_text(sentence):
    sentence=sentence.lower()
    sentence=re.sub("[^a-z]+"," ",sentence)
    sentence=sentence.split()
    sentence=[s for s in sentence if len(s)>1]
    sentence=" ".join(sentence)
    return sentence
    


# In[73]:


clean_text("A cat is sitting over the house number 64")


# In[74]:


# Clean all Captions
for key,caption_list in descriptions.items():
    for i in range(len(caption_list)):
        caption_list[i]=clean_text(caption_list[i])


# In[75]:


descriptions['1000268201_693b08cb0e']


# In[76]:


# Write the data to text file
with open('descriptions_1.txt','w') as f:
    f.write(str(descriptions))


# # Vocabulary

# In[77]:


descriptions=None
with open('descriptions_1.txt','r') as f:
    descriptions=f.read()
descriptions=json.loads(descriptions.replace("'","\""))
print(type(descriptions))


# In[78]:


descriptions['1000268201_693b08cb0e']


# In[79]:


# Vocab
vocab=set()
for key in descriptions.keys():
    [vocab.update(sentence.split()) for sentence in descriptions[key]]
print("Vocab %d"%len(vocab))


# In[80]:


# Total number of words across all the sentence
total_words=[]
for key in descriptions.keys():
    [total_words.append(i) for des in descriptions[key] for i in des.split()]
print("Total Words %d"%len(total_words))


# In[81]:


total_words[:10]


# In[82]:


# Filter words from the vocab according to certain threshold frequency
import collections
counter=collections.Counter(total_words)
freq_cnt=dict(counter)
print(freq_cnt)


# In[83]:


len(freq_cnt.keys())


# In[84]:


# Sort this dict according to frequency count
sorted_freq_cnt=sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])


# In[85]:


# Filter
threshold=10
sorted_freq_cnt=[x for x in sorted_freq_cnt if x[1]>threshold]
total_words=[x[0] for x in sorted_freq_cnt]


# In[86]:


total_words


# In[87]:


len(total_words)


# # Prepare Train/Test Data

# In[88]:


train_file_data=readTextFile('Flickr_Data/Flickr_TextData/Flickr8k.token.txt')


# In[89]:


train_file_data


# In[90]:


test_file_data=readTextFile('Flickr_Data/Flickr_TextData/Flickr_8k.testImages.txt')


# In[91]:


test_file_data


# In[92]:


train=[row.split(".")[0] for row in train_file_data.split("\n")[:-1]]
print(train[:10])


# In[93]:


test=[row.split(".")[0] for row in test_file_data.split("\n")[:-1]]


# In[94]:


# Prepare description for training data
# Tweak - Add <s> and <e> token to our training data
train_descriptions={}


# In[95]:


for img_id in train:
    train_descriptions[img_id]=[]
    for  cap in descriptions[img_id]:
        cap_to_append="startseq "+cap+" endseq"
        train_descriptions[img_id].append(cap_to_append)


# In[96]:


train_descriptions['1000268201_693b08cb0e']


# # Tranfer Learning
# - Images --> Features
# - Text --> Features

# # Step 1 : Image Feature Extraction

# In[97]:


model=ResNet50(weights='imagenet',input_shape=(224,224,3))
model.summary()


# In[98]:


model.layers[-2].output


# In[99]:


model_new=Model(model.input,model.layers[-2].output)


# In[100]:


def preprocess_img(img):
    img=image.load_img(img,target_size=(224,224))
    img=image.img_to_array(img)
    img=np.expand_dims(img,axis=0)
    # Normalisation
    img=preprocess_input(img)
    return img


# In[101]:


img=preprocess_img(IMG_PATH+'1000268201_693b08cb0e.jpg')


# In[102]:


plt.imshow(img[0])


# In[103]:


def encode_img(img):
    img=preprocess_img(img)
    feature_vector=model_new.predict(img)
    
    feature_vector=feature_vector.reshape((-1,))
    return feature_vector


# In[104]:


encode_img(IMG_PATH+'1000268201_693b08cb0e.jpg')


# In[53]:


encoding_train={}
start=time()
# image_id-->feature_vector extracted from resnet image
for ix,img_id in enumerate(train):
    img_path=IMG_PATH+"/"+img_id+".jpg"
    encoding_train[img_id]=encode_img(img_path)
    if ix%100==0:
        print("encoding in progress Time Step %d"%(ix))
end_t=time()
print("Total Time Taken:",end_t-start)


# In[54]:


# Store Everthing to disk
with open("encoding_train_features.pkl","wb") as f:
    pickle.dump(encoding_train,f)


# In[110]:


encoding_test={}
start=time()
# image_id-->feature_vector extracted from resnet image
for ix,img_id in enumerate(test):
    img_path=IMG_PATH+"/"+img_id+".jpg"
    encoding_test[img_id]=encode_img(img_path)
    if ix%100==0:
        print("encoding in progress Time Step %d"%(ix))
end_t=time()
print("Total Time Taken:",end_t-start)


# In[111]:


with open("encoding_test_features.pkl","wb") as f:
    pickle.dump(encoding_test,f)


# In[108]:


with open("encoding_train_features.pkl","rb") as f:
    encoding_train=pickle.load(f)


# In[112]:


with open("encoding_test_features.pkl","rb") as f:
    encoding_test=pickle.load(f)


# # Data Preprocessing for Captions

# In[113]:


len(total_words)


# In[114]:


word_to_idx={}
idx_to_word={}
for i,word in enumerate(total_words):
    word_to_idx[word]=i+1
    idx_to_word[i+1]=word


# In[115]:


len(idx_to_word)


# In[116]:


# Two special words
idx_to_word[1846]='startseq'
word_to_idx['startseq']=1846
idx_to_word[1847]='endseq'
word_to_idx['endseq']=1847
vocab_size=len(word_to_idx)+1
print(vocab_size)


# In[117]:


all_captions_len=[]
max_len=0
for key in train_descriptions.keys():
    for cap in train_descriptions[key]:
        all_captions_len.append(len(cap.split()))
        max_len=max(max_len,len(cap.split()))
print(max_len)


# In[118]:


with open("word_to_idx.pkl","wb") as w:
    pickle.dump(word_to_idx,w)


# In[119]:


with open("idx_to_word.pkl","wb") as w:
    pickle.dump(idx_to_word,w)


# # Data Loader(Generator)

# In[140]:


def data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size):
    x1,x2,y=[],[],[]
    n=0
    while True:
        for key,desc_list in train_descriptions.items():
            n+=1
            photo=encoding_train[key]
            for desc in desc_list:
                seq=[word_to_idx[word] for word in desc.split() if word in word_to_idx]
                for i in range(1,len(seq)):
                    xi=seq[0:i]
                    yi=seq[i]
                    xi=pad_sequences([xi],maxlen=max_len,value=0,padding='post')[0]
                    yi=to_categorical([yi],num_classes=vocab_size)[0]
                    x1.append(photo)
                    x2.append(xi)
                    y.append(yi)
                if n==batch_size:
                    yield ([np.array(x1),np.array(x2),np.array(y)])
                    x1,x2,y=[],[],[]
                    n=0


# # Word Embeddings

# In[121]:


f=open('glove.6B.50d.txt',encoding='utf8')


# In[122]:


embedding_index={}


# In[123]:


for line in f:
    values=line.split()
    print(values)
    word=values[0]
    word_embedding=np.array(values[1:],dtype='float')
    embedding_index[word]=word_embedding


# In[124]:


f.close()


# In[125]:


embedding_index['apple']


# In[126]:


def get_embedding_matrix():
    emb_dim=50
    matrix=np.zeros((vocab_size,emb_dim))
    for word,idx in word_to_idx.items():
        embedding_vector=embedding_index.get(word)
        if embedding_vector is not None:
            matrix[idx]=embedding_vector
    return matrix


# In[127]:


embedding_matrix=get_embedding_matrix()


# In[128]:


embedding_matrix.shape


# # Model Architecture

# In[129]:


input_img_features=Input(shape=(2048,))
inp_img1=Dropout(0.3)(input_img_features)
inp_img2=Dense(256,activation='relu')(inp_img1)


# In[130]:


# Caption as Input
input_captions=Input(shape=(max_len,))
inp_cap1=Embedding(input_dim=vocab_size,output_dim=50,mask_zero=True)(input_captions)
inp_cap2=Dropout(0.3)(inp_cap1)
inp_cap3=LSTM(256)(inp_cap2)


# In[131]:


decoder1=add([inp_img2,inp_cap3])
decoder2=Dense(256,activation='relu')(decoder1)
outputs=Dense(vocab_size,activation='softmax')(decoder2)


# In[132]:


# Combined Model
model=Model(inputs=[input_img_features,input_captions],outputs=outputs)
model.summary()


# In[133]:


model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable=False


# In[134]:


model.compile(loss='categorical_crossentropy',optimizer='adam')


# # Training of Model

# In[138]:


epochs=20
batch_size=3
steps=len(train_descriptions)//batch_size


# In[ ]:


from sklearn.externals import joblib


# In[72]:


encoding_train=joblib.load("encoding_train_features.pkl")


# In[141]:


def train():
    for i in range(epochs):
        generator=data_generator(train_descriptions,encoding_train,word_to_idx,max_len,batch_size)
        model.fit_generator(generator,epochs=1,steps_per_epoch=steps,verbose=1)
        model.save('./model_weights/model_'+str(i)+'.h5')
train()


# In[142]:


model=load_model('model_9.h5')


# # Predictions

# In[143]:



def predict_caption(photo):
    
    in_text = "startseq"
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence],maxlen=max_len,padding='post')
        
        ypred = model.predict([photo,sequence])
        ypred = ypred.argmax() #WOrd with max prob always - Greedy Sampling
        word = idx_to_word[ypred]
        in_text += (' ' + word)
        
        if word == "endseq":
            break
    
    final_caption = in_text.split()[1:-1]
    final_caption = ' '.join(final_caption)
    return final_caption


# In[77]:


# Pick Some random images and see result


# In[145]:


plt.style.use("seaborn")
for i in range(15):
    idx = np.random.randint(0,1000)
    all_img_names = list(encoding_test.keys())
    img_name = all_img_names[idx]
    photo_2048 = encoding_test[img_name].reshape((1,2048))
    
    i = plt.imread(IMG_PATH+img_name+".jpg")
    
    caption = predict_caption(photo_2048)
    #print(caption)
    
    plt.title(caption)
    plt.imshow(i)
    plt.axis("off")
    plt.show()


# In[ ]:




