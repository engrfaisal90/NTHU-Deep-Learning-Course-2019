#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
import pandas as pd
import re
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense,Embedding


# In[ ]:


def read():
    print ("reading  data")
    with open("en.txt", 'r', encoding='utf-8') as f:
        english = f.read().split('\n')
    with open("fr.txt", 'r', encoding='utf-8') as f:
        french = f.read().split('\n')
    print ("reading  data finished")
    return english,french


# In[ ]:


def pre_process(english, french):
    print ("preprocessing data")
    eng1= []
    for line in range(len(english)) :
        stop_words = set(stopwords.words('english')) 
        word_tokens = word_tokenize(english[line]) 
        filtered_sentence = []
        for w in word_tokens: 
            if w not in stop_words: 
                filtered_sentence.append(w) 
        eng1.append(filtered_sentence)
    
    
    for line in range(len(french)) :
        french[line] ='START_ '+ french[line] + ' _END'

    all_eng_words=set()
    for eng in english:
        for word in eng.split():
            if word not in all_eng_words:
                all_eng_words.add(word)

    all_french_words=set()
    for fre in french:
        for word in fre.split():
            if word not in all_french_words:
                all_french_words.add(word)

    len_list=[]
    for l in english:
        len_list.append(len(l.split(' ')))
    max_len_src = np.max(len_list)

    len_list=[]
    for l in french:
        len_list.append(len(l.split(' ')))
    max_len_tar = np.max(len_list)

    inp_words = sorted(list(all_eng_words))
    tar_words = sorted(list(all_french_words))

    max_len_fr=max([len(txt) for txt in french])
    max_len_eng=max([len(txt) for txt in english])
    num_enc_tokens = len(all_eng_words)
    num_dec_tokens = len(all_french_words)
    num_dec_tokens += 1
    inp_token_index = dict(
        [(word, i) for i, word in enumerate(inp_words)])
    tar_token_index = dict(
        [(word, i) for i, word in enumerate(tar_words)])
    enc_inp_data = np.zeros(
        (len(english), max_len_eng),
        dtype='float32')
    dec_inp_data = np.zeros(
        (len(french), max_len_fr),
        dtype='float32')
    dec_tar_data = np.zeros(
        (len(french), max_len_fr, num_dec_tokens),
        dtype='float32')
    for i, (inp_text, tar_text) in enumerate(zip(english, french)):
        for t, word in enumerate(inp_text.split()):
            enc_inp_data[i, t] = inp_token_index[word]
        for t, word in enumerate(tar_text.split()):
            dec_inp_data[i, t] = tar_token_index[word]
            if t > 0:
                dec_tar_data[i, t - 1, tar_token_index[word]] = 1.
    print ("preprocessing finished")
    return num_enc_tokens,num_dec_tokens,enc_inp_data,dec_inp_data,dec_tar_data,inp_token_index,tar_token_index


# In[ ]:


english, french =read()
num_enc_tokens,num_dec_tokens,enc_inp_data,dec_inp_data,dec_tar_data,inp_token_index,tar_token_index=pre_process(english, french)


# In[ ]:


embedding_size = 50
enc_inp = Input(shape=(None,))
en_x=  Embedding(num_enc_tokens, embedding_size)(enc_inp)
enc = LSTM(50, return_state=True)
enc_outputs, st_h, st_c = enc(en_x)
enc_st = [st_h, st_c]
dec_inp = Input(shape=(None,))
dex=  Embedding(num_dec_tokens, embedding_size)
final_dex= dex(dec_inp)
dec_lstm = LSTM(50, return_sequences=True, return_state=True)
dec_outputs, _, _ = dec_lstm(final_dex,
                                     initial_state=enc_st)
dec_dense = Dense(num_dec_tokens, activation='softmax')
dec_outputs = dec_dense(dec_outputs)
model = Model([enc_inp, dec_inp], dec_outputs)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
model.fit([enc_inp_data, dec_inp_data], dec_tar_data,
          batch_size=512,
          epochs=100,
          validation_split=0.05)

dec_st_inp_h = Input(shape=(50,))
dec_st_inp_c = Input(shape=(50,))
dec_st_inp = [dec_st_inp_h, dec_st_inp_c]

final_dex2= dex(dec_inp)

dec_outputs2, st_h2, st_c2 = dec_lstm(final_dex2, initial_st=dec_st_inp)
dec_st = [st_h2, st_c2]
dec_outputs2 = dec_dense(dec_outputs2)
dec_model = Model(
    [dec_inp] + dec_st_inp,
    [dec_outputs2] + dec_st)

reverse_inp_char_index = dict(
    (i, char) for char, i in inp_token_index.items())
reverse_tar_char_index = dict(
    (i, char) for char, i in tar_token_index.items())


def decode_it(inp_seq):
    st_value = model.predict(inp_seq)
    tar_seq = np.zeros((1,1))
    tar_seq[0, 0] = tar_token_index['START_']
    stop_condition = False
    line_deco = ''
    while not stop_condition:
        output_tokens, h, c = dec_model.predict(
            [tar_seq] + st_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_tar_char_index[sampled_token_index]
        line_deco += ' '+sampled_char
        if (sampled_char == '_END' or
           len(line_deco) > 52):
            stop_condition = True
        tar_seq = np.zeros((1,1))
        tar_seq[0, 0] = sampled_token_index

        st_value = [h, c]

    return line_deco

for seq_index in [14077,20122,40035,40064, 40056]:
    inp_seq = enc_inp_data[seq_index: seq_index + 1]
    decoded_sentence = decode_it(inp_seq)
    print(english[seq_index: seq_index + 1])
    print(decoded_sentence)
    
writef=""
with open("test.txt", 'r', encoding='utf-8') as f:
    test = f.read().split('\n')
    num_enc_tokens,num_dec_tokens,enc_inp_data,dec_inp_data,dec_tar_data,inp_token_index,tar_token_index= pre_process(test)
    decoded_sentence = decode_it(enc_inp_data)
    writef=writef+"\n"+decoded_sentence
check=int(input("Do you want to overwrite the file created? press 1 for yes"))
if check==1:
    file = open('test_studentid.txt','w') 
    file.write(writef)
    file.close()
    print("Data saved in file test_studentid.txt\n\n")


# In[ ]:




