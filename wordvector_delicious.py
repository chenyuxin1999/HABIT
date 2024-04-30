import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
from gensim.models.word2vec import Text8Corpus
from glove import Corpus, Glove
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import nltk
import pkg_resources
from symspellpy.symspellpy import SymSpell
import torch
from torch import nn, optim
import math
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import normalize
from sklearn import preprocessing


word2vec_output_file = 'glove.twitter.27B.100d.word2vec.txt'
# 加载模型
glove_model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
# 获得单词cat的词向量
cat_vec = glove_model['frog']
print(cat_vec)
# 获得单词frog的最相似向量的词汇
print(glove_model.most_similar('frog'))

df1 = pd.read_csv('delicious_regex.csv',encoding='utf8')
df1['tag'] = df1['tag'].str.lower()
df1.reset_index(drop=True)
df1.drop(axis=1, columns='Unnamed: 0',inplace=True)
df = df1
df['tag'] = df['tag'].replace('-','')
df['tag'] = df['tag'].str.replace(':','')
df['tag'] = df['tag'].str.replace(',','')
df['tag'] = df['tag'].str.replace('.','')
df['tag'] = df['tag'].str.replace('(','')
df['tag'] = df['tag'].str.replace(')','')
df['tag'] = df['tag'].str.replace('-','')
df['tag'] = df['tag'].str.replace('?','')
df['tag'] = df['tag'].str.replace('_','')
df['tag'] = df['tag'].str.replace('!','')
df['tag'] = df['tag'].str.replace('+','')
df['tag'] = df['tag'].str.replace('"','')
df['tag'] = df['tag'].str.replace("'","")
df['tag'] = df['tag'].str.replace('=','')
df['tag'] = df['tag'].str.replace('@','')
df['tag'] = df['tag'].str.replace('/','')
df['tag'] = df['tag'].str.replace('$','')
df['tag'] = df['tag'].str.replace('#','')
df['tag'] = df['tag'].str.replace('%','')
df['tag'] = df['tag'].str.replace('*','')
df1 = df

wordHasVector = {}
wordNotVector = []
lenlessthanthree=[]
waste = ["CC","CD","DT","EX","FW","IN","LS","JJR","JJ","JJS","MD","PDT","POS","PRP","PRP$","RB","RBR","RBS","RP","TO","WDT","WP","WP$","WRB","UH","VB","VBP","VBZ","VBD","VBN"]
for i in range(len(df1['tag'])-1):
    if df1['tag'][i] in glove_model:
        word = nltk.word_tokenize(df1['tag'][i])
        if (len(df1['tag'][i]) >= 3) and (str(df1['tag'][i]).isalnum()==True) and (str(df1['tag'][i]).isdigit()==False) and (nltk.pos_tag(word)[0][1] not in waste):
            vector = glove_model[df1['tag'][i]]
            wordHasVector[df1['tag'][i]]=vector
        else:
            lenlessthanthree.append(df1['tag'][i])
            wordNotVector.append(df1['tag'][i])
    else:
        wordNotVector.append(df1['tag'][i])

while '' in wordNotVector:
    wordNotVector.remove('')
while "" in wordNotVector:
    wordNotVector.remove("")

sym_spell = SymSpell(max_dictionary_edit_distance = 0,prefix_length=7)
dictionary_path = pkg_resources.resource_filename("symspellpy","frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(dictionary_path,term_index = 0,count_index = 1)
input_term = "webdevelopment"
result = sym_spell.word_segmentation(input_term)
print(result.segmented_string)

class PositionalEncoding(nn.Module):
    "Implement the PE function"
    def __init__(self,d_model,dropout,max_len=6):
        super(PositionalEncoding,self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        #Compute the positional encodings once in log space.
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        # pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self, x):
        x = x + Variable(self.pe[:x.size(0)],requires_grad=False)
        return self.dropout(x)

while 'nan' in wordNotVector:
    wordNotVector.remove('nan')

good_word = True
vector_list = []
goodWordList = []
badWordList = []
pe = PositionalEncoding(100,0)
for i in set(wordNotVector):
    input_term = i
    if len(str(input_term))<3:
        good_word = False
    if str(input_term) =='':
        good_word = False
    if str(input_term) == None:
        good_word = False
    if input_term =='':
        good_word = False
    if input_term ==' ':
        good_word = False
    if str(input_term) == 'nan':
        good_word = False
    if str(input_term).isalnum()==False:
        good_word = False
    if str(input_term).isdigit() ==True:
        good_word = False
    try:
        result = sym_spell.word_segmentation(str(input_term))
    except:
        print(input_term)
    word_list=result.segmented_string.split()
    if len(word_list)>5:
        good_word = False
    for j in word_list:
        if not(j in glove_model):
            good_word = False
    if good_word:
        for z in word_list:
            word_vector = glove_model[z]
            vector_list.append(word_vector)
        vector = np.mean(pe.forward(Variable(torch.tensor(vector_list))).numpy(),axis=0)
        wordHasVector[input_term]=vector
        goodWordList.append(input_term)
        vector_list = []
    else:
        badWordList.append(input_term)
        good_word = True

m = np.array(wordHasVector)
np.save('wordHasVector_delicious.npy',m)
