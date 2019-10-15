import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
import json
import gzip
import seaborn as sns
import os
from gensim.models import Word2Vec
from sklearn.manifold import TSNE


#Read LinkedIn Data 
data_dir = '/home/mohamed/Music/Skill Gap/dataset2'
profiles_path = os.path.join(data_dir, 'linkedin.json')
print(os.listdir(data_dir))

skill_docs = []
job_docs = []
for line in open(profiles_path):
    line = json.loads(line)
    if 'skills' in line:
        skill_docs.append(line['skills'])
    if 'experience' in line:
        job_docs.append([exp['title'] for exp in line['experience'] if 'title' in exp])
        
        
skill_docs_sampile = skill_docs[0:10000]
job_docs_sampile = job_docs[0:10000]

#chunkSizes of CSV files 
# =============================================================================
#  chunkSize = 10000
#  batchNumber = 1 
#  
#  for chunk in pd.read_csv("new_wuzzuf_db.csv", chunksize = chunkSize):
#      chunk.to_csv("new_wuzzuf_db"+str(batchNumber)+".csv", index = False)
#      batchNumber +=1
# =============================================================================


#Readin and Cleaning Wuzzuf from Data 
# =============================================================================
# dataSample = pd.read_csv("new_wuzzuf_db180.csv")
# dataSample = dataSample.iloc[:,[12]]
# dataSample = dataSample.dropna()
# dataSample = dataSample.reset_index()
# skills = []
# for i in range(0, len(dataSample)):
#     review = re.sub('[^a-zA-Z0-9.+/#]', ' ', dataSample['skills'][i])
#     review = review.lower()
#     review = review.split()
#     ps = PorterStemmer()
#     skills.append(review)
# =============================================================================
        
        
# train model    
skill2vec = Word2Vec(skill_docs_sampile, min_count=10, iter=100, workers=32)
print(skill2vec)
# summarize vocabulary
skillsSumary = list(skill2vec.wv.vocab)

# train model    
job2vec = Word2Vec(job_docs_sampile, min_count=10, iter=100, workers=32)
print(job2vec)
# summarize vocabulary
jobs = list(job2vec.wv.vocab)

print('Data Analyst: ', job2vec.wv.most_similar(['Data Analyst']))
