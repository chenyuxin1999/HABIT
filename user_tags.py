import json
import re
import numpy as np

def load_json(file):
    f = open(file, 'r')
    s = f.readlines()
    f.close()
    return s

aminer_authors = load_json('/data/OAG/new-OAG/AI/authors/final_aligned_aminer_authors.json')

tag = open('/data/OAG/new-OAG/AI/authors/tags.txt','r')
lines = tag.readlines()

all_tags = []
for word in lines:
    all_tags.append(word.replace('\n',''))

print(np.shape(all_tags))
user_tags = {}
each_tags = []
for i in range(len(aminer_authors)):
    print(i)
    each_author = json.loads(aminer_authors[i])
    for j in each_author['tags']:
        each_tags.append(all_tags.index(j))
    user_tags[each_author['id']] = each_tags
    each_tags = []

np.save('user_tags.npy',user_tags)
