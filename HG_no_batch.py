import numpy as np
import csv


load_data_de = np.load('../data_new/keywords_dict_delicious.npy', allow_pickle=True)
key_embedding_dict_de = load_data_de[()]

load_data_tw = np.load('../data_new/keywords_dict_twitter.npy', allow_pickle=True)
key_embedding_dict_tw = load_data_tw[()]

with open("../data_new/delicious_post.csv", "r", encoding ="utf-8") as f:
    reader = csv.reader(f)
    userinfos_de = [row for row in reader]
    del userinfos_de[0]
    #获取delicious网络下所有用户的关键词列表
    userinfos_de = np.array(userinfos_de)
    #获取delicious网络下所有用户id列表
    userIds_de = userinfos_de[:,0]

with open("../data_new/twitter_post.csv", "r", encoding ="utf-8") as f:
    reader = csv.reader(f)
    userinfos_tw = [row for row in reader]
    del userinfos_tw[0]
    #获取twitter网络下所有用户的关键词列表
    userinfos_tw = np.array(userinfos_tw)
    #获取twitter网络下所有用户id列表
    userIds_tw = userinfos_tw[:,0]

Ids = np.array([val for val in userIds_de if val in userIds_tw])
np.save('../all_ids/Ids.npy', Ids)
print(Ids.shape)

list1 = []
for i in range(len(userinfos_de)):
    if userinfos_de[i][0] in Ids:
        list1.append(i)
userinfos_de = userinfos_de[list1]

list2 = []
for i in range(len(userinfos_tw)):
    if userinfos_tw[i][0] in Ids:
        list2.append(i)
userinfos_tw = userinfos_tw[list2]

max_key_de = 0
max_key_tw = 0
with open("../data_new/delicious用户post文件.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    postsinfos_de = [row for row in reader]
    del postsinfos_de[0]
    #获取delicious网络下所有用户的post信息
    postsinfos_de = np.array(postsinfos_de)
    #构造每个用户对应所有post的字典，key为用户id，value为该用户所有post信息
    dict_de = {}
    for id_de in Ids:
        id_all_keyword_list_de = []
        id_post_arr = postsinfos_de[postsinfos_de[:,1] == id_de]
        if len(id_post_arr) > 100:
            id_post_arr = id_post_arr[0:100]
        dict_de[id_de] = id_post_arr

        for post in id_post_arr:
            key_list = post[3][1:-1].split(',')
            for i in range(len(key_list)):
                key_list[i] = key_list[i].strip()
                key_list[i] = key_list[i].strip("'")
            for key in key_list:
                id_all_keyword_list_de.append(key)
        id_all_keyword_array_de = np.unique(id_all_keyword_list_de)
        if max_key_de < len(id_all_keyword_array_de):
            max_key_de = len(id_all_keyword_array_de)





with open("../data_new/twitter用户post文件.csv", "r", encoding="utf-8") as f:
    reader = csv.reader(f)
    postsinfos_tw = [row for row in reader]
    del postsinfos_tw[0]
    #获取delicious网络下所有用户的post信息
    postsinfos_tw = np.array(postsinfos_tw)
    #构造每个用户对应所有post的字典，key为用户id，value为该用户所有post信息
    dict_tw = {}
    for id_tw in Ids:
        id_all_keyword_list_tw = []
        id_post_arr = postsinfos_tw[postsinfos_tw[:, 1] == id_tw]
        if len(id_post_arr) > 100:
            id_post_arr = id_post_arr[0:100]
        dict_tw[id_tw] = id_post_arr
        for post in id_post_arr:
            key_list = post[3][1:-1].split(',')
            for i in range(len(key_list)):
                key_list[i] = key_list[i].strip()
                key_list[i] = key_list[i].strip("'")
            for key in key_list:
                id_all_keyword_list_tw.append(key)
        id_all_keyword_array_tw = np.unique(id_all_keyword_list_tw)
        if max_key_tw < len(id_all_keyword_array_tw):
            max_key_tw = len(id_all_keyword_array_tw)


per_5_users_de = []
per_5_users_tw = []


per_5_users_word_de = []
per_5_users_word_tw = []

for i in range(len(userinfos_de)):
    # 超图矩阵的维度参数
    colum_de = max_key_de
    row_de = 100
    colum_tw = max_key_tw
    row_tw = 100

    # 构造关键词字典，key为关键词，value为关键词对应的列数position,
    keywords_dict_de = {}
    keywords_dict_tw = {}

    # 构造关键词字典2，key为position，value为关键词,
    keywords_dict_de_pk = {}
    keywords_dict_tw_pk = {}

    id_all_keyword_list_de = []
    id_post_arr_de = dict_de[Ids[i]]
    for post in id_post_arr_de:
        key_list = post[3][1:-1].split(',')
        for m in range(len(key_list)):
            key_list[m] = key_list[m].strip()
            key_list[m] = key_list[m].strip("'")
        for key in key_list:
            id_all_keyword_list_de.append(key)
    id_all_keyword_array_de = np.unique(id_all_keyword_list_de)

    id_all_keyword_list_tw = []
    id_post_arr_tw = dict_tw[Ids[i]]
    for post in id_post_arr_tw:
        key_list = post[3][1:-1].split(',')
        for n in range(len(key_list)):
            key_list[n] = key_list[n].strip()
            key_list[n] = key_list[n].strip("'")
        for key in key_list:
            id_all_keyword_list_tw.append(key)
    id_all_keyword_array_tw = np.unique(id_all_keyword_list_tw)

    num_de = 0
    num_tw = 0
    for keyword in id_all_keyword_array_de:
        keywords_dict_de[keyword] = num_de
        keywords_dict_de_pk[num_de] = keyword
        num_de += 1

    for keyword in id_all_keyword_array_tw:
        keywords_dict_tw[keyword] = num_tw
        keywords_dict_tw_pk[num_tw] = keyword
        num_tw += 1



    word_de = np.zeros((colum_de, 100))
    for key_num in range(len(id_all_keyword_array_de)):
        word_de[key_num] = key_embedding_dict_de[id_all_keyword_array_de[key_num]]
    word_tw = np.zeros((colum_tw, 100))
    for key_num2 in range(len(id_all_keyword_array_tw)):
        word_tw[key_num2] = key_embedding_dict_tw[id_all_keyword_array_tw[key_num2]]
    per_5_users_word_de.append(word_de)
    per_5_users_word_tw.append(word_tw)

    # 构造该用户在2网络下的超图矩阵
    HG_de = np.zeros((row_de, colum_de))
    HG_tw = np.zeros((row_tw, colum_tw))

    posts_arr_de = dict_de[Ids[i]]
    for j in range(len(posts_arr_de)):
        keywords_in_post_de = posts_arr_de[j][3][1:-1]
        keywords_in_post_list_de = keywords_in_post_de.split(', ')
        for keyword_in_post_list in keywords_in_post_list_de:
            HG_de[j][keywords_dict_de[keyword_in_post_list[1:-1]]] = 1

    posts_arr_tw = dict_tw[Ids[i]]
    for j in range(len(posts_arr_tw)):
        posts_arr_tw = dict_tw[Ids[i]]
        keywords_in_post_tw = posts_arr_tw[j][3][1:-1]
        keywords_in_post_list_tw = keywords_in_post_tw.split(', ')
        for keyword_in_post_list in keywords_in_post_list_tw:
            HG_tw[j][keywords_dict_tw[keyword_in_post_list[1:-1]]] = 1

    per_5_users_de.append(HG_de)
    per_5_users_tw.append(HG_tw)


per_5_users_de = np.array(per_5_users_de)
per_5_users_tw = np.array(per_5_users_tw)


per_5_users_word_de = np.array(per_5_users_word_de)
per_5_users_word_tw = np.array(per_5_users_word_tw)

np.save('../userHyperGraph_new/user_hyper_graph_de.npy', per_5_users_de)
np.save('../userHyperGraph_new/user_hyper_graph_tw.npy', per_5_users_tw)

np.save('../userKeyword_Embedding/uke_de.npy', per_5_users_word_de)
np.save('../userKeyword_Embedding/uke_tw.npy', per_5_users_word_tw)










