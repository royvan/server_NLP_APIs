from django.shortcuts import render

# Create your views here.

from django.http import HttpResponse
from gensim.models.word2vec import Word2Vec
from nltk.corpus import stopwords
from operator import itemgetter
import datetime
import time
import json
import csv
import pandas
from pandas.io import gbq
from mwviews.api import PageviewsClient
import json
import sys
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
from .BqLoader import *
sys.stdout.flush()

#######################################################################################################
# Setting
#######################################################################################################
test_str = 'burgerking'
test_str = test_str.lower()

ROOT_PROJ_PATH =os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#local path
MODEL_PATH = ROOT_PROJ_PATH + "/create_model/embedding_model_SkipGram"
TABLE_PATH = ROOT_PROJ_PATH + "/create_model/word_count_table.json"
WM_PATH = ROOT_PROJ_PATH + "/create_model/weight_matrix.csv"
TOKENIZER_PATH = ROOT_PROJ_PATH + "/create_model/keras_tokenizer.pickle"
SENTIMENT_MODEL_PATH = ROOT_PROJ_PATH + "/create_model/yelp_sentiment_model.hdf5"

#Server path
#MODEL_PATH = "/usr/local/etc/django/model/embedding_model_SkipGram"
#TABLE_PATH = "/usr/local/etc/django/model/word_count_table.json"
#WM_PATH = "/usr/local/etc/django/model/weight_matrix.csv"
#TOKENIZER_PATH = "/usr/local/etc/django/model/keras_tokenizer.pickle"
#SENTIMENT_MODEL_PATH = "/usr/local/etc/django/model/yelp_sentiment_model.hdf5"

model = Word2Vec.load(MODEL_PATH)

with open(TABLE_PATH) as f:
    term_frequencys = f.readlines()

with open(TOKENIZER_PATH, "rb") as g:
    tokenizer = pickle.load(g)
sent_model = load_model(SENTIMENT_MODEL_PATH)


#######################################################################################################
#   NLP_API1 Functions
#######################################################################################################
# return frequency function
term_frequency_list = [json.loads(term_frequency, encoding = 'utf-8') for term_frequency in term_frequencys]
term_frequency_dict = term_frequency_list[0]
def return_frequency(word):
    word = word.lower()
    frequency = 0
    try :
        frequency = term_frequency_dict[str(word)]
    except :
        frequency = frequency
    return frequency

# input word -> similar word list return function(output : 2차원 리스트)
def return_similar_word_list(word,level):
    similar_output_list = []
    xx = model.most_similar(positive = [str(word)], topn = 40)
    for i in xx:
        yy = i[0]
        similar_tem_list=[]
        similar_tem_list.append(yy)
        for j in xx:
            if model.similarity(yy, j[0]) >= float(level):
                similar_tem_list.append(j[0])
                xx.remove(j)
        similar_output_list.append(similar_tem_list)
    return similar_output_list

# input word -> similar word list return function(세부항목 name 모두)
def return_similar_word_name(kb_list):
    similar_output_list_name = []
    for i in kb_list:
        similar_output_list_name.append(i[0])
    return similar_output_list_name

# similar_word_list -> average frequency function (output: 1차원 리스트)
def return_avg_frequency(word_list):
    similar_word_frequency = []
    for i in word_list:
        frequency_tem_list = []
        avg = 0
        for j in i:
            frequency_tem_list.append(return_frequency(j))
        #avg = sum(frequency_tem_list) / len(frequency_tem_list)
        avg = sum(frequency_tem_list)
        similar_word_frequency.append(int(avg))
    return(similar_word_frequency)


# similar_word_list -> sentiment score list function
"""
def return_sentiment_score(word_list):
    for z in word_list:
        for keyword in z:
            newtexts = [keyword]
            sequences = tokenizer.texts_to_sequences(newtexts)
            data = pad_sequences(sequences, maxlen = 300)
            predictions = sent_model.predict(data)
            print(predictions[0][0])
"""


# Final_API1_function
def API1(input_keyword, level):
    input_keyword = input_keyword.lower()
    input_keyword = input_keyword.strip()
    input_keyword = input_keyword.replace("\'",'')
    input_keyword = input_keyword.replace(' ','_')
    out_list = []
    try :
        if return_similar_word_list(input_keyword, level) != []:
            kb = return_similar_word_list(input_keyword, level)
            #return_sentiment_score(kb)
            dic_name = return_similar_word_name(kb)
            dic_frequency = return_avg_frequency(kb)

            for i in range(len(dic_name)):
                out_dict = {}
                out_dict['name'] = dic_name[i].replace("_"," ")
                out_dict['frequency'] = dic_frequency[i]
                out_list.append(out_dict)

            out_list3 = sorted(out_list, key=itemgetter('frequency'), reverse = True)
            result = json.dumps(out_list3)

    except KeyError :
        out_list = ['No Result']
        result = json.dumps(out_list)

    return(result)


#######################################################################################################
#   NLP_API2 Function
#######################################################################################################
stop_words = set(stopwords.words('english'))

# Load Weight_matrix
weight_final_df = pandas.read_csv(WM_PATH, encoding = 'utf-8', skiprows = 1)
weight_final_df.index = ['food','service','ambience','value']
col_list = list(weight_final_df.columns.values)

# Return Score Function
def API2_function(input_list):
    menu_name = input_list
    food_score_list = []
    service_score_list = []
    ambience_score_list = []
    value_score_list =[]

    for i in menu_name:
        test_str = i.lower()
        test_str = test_str.replace(".","")
        test_str = test_str.replace("!","")
        test_str = test_str.strip()
        #input으로 메뉴 이름'만' 들어온다는 가정하에, 모든 ' ' -> '_'
        test_str = test_str.replace(" ","_")        
        try:
            food_score_list.append(weight_final_df[test_str][0])
        except:
            food_score_list.append(0)
        try:
            service_score_list.append(weight_final_df[test_str][1])
        except:
            service_score_list.append(0)
        try:
            ambience_score_list.append(weight_final_df[test_str][2])
        except:
            ambience_score_list.append(0)
        try:
            value_score_list.append(weight_final_df[test_str][3])
        except:
            value_score_list.append(0)
            
    menu_name_list = menu_name
    out_list = []
    for i in range(len(menu_name_list)):
        out_dict = {}
        out_dict['menu'] = menu_name_list[i]
        out_dict['price_score'] = round(value_score_list[i],4)
        out_dict['taste_score'] = round(food_score_list[i],4)
        out_dict['service_score'] = round(service_score_list[i],4)
        out_dict['ambience_score'] = round(ambience_score_list[i],4)
        out_dict['avg_score'] = round((value_score_list[i] + food_score_list[i] + service_score_list[i] + ambience_score_list[i])/4 , 4)
        out_list.append(out_dict)
    #out_json = json.dumps(out_list)
    return(out_list)

def API2(menu_list):
    test_list = menu_list.split(',')
    out_list = API2_function(test_list)
    out_before_sort = sorted(out_list, key=itemgetter('avg_score'))
    if len(out_before_sort) > 3:
        out_list_sorting = [out_before_sort[-1], out_before_sort[-2], out_before_sort[-3], out_before_sort[-4]]
    elif len(out_before_sort) == 3:
        out_list_sorting = [out_before_sort[-1], out_before_sort[-2], out_before_sort[-3]]
    elif len(out_before_sort) == 2:
        out_list_sorting = [out_before_sort[-1], out_before_sort[-2]]
    else:
        out_list_sorting = [out_before_sort[-1]]

    result = json.dumps(out_list_sorting)
    return(result)


#######################################################################################################
#   SNS API
#######################################################################################################
def sns_api(res_name):
    output_list = []
    q_select = "*"
    q_from = "[datamingo:SNS_data.test]"
    q_where = "name like" + "\'" + res_name + "\'"
    q_order = "date"
    query = "SELECT " + q_select + "\nFROM " + q_from + "\nWHERE " + q_where + "\nORDER BY " + q_order

    query_data = pandas.read_gbq(query = query, project_id = 'datamingo')

    for i in range(len(query_data)):
        tem_dict = {}
        tem_dict['date'] = query_data['date'][i].strftime("%Y-%m-%d %H:%M:%S")
        tem_dict['name'] = query_data['name'][i]
        tem_dict['instagram_follower'] = int(query_data['instagram_follow_count'][i])
        tem_dict['instagram_hashtag'] = int(query_data['instagram_hashtag_count'][i])
        tem_dict['twitter_follower'] = int(query_data['twitter_follower_count'][i])
        tem_dict['twitter_favorite'] = int(query_data['twitter_favorite_count'][i])
        tem_dict['twitter_friend'] = int(query_data['twitter_friend_count'][i])
        tem_dict['twitter_list'] = int(query_data['twitter_listed_count'][i])
        tem_dict['youtube_comment'] = int(query_data['youtube_comment_count'][i])
        tem_dict['youtube_subscriber'] = int(query_data['youtube_subscriber_count'][i])
        tem_dict['youtube_video'] = int(query_data['youtube_video_count'][i])
        tem_dict['youtube_view'] = int(query_data['youtube_view_count'][i])
        tem_dict['wiki_pageview'] = 0
        output_list.append(tem_dict)

    result = json.dumps(output_list)
    return result


#######################################################################################################
#   SNS _wiki API
#######################################################################################################
def wiki_api(keyword, start, end, agent = 'user'):
    output_list = []
    
    p = PageviewsClient('what is it..?') #parameter로 스트링이 들어가야함. 아무거나 넣어도 가능..
    output_dict = dict(p.article_views('en.wikipedia.org', [keyword], start = start, end = end, agent = agent))
    
    for key,val in output_dict.items():
        tem_dict = {}
        tem_dict['date'] = key.strftime("%Y%m%d")
        tem_dict['view_count'] = val[keyword.replace(" ","_")]
        output_list.append(tem_dict)
    
    result = json.dumps(output_list)
    return result

#######################################################################################################
#   BigQuery
#######################################################################################################
def searchRestaurantsInformation(city, cate):
    what = ['name',
                      'menuName',
                      'menuDescription',
                      'menuBasePrice',
                      'zip',
                      'Phone',
                      'latitude',
                      'longitude',
                      'logoUrl']
    ins = ResInfoLoader('datamingo', 'company', 'us_restaurants', what)
    ret = ins.query_information_from_us_restaurants(city, cate)
    return ret

def searchCityList():
    what = ['city']
    ins = ResCityLoader('datamingo', 'company', 'us_city', what)
    ret = ins.query_city_from_us_restaurants()
    return ret

def searchMenuCategoryList():
    what = ['menuCategory']
    ins = ResMenuCateLoader('datamingo', 'company', 'us_menuCategory', what)
    ret = ins.query_city_from_us_restaurants()
    return ret

def searchCompanyList(city, menuCete):
    what = ['DISTINCT name']
    ins = ResInfoLoader('datamingo', 'company', 'us_restaurants', what)
    ret = ins.query_information_from_us_restaurants(city, menuCete)
    return ret

#######################################################################################################
# 리퀘스트 id 로 파라미터 보내기
#######################################################################################################
result = "{}"
#keyword search trend : bigmac & big_mac -> big mac 전처리 추가.
#best seller : ''

def index(request):
    # NLP API
    if request.GET["id"] == "nlp1" :
        level = request.GET["level"]
        test_str = request.GET["keyword"]
        print("API1_Starts : " , datetime.datetime.now().time(), flush = True)
        result = API1(test_str, level)
        print("API1_Ends   : " , datetime.datetime.now().time() , " : " , result, flush = True)

    elif request.GET["id"] == "nlp2" :
        test_str = request.GET["menu_list"]
        print("API2_Starts : " , datetime.datetime.now().time(), flush = True)
        result = API2(test_str)
        print("API2_Ends   : " , datetime.datetime.now().time() , " : " , result, flush = True)

    # SNS API
    elif request.GET["id"] == "sns":
        res_name = request.GET['res_name']
        res_name = res_name.lower()
        res_name = res_name.replace(",","")
        res_name = res_name.replace(".","")
        res_name = res_name.replace("-","")
        res_name = res_name.replace(" ","")
        res_name = res_name.replace("\'","")
        res_name = res_name.replace("’","")
        result = sns_api(res_name)

    # SNS_WIKI API
    elif request.GET["id"] == "sns_wiki":
        res_name = request.GET['res_name']
        start = str(request.GET['start'])
        end = str(request.GET['end'])
        result = wiki_api(res_name, start, end)
    # bq
    elif request.GET["id"] == "resInfo":
        print(request.GET['city'])
        print(request.GET['menuCate'])
        result = searchRestaurantsInformation(request.GET['city'], request.GET['menuCate'])
    elif request.GET["id"] == "resCityList":
        result = searchCityList()
    elif request.GET["id"] == "resMenuCateList":
        result = searchMenuCategoryList()
    elif request.GET["id"] == "resList":
        result = searchCompanyList(request.GET['city'], request.GET['menuCate'])
            
    return HttpResponse(result)

# http://localhost:8000/api/?id=resInfo&city=Omaha&menuCate=Appetizers