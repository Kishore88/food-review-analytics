import pandas as pd
import os, sys, getopt, cPickle, csv, sklearn
import os.path
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
# from django.core.files.storage import FileSystemStorage
import json, re
import numpy as np
import itertools
import string
from itertools import groupby
from random import randint
import glob
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.util import ngrams
from werkzeug import secure_filename
import sentiment
from nltk.corpus import stopwords
import boto3
import boto3.session
import uuid
import time
import logging
import config

# @author  Kishore Suthar
# @version  1.0
# contains all the functions for upload dataset, train the dataset and classify the review

# constants
DATASET = 'dataset'
PROBABILITY = 'probability'
MODEL_DATA = 'model_data'
CATEGORY = 'Category'
SUB_CATEGORY = 'Sub_Category'
CSV = 'csv'
CLASSIFY = 'classify'
TEXT = 'Text'
LABEL = 'label'
PKL = '.pkl'
BLANK = 'blank'
NO_CATEGORY = 'No category'
NO_LABEL = 'No Label'
MODELS = 'models'
SENTIMENT = 'sentiment'
NEUTRAL = 'neutral'
POSITIVE = 'positive'
NEGATIVE = 'negative'
POSITIVE_TAG = 'positive tag'
NEGATIVE_TAG = 'negative tag'
OVERALL = 'Overall'
NO_MODEL_EXIST = 'NO MODEL EXIST'

# stop words, list of operator words that you take out of the stopword list
operators = set(("on","above","after","again","against","no","any","aren't","before","below","but","can't","cannot","could","couldn't","didn't","doesn't","don't","down","hadn't","hasn't","haven't","isn't","itself","let's","more","most","mustn't","no","nor","not","off","only","out","over","shouldn't","so","some","too","under","until","up","very","wasn't","weren't","why","why's","with","won't","wouldn't"))
cachestopwords = set(stopwords.words('english')) - operators

# allowed extension for input training set. right now model will support only .csv file
ALLOWED_EXTENSIONS = set(['csv'])

# global variable for root directory
MODEL_PATH =os.path.join(os.getcwd(), MODELS)

# This function is classify the review based on training model.It convert the whole review into sentences and classify each sentences.
# It also have dynamic property to train the model. If model is not trained, it will train the model at runtime
# @param review
# @param classify id
# @return classified sentences
def main(review):
  # create list to store postive and negative category tag
  positiveList = []
  negativeList = []
  # calculate overall sentiment of review
  overall_sentiment = classify_overall_sentiment(review,False)
 
  review = ' '.join([word for word in review.split() if word not in cachestopwords])
  review = re.sub('[.]+','.', review);
  review = re.sub('[!?]','', review).replace('.','. ').strip()
  try:
    # It will convert the whole review into sentences by using ntlk
    sent_text = nltk.sent_tokenize(review)
    for sentence in sent_text:
      break_sen = re.split(r'\s(?=(?:aside from|but|however|nevertheless|yet|though|although|still|except|apart from|because|unless|therefore)\b)',sentence,flags=re.IGNORECASE)
      for sen in break_sen : 
         polarity = classify_sentiment(sen)
         if polarity == NO_MODEL_EXIST :
            return {'status':'failed, no model exist. Please train the model'}
         result = classify(polarity, sen) 
         if result[LABEL] != NO_LABEL: 
            if polarity == NEGATIVE and result[LABEL] != NO_CATEGORY:
               negativeList.append(result[LABEL])
            elif polarity == POSITIVE:
               positiveList.append(result[LABEL])
         else :
            return result
  except Exception, e:
    logging.exception(e.message)
    pass
  if OVERALL in positiveList and len(negativeList) > 0:
      positiveList.remove(OVERALL) 
  if overall_sentiment[LABEL] == POSITIVE and len(positiveList) == 0 and len(negativeList) > 0:
     return overall_sentiment, {POSITIVE_TAG:[],NEGATIVE_TAG:[]}  
  elif overall_sentiment[LABEL] == NEGATIVE and len(negativeList) == 0 and len(positiveList) > 0:
     return overall_sentiment, {POSITIVE_TAG:[],NEGATIVE_TAG:[]}  
  else :   
     return overall_sentiment, {POSITIVE_TAG:list(set(positiveList)),NEGATIVE_TAG:list(set(negativeList))}  

# This function is used classify the sentence  
# @param polarity
# @param sentence
# @return prediction
def classify(polarity, sentence) : 
   if polarity == NEUTRAL :
      polarity = NEGATIVE  
   non_alphanumeric_cleaning = re.sub(ur"[^\w\d'\s]+", "", sentence)
   # positive model .pk file for predict positive classification category     
   pos_file = os.path.join(MODEL_PATH,POSITIVE + '_' + MODEL_DATA + PKL)
   # negative model .pk file for predict negative classification category     
   neg_file = os.path.join(MODEL_PATH,NEGATIVE + '_' + MODEL_DATA + PKL)

   # Check whether model is trained or not
   if(os.path.exists(pos_file) == False and os.path.exists(neg_file) == False) :
      return {PROBABILITY : 0, LABEL : NO_LABEL , 'status':'failed, no model exist. Please train the model' }
   # if the review is not related to given category, it will tag 'No Category' and filter from the output
   try:
      filepath = os.path.join(MODEL_PATH, polarity + '_' + MODEL_DATA + PKL)
      prediction = predict(non_alphanumeric_cleaning, filepath)
   except Exception, e:
     logging.exception(e.message)
   return prediction  

# This function is used classify the sentiment  
# @param sentence
# @return sentiment
def classify_sentiment(sentence) :
  # create list to store multiple polarity
  negative = []
  positive = []
  # dictionary method
  polarity = sentiment.sentiment(sentence)
  # print "dictionary method->    "+polarity
  if polarity == POSITIVE:
     positive.append(POSITIVE)
  elif polarity == NEGATIVE:
     negative.append(NEGATIVE)
  non_alphanumeric_cleaning = re.sub(ur"[^\w\d'\s]+", "", sentence)
  li = ['naive_bayes','svm']
  for model in li: 
    try:
      filepath = os.path.join(MODEL_PATH,str(model) + PKL)
      # Check whether model is trained or not
      if(os.path.exists(filepath) == False ) :
         return NO_MODEL_EXIST
      prediction = predict_sentiment(non_alphanumeric_cleaning, filepath)
      if prediction[LABEL] == NEGATIVE:
         negative.append(NEGATIVE)
      elif prediction[LABEL] == POSITIVE:
         positive.append(POSITIVE)
    except Exception as e:
      logging.exception(e.message)
      pass
  if len(negative) > 1 :
     return NEGATIVE
  elif len(positive) > 1 :
     return POSITIVE
  else :         
    return NEUTRAL

def remove_punctuation(s):
    table = string.maketrans("","")
    return s.translate(table, string.punctuation.replace("'",""))


# This function is used classify the sentiment  
# @param sentence
# @return sentiment
def classify_overall_sentiment(sentence,clean) :
  # if the stopword already remove from sentence 
  if clean :
     sentence = ' '.join([word for word in sentence.split() if word not in cachestopwords])
  tagged_sentence = nltk.tag.pos_tag(sentence.split())
  edited_sentence = [word for word,tag in tagged_sentence if tag != 'CD' and tag != 'PRP$' and tag != 'PRP' and tag != 'TO' and tag != 'WRB' and tag != 'WDT' and tag != 'DT' and tag != 'NNP' and tag != 'NNPS' and tag != 'IN' and tag != 'CC']
  x = (' '.join(edited_sentence))
  non_alphanumeric_cleaning = re.sub(ur"[^\w\d'\s]+", "", x)
  try:
    filepath = os.path.join(MODEL_PATH, 'overall_sentiment_naive_bayes' + PKL)
    # Check whether model is trained or not
    if(os.path.exists(filepath) == False ) :
       return NO_MODEL_EXIST
    return predict(non_alphanumeric_cleaning, filepath)
  except Exception as e:
    logging.exception(e.message)
    pass
    
# This function is basically used for to train the data model. The function used logistic regression alogrithm
# for traing the model.
# @param dataframe
# @param level (ex - for category level, sub_category level or sub_sub_category level) 
# @param filepath (filepath where the model with stored after creating)
# @return pickle file (trained model)
def train_logisticregression(s3client, datas, level, filepath):
  if(os.path.isfile(filepath) == True):
     os.remove(filepath)
  print 'Creating Logistic Regression Model for Categories.....'

  # remove the additional fields from dataframe
  column_list = [TEXT,CATEGORY,SUB_CATEGORY]
  datas = datas[column_list]
  # remove the empty row from dataframe
  datas = datas.dropna()
  print len(datas.index)

  datas[TEXT] = datas[TEXT].apply(remove_punctuation)
  # split dataset for cross validation
  msg_train, msg_test, label_train, label_test = train_test_split(datas[TEXT], datas[level], test_size=0.1)

  pipeline = Pipeline([
       ('bow', CountVectorizer(ngram_range=(1, 3), max_features = 30000, lowercase=True)),  # strings to token integer counts
       ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
       ('classifier', LogisticRegression(C=10, class_weight=None, dual=False, fit_intercept=True, intercept_scaling=1, max_iter=100, multi_class='multinomial', n_jobs=1, penalty='l2', random_state=None, solver='lbfgs', tol=0.0001, verbose=0, warm_start=False)),  # train on TF-IDF vectors w/ Naive Bayes classifier
   ])
  # pipeline parameters to automatically explore and tune
  params = {
  'tfidf__use_idf': (True, False)
  }
  grid = GridSearchCV(
      pipeline,
      params, # parameters to tune via cross validation
      refit=True, # fit using all data, on the best detected classifier
      n_jobs=-1,
      scoring='accuracy',
      cv=StratifiedKFold(label_train, n_folds=5),
  )
  # train
  nb_detector = grid.fit(msg_train, label_train)
  # print("Best score: %0.3f" % grid.best_score_)
  # print("Best parameters set:")
  # best_parameters = grid.best_estimator_.get_params()
  # for param_name in sorted(params.keys()):
  #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
  print ''
  predictions = nb_detector.predict(msg_test)
  print ':: Confusion Matrix'
  print ''
  print confusion_matrix(label_test, predictions)
  print ''
  print ':: Classification Report'
  print ''
  print classification_report(label_test, predictions)
  print('Classifier accuracy percent:',(accuracy_score(predictions, label_test)))
  # save model to pickle file

  file_name = filepath
  with open(file_name, 'wb') as fout:
      cPickle.dump(nb_detector, fout)
  print 'model written to: ' + file_name
  s3client.delete_object(Bucket=config.bucket, Key=os.path.join(config.models_dir,os.path.basename(filepath)))
  s3client.upload_file(filepath,config.bucket,os.path.join(config.models_dir,os.path.basename(filepath)))

def svm_model(s3client, datas, level, filepath):
  
  if(os.path.isfile(filepath) == True):
     os.remove(filepath)
  # remove the additional fields from dataframe
  column_list = [TEXT,'Sentiment']
  datas = datas[column_list]
  # remove the empty row from dataframe
  datas = datas.dropna()
  df = datas.dropna();
  print len(df.index)

  datas[TEXT] = datas[TEXT].apply(remove_punctuation)
  msg_train, msg_test, label_train, label_test = train_test_split(datas[TEXT], datas[level], test_size=0.2)
  #create pipeline  
  print 'creating svm sentiment model....'  
  pipeline = Pipeline([('bow', CountVectorizer(ngram_range=(1, 3), max_features = 30000)),('tfidf', TfidfTransformer()),('classifier', SVC())])
      # pipeline parameters to automatically explore and tune
  params = [
      {'classifier__C': [1, 10, 100, 1000], 'classifier__gamma': [0.001, 0.0001], 'classifier__kernel': ['rbf', 'poly']},
  ]
  grid = GridSearchCV(
     pipeline,
     param_grid=params, # parameters to tune via cross validation
     refit=True, # fit using all data, on the best detected classifier
     n_jobs=-1,
     scoring='accuracy',
     cv=StratifiedKFold(label_train, n_folds=5),
  )  

  # train
  svm_detector = grid.fit(msg_train, label_train)
  # print("Best score: %0.3f" % grid.best_score_)
  # print("Best parameters set:")
  # best_parameters = grid.best_estimator_.get_params()
  # for param_name in sorted(params.keys()):
  #     print("\t%s: %r" % (param_name, best_parameters[param_name]))
  print ""
  print ":: Confusion Matrix"
  print ""
  print confusion_matrix(label_test, svm_detector.predict(msg_test))
  print ""
  print ":: Classification Report"
  print ""
  print classification_report(label_test, svm_detector.predict(msg_test))
    # save model to pickle file
  with open(filepath, 'wb') as fout:
      cPickle.dump(svm_detector, fout)
  print 'model written to: ' + filepath
  s3client.delete_object(Bucket=config.bucket, Key=os.path.join(config.models_dir,os.path.basename(filepath)))
  s3client.upload_file(filepath,config.bucket,os.path.join(config.models_dir,os.path.basename(filepath)))


def multinomial_nb_model(s3client, datas, level, filepath):
  if(os.path.isfile(filepath) == True):
     os.remove(filepath)
  print os.path.join(config.models_dir,os.path.basename(filepath))
  print 'creating naive bayes sentiment model...'
  # remove the additional fields from dataframe
  column_list = [TEXT,'Sentiment']
  datas = datas[column_list]
  # remove the empty row from dataframe
  datas = datas.dropna()
  df = datas.dropna();
  print len(df.index)

  datas[TEXT] = datas[TEXT].apply(remove_punctuation)
  msg_train, msg_test, label_train, label_test = train_test_split(datas[TEXT], datas[level], test_size=0.2)
    
  # create pipeline
  pipeline = Pipeline([('bow', CountVectorizer(ngram_range=(1, 3), max_features = 30000)), ('tfidf', TfidfTransformer()), ('classifier', MultinomialNB())])
  # pipeline parameters to automatically explore and tune
  params = {
            'tfidf__use_idf': (True, False),
            'classifier__alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
            'tfidf__sublinear_tf': (True, False),
  }
  grid = GridSearchCV(
      pipeline,
      params, # parameters to tune via cross validation
      refit=True, # fit using all data, on the best detected classifier
      n_jobs=-1,
      scoring='accuracy',
      cv=StratifiedKFold(label_train, n_folds=5),
  )
      # train
  nb_detector = grid.fit(msg_train, label_train)
  print ""
  predictions = nb_detector.predict(msg_test)
  print ":: Confusion Matrix"
  print ""
  print confusion_matrix(label_test, predictions)
  print ""
  print ":: Classification Report"
  print ""
  print classification_report(label_test, predictions)
      # save model to pickle file
  with open(filepath, 'wb') as fout:
      cPickle.dump(nb_detector, fout)
  print 'model written to: ' +  filepath
  s3client.delete_object(Bucket=config.bucket, Key=os.path.join(config.models_dir,os.path.basename(filepath)))
  s3client.upload_file(filepath,config.bucket,os.path.join(config.models_dir,os.path.basename(filepath)))


# This function  used for allow file for upload ex .csv, .txt 
# @ param file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def getS3Client() :
    return boto3.client('s3', config= boto3.session.Config(signature_version=config.signature_version),aws_access_key_id=config.aws_access_key_id,
         aws_secret_access_key=config.aws_secret_access_key)
    

# This function  used for upload file on server. 
# @ param file
def upload(file, model_type):
  s3client = getS3Client() 
  if s3client is None : 
    return {"msg":"invalid credential or signature version"}
  # 
  bucket_name = config.bucket
  print file.filename
  try:
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
       new_file = os.path.splitext(file.filename)[0]+"_"+str(time.strftime("%Y%m%d_%H%M%S"))+".csv"
       filepath = os.path.join(os.getcwd()+"/dataset", new_file)  
       # check the header of the uploaded file
       if model_type == 'sentiment' :
            try:
              file.save(filepath)
              columns = list(pd.read_csv(filepath))
              if columns[0] == TEXT and columns[1] =='Sentiment':
                 # Uploads the given file using a managed uploader, which will split up large
                 # files automatically and upload parts in parallel.
                 s3client.upload_file(filepath, bucket_name, 'sentiment-data/{}'.format(new_file))
                 os.remove(filepath)
                 return {'status': 'success, file uploaded successfully'}  
              else:
                 os.remove(filepath)
                 return {'status': 'failed, File header not matched. Header name should be [Text, Sentiment]'}          
            except Exception as e:
              os.remove(filepath)
              return {'status': 'failed, File header not matched. Header name should be [Text, Sentiment]'}             
       elif model_type == 'classification' :
            try:
              file.save(filepath)
              columns = list(pd.read_csv(filepath))
              if columns[0] == TEXT and columns[1] == 'Sentiment' and columns[2] == CATEGORY and columns[3] == SUB_CATEGORY :
                  # Uploads the given file using a managed uploader, which will split up large
                  # files automatically and upload parts in parallel.
                  s3client.upload_file(filepath, bucket_name, 'classification-data/{}'.format(new_file))
                  os.remove(filepath)
                  return {'status': 'success, file uploaded successfully'}
              else:
                  os.remove(filepath)
                  return {'status': 'failed, File header not matched. Header name should be [Text, Sentiment, Category, Sub_Category]'}
            except Exception as e:
              os.remove(filepath)
              return {'status': 'failed, File header not matched. Header name should be [Text, Sentiment, Category, Sub_Category]'}           
       else:
         return {'status': 'failed, wrong file format'}
  except Exception as e:
    logging.exception(e.message)

# This function is used for to train the data model
# @param tagged dataset file in .csv format
# @return cPickle file (trained model)
def train(model_type) :
  s3client = getS3Client() 
  if s3client is None : 
    return {"status":"failed, invalid credential or signature version"}
  if model_type == 'sentiment' :
    try:
      list_dataframe = getDataframe(s3client, model_type)  
      if list_dataframe :       
        df = pd.concat(list_dataframe)
        df_overall_sentiment = df.dropna()
        path_nb = os.path.join(MODEL_PATH,'overall_sentiment_naive_bayes'+ PKL)
        multinomial_nb_model(s3client,df_overall_sentiment, 'Sentiment', path_nb)
        return {'status':'model trained'}
      else :
        return {'status':'failed, not file found'}  
    except Exception as e:
      logging.exception(e.message)
      return {'status':'failed, '+e.message}
  elif model_type == 'classification' :
    try:
      list_dataframe = getDataframe(s3client, model_type) 
      if list_dataframe :       
        df = pd.concat(list_dataframe)
        SENTIMENT = 'Sentiment'

        # path_nb = os.path.join(MODEL_PATH,'naive_bayes'+ PKL)
        # multinomial_nb_model(s3client, df, 'Sentiment', path_nb)
      
        path_svm = os.path.join(MODEL_PATH,'svm'+ PKL)
        svm_model(s3client,df, SENTIMENT, path_svm)

        df_neg = df[(df[SENTIMENT] == NEGATIVE) | (df[SENTIMENT] == NEUTRAL)] 
        path_neg = os.path.join(MODEL_PATH,NEGATIVE + '_' + MODEL_DATA + PKL)
        train_logisticregression(s3client,df_neg, CATEGORY, path_neg)
        
        df_pos = df[(df[SENTIMENT] == POSITIVE)]
        path_pos = os.path.join(MODEL_PATH,POSITIVE + '_' + MODEL_DATA + PKL)
        train_logisticregression(s3client,df_pos, CATEGORY, path_pos) 
        return {'status':'model trained'}
      else :
        return {'status':'failed, no file found'}  
    except Exception as e:
      return {'status':'failed, ' + e.message}
  else :
    return {'status':'failed, wrong model_type param'}

# This function is used get all the csv file from s3 specific directory into dataframe . 
# @ param s3 connection
# @ param model_type
# @ return dataframe
def getDataframe(s3client, model_type) :
  df = pd.DataFrame()
  list_dataframe = []
  for key in s3client.list_objects(Bucket=config.bucket)['Contents']:
    print key['Key']
    if key['Key'].find(model_type) != -1 and  key['Key'].find(".csv") != -1:
        try:
          response = s3client.get_object(Bucket=config.bucket, Key=key['Key'])
          chunkDataframe = pd.read_csv(response['Body'],header=0, chunksize=1000)
          dataframe = pd.concat(chunkDataframe, ignore_index=True)
          clean_df = dataframe.dropna(subset=[TEXT], how='all')
          clean_df[TEXT] = clean_df[TEXT].apply(lambda x : ' '.join([item for item in string.split(x.lower()) if item not in cachestopwords]))
          list_dataframe.append(clean_df)
        except Exception as e:
          logging.exception(e.message)
    else :
       continue;
  return list_dataframe     

# This function is used for predict the review category based on training model. 
# @ param review
# @ model file (.pkl)
# @ return predicted category with probability 
def predict(message, filepath):
  nb_detector = cPickle.load(open(filepath))
  # calculate predicted category for given message
  nb_predict = nb_detector.predict([message])[0]
  # calculate probabilty for predicted category for given message
  nb_perc = max(nb_detector.predict_proba([message])[0])
  return {PROBABILITY : round(nb_perc,5), LABEL : nb_predict }

# This function is used for predict the review category based on training model. 
# @ param review
# @ model file (.pkl)
# @ return predicted category with probability 
def predict_sentiment(message, filepath):
  nb_detector = cPickle.load(open(filepath))
  # calculate predicted category for given message
  nb_predict = nb_detector.predict([message])[0]
  # calculate probabilty for predicted category for given message
  #nb_perc = max(nb_detector.predict_proba([message])[0])
  return {LABEL : nb_predict }

# This function is used for download the trained models from s3. 
# @ models (.pkl)
def download():
  try:
    s3client = getS3Client()
    for key in s3client.list_objects(Bucket=config.bucket)['Contents']:
      if key['Key'].find("models") != -1 and  key['Key'].find(".pkl") != -1:
         s3client.download_file(config.bucket, key['Key'], os.path.join(MODEL_PATH,os.path.basename(key['Key'])))
      else :
         continue;
    return {'status' : 'models have been downloaded to local directory'}           
  except Exception as e:
      return {'status' : 'failed, '+e.message}
  
  
