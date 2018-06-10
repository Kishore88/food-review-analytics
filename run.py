#! /usr/bin/env python
from flask import Flask, render_template, request, jsonify, redirect, url_for,  send_from_directory
from flask.ext.iniconfig import INIConfig
from werkzeug import secure_filename
import model
import json, os, time

application = Flask(__name__)

# app.config.from_inifile(app.instance_path + '/config.ini')

# this service use for validating the review
@application.route('/testing', methods=['GET'])
def classify():
        return render_template('classify.html')

# service use for category classify the review
@application.route('/classifier/classify', methods=['POST'])
def classify_review():
    starttime = time.time()
    response = None
    if request.method == 'POST' :
        try:
            if request.get_json() :
                req_data = request.get_json()
                overall_sentiment,return_data = model.main(req_data['data'])
                if not return_data :
                    response = {'review': req_data['data'], 'data':'None'}    
                else :
                    response = {'review': req_data['data'], 'sentiment':overall_sentiment,'data':return_data}
                endtime = time.time()
                print "Time taken for the call: " + str((endtime-starttime)) + " ms"
            else :
               response = {'status':'failed , invalid request'} 
        except Exception as e:
            return respond(e)
    else :
        response ={'status':'failed'}   
    return respond(None, res=response)         

# service use for sentiment of the review
@application.route('/classifier/sentiment', methods=['POST'])
def sentiment():
    starttime = time.time()
    response = None
    if request.method == 'POST' :
        try:
            if request.get_json() :
                req_data = request.get_json()
                sentiment_data = model.classify_overall_sentiment(req_data['data'],True)
                if not sentiment_data:
                    response = {'review': req_data['data'], 'sentiment':'None'}    
                else :
                    if sentiment_data == 'NO MODEL EXIST':
                       response = {'status':'failed, no model exist. Please train the model'}    
                    else :
                       response = {'review': req_data['data'], 'sentiment':sentiment_data}                    
                endtime = time.time()
                print "Time taken for the call: " + str((endtime-starttime)) + " ms"
            else :
               response = {'status':'failed , invalid request'} 
        except Exception as e:
            return respond(e)
    else :
        response ={'status':'failed'}        
    return respond(None, res=response)

# # Route that will process the file upload
@application.route('/classifier/upload', methods=['POST'])
def upload():
    starttime = time.time()
    response = None
    if request.method == 'POST' :
        try:
            file_type = request.args.get('file_type')
            if file_type :
                response = model.upload(request.files['_file'], file_type)
                endtime = time.time()
                print "Time taken for the call: " + str((endtime-starttime)) + " ms"
            else :
                response = {"status":"argument file_type missing"}
        except Exception as e:
            return respond(e)
    else :
        response ={'status':'failed'}
    return respond(None, res=response)

# Route that will process the data training
@application.route('/classifier/train', methods=['GET'])
def train():
    starttime = time.time()
    response = None
    if request.method == 'GET' :
        try:
            model_type = request.args.get('model_type')
            if model_type :
               response = model.train(model_type)
               endtime = time.time()
               print "Time taken for the call: " + str((endtime-starttime)) + " ms"
            else :
               response = {"status":"argument model_type missing"}   
        except Exception as e:
            return respond(e)
    else :
        response ={'status':'failed'}    
    return respond(None, res=response)

@application.route('/classifier/download', methods=['POST'])
def downloadModelsFromS3() :
    response = None
    if request.method == 'POST' :
        try:
            response = model.download()
        except Exception as e:
            return respond(e)    
    else :
        response ={'status':'failed'} 
    return respond(None, res=response)

def respond(err, res=None):
    return_res =  {
        'status_code': 400 if err else 200,
        'body': err.message if err else res,
    }
    return jsonify(return_res)


# start the server with the 'run()' method
if __name__ == '__main__':
    application.run(host='0.0.0.0')