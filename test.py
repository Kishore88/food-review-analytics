from flask.ext.testing import TestCase
from flask import Flask, render_template, request, jsonify, redirect, url_for,  send_from_directory, jsonify
from flask.ext.iniconfig import INIConfig
from werkzeug import secure_filename
import model
import json, os
import run
from run import application
import unittest
import model


class FlaskTestCase(unittest.TestCase):
    def test_classify(self):
        tester = application.test_client(self)
        response = tester.post('/classifier/classify', data=json.dumps({"data":"Food  is very nice"}), content_type='application/json')
        json_response = json.loads(response.data)
        self.assertEqual(200, json_response['status_code'])
        self.assertNotEqual('None', json_response['body']['data']) 

    def test_sentiment(self):
        tester = application.test_client(self)
        response = tester.post('/classifier/sentiment', data=json.dumps({"data":"Food  is very nice"}), content_type='application/json')
        json_response = json.loads(response.data)
        self.assertEqual(200, json_response['status_code'])
        self.assertNotEqual('None', json_response['body']['sentiment'])  

    def test_downloadModelsFromS3(self):
        tester = application.test_client(self)
        response = tester.post('/classifier/download')
        json_response = json.loads(response.data)
        self.assertEqual(200, json_response['status_code'])
        self.assertEqual('models have been downloaded to local directory', json_response['body']['status'])   
"""
  uncomment the below function if you want to test the train.
  It will train all the model and upload on S3. 
"""   

    # def test_train(self):
    #     tester = application.test_client(self)
    #     response = tester.get('/classifier/train')
    #     json_response = json.loads(response.data)
    #     self.assertEqual(200, json_response['status_code']) 

if __name__ == '__main__':
    unittest.main()  