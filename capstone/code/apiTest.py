import os
import sys
import json
import time
import inspect 
import unittest
import requests

BASE_URL = "http://127.0.0.1"
STATUS_CODE = 200
port = 80

try:
    requests.get(BASE_URL.format(port))
    server_available = True
except:
    server_available = False
      
class APITest(unittest.TestCase): 

    @unittest.skipUnless(server_available,"local server is not running")
    def test_home(self):
        relative_url= "/"
        api_url = BASE_URL + relative_url
        resp = requests.get(api_url)              
        self.assertEqual(resp.status_code, STATUS_CODE)
    
    
    @unittest.skipUnless(server_available,"local server is not running")    
    def test_loadData(self):
        relative_url= "/loadData"
        api_url = BASE_URL + relative_url
        resp = requests.get(api_url)              
        self.assertEqual(resp.status_code, STATUS_CODE)    
       
    @unittest.skipUnless(server_available,"local server is not running")    
    def test_cleanData(self):
        relative_url= "/cleanData"
        api_url = BASE_URL + relative_url
        resp = requests.get(api_url)              
        self.assertEqual(resp.status_code, STATUS_CODE) 
    
        
    @unittest.skipUnless(server_available,"local server is not running")
    def test_modelPredictAll(self):
        relative_url= "/modelPredictAll"
        api_url = BASE_URL + relative_url
        resp = requests.get(api_url)              
        self.assertEqual(resp.status_code, STATUS_CODE) 
     
    
    @unittest.skipUnless(server_available,"local server is not running")
    def test_modelCompare(self):
        relative_url= "/modelCompare"
        api_url = BASE_URL + relative_url
        resp = requests.get(api_url)              
        self.assertEqual(resp.status_code, STATUS_CODE) 
    
    
if __name__ == '__main__': 
    unittest.main() 
