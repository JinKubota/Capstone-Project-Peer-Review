import os
import sys
import json
import time
import unittest
import inspect 
from logopr import get_logger
from dataFetchClean import fetch_data, correct_input
from model import model_compare, model_predict_all, model_predict_slr, model_predict_mlr, model_predict_plr

config_path = "C:/work_product/python_al/capstone/config/"
config_file = "config.json"
with open(config_path+config_file) as file_config:
    fconfig = json.load(file_config) 
    
LOG_DIR = fconfig["logdir"]
LOGGER = get_logger(LOG_DIR, __name__)
LOGGER.info('[init] : Starting modelTest.py') 

logger = get_logger(LOG_DIR, __name__)
fn_name_from_inspect = inspect.stack()[0][3]
      
class ModelTest(unittest.TestCase): 

    def test_input_data_load(self): 
        outmsg = fetch_data(logger, fconfig["inputdata"], fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)
    
    def test_input_data_cleanup(self): 
        outmsg = correct_input(logger, fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)    
            
    def test_model_predict_slr(self):        
        outmsg = model_predict_slr(logger, fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)
    
    def test_model_predict_mlr(self):
        outmsg = model_predict_mlr(logger, fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)
            
    def test_model_predict_plr(self):
        outmsg = model_predict_mlr(logger, fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)        

    def test_model_predict_all(self):
        reg_type = "ALL"
        outmsg = model_predict_all(logger, fconfig["processdata"], reg_type)
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)
   
    def test_model_compare(self): 
        outmsg = model_compare(logger, fconfig["processdata"])
        if "errorType" in outmsg.keys():            
            self.assertTrue(False)
        elif "isSuccess" in outmsg.keys() and outmsg.get("isSuccess"):
            self.assertTrue(True)
     
if __name__ == '__main__': 
    unittest.main() 
