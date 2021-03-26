import os
import sys
import json
import time
import re
import datetime
import pandas as pd
import numpy as np
import logging
import inspect
from logopr import get_logger
from flask import Flask, jsonify
from dataFetchClean import fetch_data, correct_input
import model

app = Flask(__name__)

config_path = "C:/work_product/python_al/capstone/config/"
config_file = "config.json"
with open(config_path+config_file) as file_config:
    fconfig = json.load(file_config) 
    
LOG_DIR = fconfig["logdir"]
LOGGER = get_logger(LOG_DIR, __name__)
LOGGER.info('[init] : Starting app.py')    

@app.route("/")
def index():
    return "hello"


@app.route("/loadData")    
def loadData():
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "...fetching data"]))
    try: 
        outmsg = fetch_data(logger, fconfig["inputdata"], fconfig["processdata"]) 
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                                 "Data load failed due to...", str(outmsg["errorOn"])]))
            raise Exception
        if "isSuccess" in outmsg.keys() and "noOfRecord" in outmsg.keys() and outmsg.get("isSuccess"):    
            logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                                 "Data load successful : record count :", str(outmsg["noOfRecord"])]))
        output = "Data load successful. Record count = " + str(outmsg["noOfRecord"])                     
        return output
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error

        
@app.route("/cleanData")    
def cleanData():    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......correct / clean input "]))
    try:        
        outmsg = correct_input(logger, fconfig["processdata"])  
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                                 "Data cleanup failed due to...", str(outmsg["errorOn"])]))
            raise Exception
        if "isSuccess" in outmsg.keys() and "noOfRecord" in outmsg.keys() and outmsg.get("isSuccess"):    
            logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                                 "Data cleanup successful : record count :", str(outmsg["noOfRecord"])]))
        output = "Data cleaned successful. Record count = " + str(outmsg["noOfRecord"])                     
        return output
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error
        
        
@app.route("/modelPredictSlr")    
def modelPredictSlr():    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......model modelPredictSlr"]))
    try:
        ret_str = "No simple linear regression provided"
        output = model.model_predict_slr(logger, fconfig["processdata"]) 
        if "errorType" in output.keys() and "errorOn" in output.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_predict_slr due to.. : ", str(output["errorOn"])])) 
            raise Exception
        if "isSuccess" in output.keys() and output.get("isSuccess"):
            if "Yhat" in output.keys() and  "intercept"  in output.keys() and "slope"  in output.keys():
                ret_str = ''' 
                    Simple Linear Regression Model =>
                            Yhat : {}
                            Intercept : {}
                            Slope : {} 
                    '''.format(str(output["Yhat"]), str(output["intercept"]), str(output["slope"]))
        return ret_str
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error


@app.route("/modelPredictMlr")    
def modelPredictMlr():    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......model modelPredictMlr"]))
    try:
        ret_str = "No multiple linear regression provided"
        output = model.model_predict_mlr(logger, fconfig["processdata"]) 
        if "errorType" in output.keys() and "errorOn" in output.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_predict_mlr due to.. : ", str(output["errorOn"])])) 
            raise Exception
        if "isSuccess" in output.keys() and output.get("isSuccess"):
            if "Yhat" in output.keys() and  "intercept"  in output.keys() and "slope"  in output.keys():
                ret_str = ''' 
                    Multiple Linear Regression Model =>
                            Yhat : {}
                            Intercept : {}
                            Slope : {} 
                    '''.format(str(output["Yhat"]), str(output["intercept"]), str(output["slope"]))
        return ret_str
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error


@app.route("/modelPredictPlr") 
def modelPredictPlr():    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......model modelPredictPlr"]))
    try:
        ret_str = "No polynimial regression provided"
        output = model.model_predict_plr(logger, fconfig["processdata"]) 
        if "errorType" in output.keys() and "errorOn" in output.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_predict_plr due to.. : ", str(output["errorOn"])])) 
            raise Exception
        if "isSuccess" in output.keys() and output.get("isSuccess"):
            if "p" in output.keys():
                ret_str = "Polynomial Regression Model => p : {}".format(str(output["p"]))
        return ret_str
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error


@app.route("/modelPredictAll")    
def modelPredictAll(reg_type="ALL"):    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......modelPredictAll...."]))
    try:
        ret_str = ""
        #reg_type = request.args["regType"]        
        output = model.model_predict_all(logger, fconfig["processdata"], reg_type) 
        if "errorType" in output.keys() and "errorOn" in output.keys():            
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_predict_all due to.. : ", str(output["errorOn"])])) 
            raise Exception
        if "isSuccess" in output.keys() and output.get("isSuccess"): 
            if "slrYhat" in output.keys() and  "slrintercept"  in output.keys() and "slrslope"  in output.keys():
                ret_str = ret_str + \
                    ''' 
                    Simple Linear Regression Model =>
                            Yhat : {}
                            Intercept : {}
                            Slope : {} 
                    '''.format(str(output["slrYhat"]), str(output["slrintercept"]), str(output["slrslope"]))
            if "mlrYhat" in output.keys() and  "mlrintercept"  in output.keys() and "mlrslope"  in output.keys():
                ret_str = ret_str + \
                    ''' 
                    Multiple Linear Regression Model =>
                            Yhat : {}
                            Intercept : {}
                            Slope : {} 
                    '''.format(str(output["mlrYhat"]), str(output["mlrintercept"]), str(output["mlrslope"]))
            if "plrp" in output.keys():
                ret_str = ret_str + "Polynomial Regression Model => p : {}".format(str(output["plrp"]))        
        return ret_str
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error


@app.route("/modelCompare")    
def modelCompare():    
    logger = get_logger(LOG_DIR, __name__)
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Initializing", "......model modelCompare "]))
    try:
        ret_str = "No comparision provided"
        output = model.model_compare(logger, fconfig["processdata"]) 
        if "errorType" in output.keys() and "errorOn" in output.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_compare due to.. : ", str(output["errorOn"])])) 
            raise Exception
        if "isSuccess" in output.keys() and output.get("isSuccess"):
            if "slr_rSqScore" in output.keys() and  "slr_mse"  in output.keys() \
                    and "mlr_rSqScore"  in output.keys() and "mlr_mse" in output.keys() \
                    and "pr_rSqScore" in output.keys() and "pr_mse" in output.keys():
                ret_str = ''' 
                    Comparision between different model : \n
                        Simple linear regression => \n
                            R-square : {}
                            MSE : {}
                        Multiple linear regression => \n
                            R-square : {}
                            MSE : {}
                        Polynomial regression => \n
                            R-square : {}
                            MSE : {} 
                    '''.format(str(output["slr_rSqScore"]), str(output["slr_mse"]), 
                               str(output["mlr_rSqScore"]), str(output["mlr_mse"]),
                               str(output["pr_rSqScore"]), str(output["pr_mse"]))
        return ret_str
    except Exception:
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : Untracked Exception in 1st block"]))
        error_trace = sys.exc_info()
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        error = json.dumps(error)
        return error

if __name__ == "__main__" :    
    app.run(host='0.0.0.0', port=80 , debug= True)
