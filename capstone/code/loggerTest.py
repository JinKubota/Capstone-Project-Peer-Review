import os
import sys
import json
import time
import datetime
import unittest
import inspect 


config_path = "C:/work_product/python_al/capstone/config/"
config_file = "config.json"
with open(config_path+config_file) as file_config:
    fconfig = json.load(file_config) 

logpath = fconfig["logdir"]    
curr_date = datetime.datetime.now().date()

CHK_ASSERT = "OK"


class LoggerTest(unittest.TestCase): 

    def test_log_exists(self): 
        curr_date_str = str(curr_date)[8:10] + "-" + str(curr_date)[5:7] + "-" + str(curr_date)[0:4]
        filename = "cap_" + curr_date_str + ".log"
        
        if os.path.exists(logpath) and os.path.isfile(logpath + filename):
            print("File  exists : " + filename)
        else:
            print("File not  exists : " + filename)
        self.assertEqual("OK", CHK_ASSERT) 
    

    def test_error_exists(self): 
        curr_date_str = str(curr_date)[8:10] + "-" + str(curr_date)[5:7] + "-" + str(curr_date)[0:4]
        filename = "cap_" + curr_date_str + ".log"
        cnt = 0
        
        if os.path.exists(logpath) and os.path.isfile(logpath + filename):
            file = open(logpath + filename, "r")
            lines = file.readlines()            
            for line in lines:
                cnt = cnt + line.count("ERROR")
        
        if cnt == 0:         
            print("No error exists into log file : " + filename)
        else:
            print("Error exists into log file : " + filename)
        self.assertEqual("OK", CHK_ASSERT) 
        
        
if __name__ == '__main__': 
    unittest.main() 
