import os
import sys
import json
import time
import pandas as pd
import numpy as np
import re
import inspect


def get_config(config_path, config_file):
    try:        
        with open(config_path+config_file) as file_config:
            fconfig = json.load(file_config)   
        return fconfig        
    except:
        error_trace = sys.exc_info()
        print("error in configuration file load due to ....", error_trace)

def fetch_data(logger, inputfilepath, processfilepath):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of fetch_data : ", str(time.time())]))
    try:
        df = pd.DataFrame()
        for i in os.listdir(inputfilepath):
            if i.split(".")[-1].lower() == "json":
                with open(inputfilepath+i) as file1:
                    dict1 = json.load(file1)
                    df1 = pd.DataFrame(dict1)
                    df = pd.concat([df, df1], axis=0, ignore_index=True)  
        if not df.empty:
            logger.info("".join(["[" , fn_name_from_inspect , "]" , "df=", str(len(df)), str(df.columns.tolist())]))
            df.to_csv(processfilepath + "input.csv", index=False)  
        return {"isSuccess": True, "noOfRecord": len(df)} 
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in fetch data due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error
        

def correct_input(logger, processfilepath):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Istart of correct_input : ", str(time.time())]))
    try:    
        df = pd.read_csv(processfilepath+"input.csv")
        df_cols = df.columns.tolist()    
        """
        if 'StreamID' in df_cols:
            df.rename(columns={'StreamID':'stream_id'},inplace=True)
        if 'TimesViewed' in df_cols:
            df.rename(columns={'TimesViewed':'times_viewed'},inplace=True)
        """    
        years,months,days = df['year'].values,df['month'].values,df['day'].values
        dates = ["{}-{}-{}".format(years[i],str(months[i]).zfill(2),str(days[i]).zfill(2)) for i in range(df.shape[0])]
        df['invoice_date'] = np.array(dates,dtype='datetime64[D]')           
        df['invoice'] = [re.sub("\D+","",i) for i in df['invoice'].values]   
        df = df.fillna(0)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , "df=", str(len(df)), str(df.columns.tolist())]))
        df.to_csv(processfilepath+"input_clean.csv", index=False)
        return {"isSuccess": True, "noOfRecord": len(df)} 
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " :error in correct input data due to ...", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error
