import sys
import time
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from dataFetchClean import fetch_data, correct_input

def split_train_test(logger, feature, target):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of split_train_test : ", str(time.time())]))
    try:
        feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.4, random_state=0) 
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             "number of test samples :", str(feature_test.shape[0]),
                             "number of training samples:", str(feature_train.shape[0])
                            ]))
        return {
            "feature_train": feature_train, "target_train": target_train, 
            "feature_test": feature_test, "target_test": target_test
            }                    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in split_train_test due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error

def model_simple_linear_reg(logger, x, y):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_simple_linear_reg : ", str(time.time())]))
    try: 
        lm = LinearRegression()
        
        # check model with "times_viewed" with "price"
        lm.fit(x, y)
        Yhat = lm.predict(x)
        intercept = lm.intercept_
        slope = lm.coef_        
        
        logger.info("".join(["[" , fn_name_from_inspect , "]" ,
                             " : Prediction using data : ", str(Yhat[0:5]), " : intercept : ", str(lm.intercept_), " : slope : ", str(slope)]))
        
        #Calculate the R^2
        rSqScore = lm.score(x, y)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The R-square is : ", str(rSqScore)]))
        
        # calculate the MSE
        mse = mean_squared_error(y, Yhat)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The mean square error is : ", str(mse)]))
        
        return {"Yhat": Yhat, "intercept": intercept, "slope": slope, "rSqScore": rSqScore, "mse": mse}                    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_simple_linear_reg due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error 
                
def model_multiple_linear_reg(logger, z, y):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_multiple_linear_reg : ", str(time.time())]))
    try: 
        lm = LinearRegression()        
        # check model with 'times_viewed', 'invoice', 'year', 'month', 'day' with "price"
        lm.fit(z, y)
        Yhat = lm.predict(z)
        intercept = lm.intercept_
        slope = lm.coef_
        logger.info("".join(["[" , fn_name_from_inspect , "]" ,
                             " : Prediction using data : ", str(Yhat[0:5]), " : intercept : ", str(lm.intercept_), " : slope : ", str(slope)]))
                             
        #Calculate the R^2
        rSqScore = lm.score(z, y)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The R-square is : ", str(rSqScore)]))
        
        # calculate the MSE
        mse = mean_squared_error(y, Yhat)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The mean square error is : ", str(mse)]))
                             
        return {"Yhat": Yhat, "intercept": intercept, "slope": slope, "rSqScore": rSqScore, "mse": mse}                    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_multiple_linear_reg due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()
    
def model_polynomial_reg(logger, x, y):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_polynomial_reg : ", str(time.time())]))
    try: 
        lm = LinearRegression()

        # Here we use a polynomial of the 3rd order (cubic) 
        f = np.polyfit(x, y, 3)
        p = np.poly1d(f)
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : p : ", str(p),]))
                              
        # Calculate the R^2
        rSqScore = r2_score(y, p(x))
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The R-square is : ", str(rSqScore)]))
        
        # calculate the MSE
        mse = mean_squared_error(y, p(x))
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : The mean square error is : ", str(mse)]))
                    
        return {"p": p, "rSqScore": rSqScore, "mse": mse}                    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_polynomial_reg due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error 
        
def model_predict(logger, x_train, x_test, y_train, y_test): 
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_predict : ", str(time.time())]))
    try: 
        lr = LinearRegression()
        lr.fit(x_train[["invoice_date", "times_viewed"]], y_train)        
        yhat_train = lr.predict(x_train[["invoice_date", "times_viewed"]])
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Prediction using training data : ", str(yhat_train[0:5])]))
        
        lr.fit(x_test[["invoice_date", "times_viewed"]], y_test) 
        yhat_test = lr.predict(x_test[["invoice_date", "times_viewed"]])
        logger.info("".join(["[" , fn_name_from_inspect , "]" , " : Prediction using test data : ", str(yhat_test[0:5])]))
        
        return {"yhat_train": yhat_train, "yhat_test": yhat_test}                    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error    

def model_predict_slr(logger, pathprocessdata):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_predict_slr : ", str(time.time())]))
    try: 
    
        df = pd.read_csv(pathprocessdata + "input_clean.csv")
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " : No of record fetch : ", str(len(df)), " Columns of dataframe ", str(df.columns.tolist())
                            ])) 
        
        df = df.drop(["total_price","StreamID","TimesViewed"], axis=1)
        df = df.fillna(0)
        df.customer_id = df.customer_id.astype('object')
        df.stream_id = df.stream_id.astype('object')
        df.times_viewed = df.times_viewed.astype('int64')
        
        x = df[["times_viewed"]]
        y = df["price"]
        outmsg = split_train_test(logger, x, y)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        x_train = outmsg["feature_train"]
        y_train = outmsg["target_train"]
        x_test = outmsg["feature_test"]
        y_test = outmsg["target_test"]
        
        outmsg = model_simple_linear_reg(logger, x_train, y_train)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_simple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception

        outmsg.update({"isSuccess": True})
        return outmsg    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error    

def model_predict_mlr(logger, pathprocessdata):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_predict_mlr : ", str(time.time())]))
    try:     
        df = pd.read_csv(pathprocessdata + "input_clean.csv")
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " : No of record fetch : ", str(len(df)), " Columns of dataframe ", str(df.columns.tolist())
                            ])) 
        
        df = df.drop(["total_price","StreamID","TimesViewed"], axis=1)
        df = df.fillna(0)
        df.customer_id = df.customer_id.astype('object')
        df.stream_id = df.stream_id.astype('object')
        df.times_viewed = df.times_viewed.astype('int64')
        
        z = df[["times_viewed", "invoice", "year", "month", "day"]]
        y = df["price"]
        outmsg = split_train_test(logger, z, y)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        z_train = outmsg["feature_train"]
        y_train = outmsg["target_train"]
        z_test = outmsg["feature_test"]
        y_test = outmsg["target_test"]
        
        outmsg = model_multiple_linear_reg(logger, z_train, y_train)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_multiple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception

        outmsg.update({"isSuccess": True})
        return outmsg    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict_mlr due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error    

def model_predict_plr(logger, pathprocessdata):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_predict_plr : ", str(time.time())]))
    try: 
    
        df = pd.read_csv(pathprocessdata + "input_clean.csv")
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " : No of record fetch : ", str(len(df)), " Columns of dataframe ", str(df.columns.tolist())
                            ])) 
        
        df = df.drop(["total_price","StreamID","TimesViewed"], axis=1)
        df = df.fillna(0)
        df.customer_id = df.customer_id.astype('object')
        df.stream_id = df.stream_id.astype('object')
        df.times_viewed = df.times_viewed.astype('int64')
        
        x = df["times_viewed"]
        y = df["price"]
        outmsg = split_train_test(logger, x, y)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        x_train = outmsg["feature_train"]
        y_train = outmsg["target_train"]
        x_test = outmsg["feature_test"]
        y_test = outmsg["target_test"]
        
        outmsg = model_polynomial_reg(logger, x_train, y_train)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_polynomial_reg due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception

        outmsg.update({"isSuccess": True})
        return outmsg    
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict_plr due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error    

def model_predict_all(logger, pathprocessdata, reg_type):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_predict_all : ", str(time.time())]))
    try: 
        outmsg_all = {}
        df = pd.read_csv(pathprocessdata + "input_clean.csv")
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " : No of record fetch : ", str(len(df)), " Columns of dataframe ", str(df.columns.tolist())
                            ]))         
        df = df.drop(["total_price","StreamID","TimesViewed"], axis=1)
        df = df.fillna(0)
        df.customer_id = df.customer_id.astype('object')
        df.stream_id = df.stream_id.astype('object')
        df.times_viewed = df.times_viewed.astype('int64')
        
        if reg_type == "SLR":
            
            x = df[["times_viewed"]]
            y = df["price"]
            
            outmsg = split_train_test(logger, x, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , 
                                    " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            x_train = outmsg["feature_train"]
            x_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]        
            y_test = outmsg["target_test"]            
            
            outmsg = model_simple_linear_reg(logger, x_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_simple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            if "Yhat" in outmsg.keys() and  "intercept"  in outmsg.keys() and "slope"  in outmsg.keys():
                outmsg_all.update({
                    "slrYhat": outmsg["Yhat"],
                    "slrintercept": outmsg["intercept"],
                    "slrslope": outmsg["slope"]
                    })
                
        elif reg_type == "MLR":
        
            z = df[["times_viewed", "invoice", "year", "month", "day"]]
            y = df["price"]
            outmsg = split_train_test(logger, z, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            z_train = outmsg["feature_train"]
            z_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]
            y_test = outmsg["target_test"]
            
            outmsg = model_multiple_linear_reg(logger, z_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_multiple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            if "Yhat" in outmsg.keys() and  "intercept"  in outmsg.keys() and "slope"  in outmsg.keys():
                outmsg_all.update({
                    "mlrYhat": outmsg["Yhat"],
                    "mlrintercept": outmsg["intercept"],
                    "mlrslope": outmsg["slope"]
                    })    
                
        elif reg_type == "PLR":
        
            x = df["times_viewed"]
            y = df["price"]
            outmsg = split_train_test(logger, x, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            x_train = outmsg["feature_train"]
            x_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]            
            y_test = outmsg["target_test"]
            
            outmsg = model_polynomial_reg(logger, x_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_polynomial_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            if "p" in outmsg.keys():
                outmsg_all.update({"plrp": outmsg["p"]})      
        
        elif reg_type == "ALL":
        
            # SLR
            x = df[["times_viewed"]]
            y = df["price"]
            
            outmsg = split_train_test(logger, x, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , 
                                    " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            x_train = outmsg["feature_train"]
            x_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]        
            y_test = outmsg["target_test"]
            
            outmsg = model_simple_linear_reg(logger, x_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_simple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            if "Yhat" in outmsg.keys() and  "intercept"  in outmsg.keys() and "slope"  in outmsg.keys():                
                outmsg_all.update({
                    "slrYhat": outmsg["Yhat"],
                    "slrintercept": outmsg["intercept"],
                    "slrslope": outmsg["slope"]
                    })    
                  
            # MLR            
            z = df[["times_viewed", "invoice", "year", "month", "day"]]
            y = df["price"]
            outmsg = split_train_test(logger, z, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            z_train = outmsg["feature_train"]
            z_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]
            y_test = outmsg["target_test"]
            
            outmsg = model_multiple_linear_reg(logger, z_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_multiple_linear_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception  
            if "Yhat" in outmsg.keys() and "intercept"  in outmsg.keys() and "slope" in outmsg.keys():
                outmsg_all.update({
                    "mlrYhat": outmsg["Yhat"],
                    "mlrintercept": outmsg["intercept"],
                    "mlrslope": outmsg["slope"]
                    })
            
            # PLR
            x = df["times_viewed"]
            y = df["price"]
            outmsg = split_train_test(logger, x, y)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in split_train_test due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            x_train = outmsg["feature_train"]
            x_test = outmsg["feature_test"]
            y_train = outmsg["target_train"]            
            y_test = outmsg["target_test"]
            
            outmsg = model_polynomial_reg(logger, x_train, y_train)
            if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
                logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in model_polynomial_reg due to.. : ", str(outmsg["errorOn"])])) 
                raise Exception
            outmsg_all.update({"plrp": outmsg["p"]})     

        outmsg_all.update({"isSuccess": True})
        return outmsg_all        
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict_all due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error      
        

def model_compare(logger, pathprocessdata):
    fn_name_from_inspect = inspect.stack()[0][3]
    logger.info("".join(["[" , fn_name_from_inspect , "]" , " : start of model_compare : ", str(time.time())]))
    try:
    
        df = pd.read_csv(pathprocessdata + "input_clean.csv")
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " : No of record fetch : ", str(len(df)), " Columns of dataframe ", str(df.columns.tolist())
                            ])) 
        
        df = df.drop(["total_price","StreamID","TimesViewed"], axis=1)
        df = df.fillna(0)
        df.customer_id = df.customer_id.astype('object')
        df.stream_id = df.stream_id.astype('object')
        df.times_viewed = df.times_viewed.astype('int64')
        
        x = df[["times_viewed"]]
        y = df["price"]
        z = df[["times_viewed", "invoice", "year", "month", "day"]]
        
        # simple linear reg
        slr_rSqScore = None
        slr_mse = None
        outmsg = model_simple_linear_reg(logger, x, y)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in SLR due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        if "rSqScore" in outmsg.keys() and "mse" in outmsg.keys():
            slr_rSqScore = outmsg["rSqScore"]
            slr_mse = outmsg["mse"]
        
        # multi linear reg
        mlr_rSqScore = None
        mlr_mse = None
        outmsg = model_multiple_linear_reg(logger, z, y)
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in MLR due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        if "rSqScore" in outmsg.keys() and "mse" in outmsg.keys():
            mlr_rSqScore = outmsg["rSqScore"]
            mlr_mse = outmsg["mse"]
            
        # polynomial reg
        pr_rSqScore = None
        pr_mse = None
        outmsg = model_polynomial_reg(logger, df["times_viewed"], df["price"])
        if "errorType" in outmsg.keys() and "errorOn" in outmsg.keys():
            logger.error("".join(["[" , fn_name_from_inspect , "]" , " : error in PolyR due to.. : ", str(outmsg["errorOn"])])) 
            raise Exception
        if "rSqScore" in outmsg.keys() and "mse" in outmsg.keys():
            pr_rSqScore = outmsg["rSqScore"]
            pr_mse = outmsg["mse"]    
            
        logger.info("".join(["[" , fn_name_from_inspect , "]" , 
                             " Comparision between different model : ",
                             " \n Simple linear regression => : R-square : ", str(slr_rSqScore), " : MSE : ", str(slr_mse),
                             " \n Multiple linear regression => : R-square : ", str(mlr_rSqScore), " : MSE : ", str(mlr_mse),
                             " \n Polynomial regression => : R-square : ", str(pr_rSqScore), " : MSE : ", str(pr_mse)])) 
        output = {
            "isSuccess": True,
            "slr_rSqScore": slr_rSqScore, "slr_mse": slr_mse, 
            "mlr_rSqScore": mlr_rSqScore, "mlr_mse": mlr_mse,
            "pr_rSqScore": pr_rSqScore, "pr_mse": pr_mse
           } 
        return output 
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error        
  
  
def all_model(logger, df):        
    try:
        feature = df[["times_viewed"]]
        target = df["price"]
        outmsg = split_train_test(logger, feature, target)
        if "errorType" in outmsg.keys():
            raise Exception
        x_train = outmsg["feature_train"]
        y_train = outmsg["target_train"]
        x_test = outmsg["feature_test"]
        y_test = outmsg["target_test"]
        
        outmsg = model_predict(logger, x_train, x_test, y_train, y_test)
        if "errorType" in outmsg.keys():
            raise Exception
        yhat_train = outmsg["yhat_train"]
        yhat_test = outmsg["yhat_test"]
        return outmsg
    except:
        error_trace = sys.exc_info()
        logger.exception("".join(["[", fn_name_from_inspect, "]", " : error in model_predict due to ....", str(error_trace)]))
        error = {
            "errorType": str(error_trace[0])[8:][:-2],
            "errorOn": str(error_trace[1])
        }
        return error        