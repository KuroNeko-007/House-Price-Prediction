import os
import sys
from dataclasses import dataclass
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logger

from src.utils import save_object,evaluate_models
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logger.info("Split training and test input data")

            X_test_final = test_array
            sample = pd.read_csv('artifacts/sample_submission.csv')
            logger.info(train_array.shape)
            logger.info(test_array.shape)
            X = train_array[:, :-1] 
            y = train_array[:, -1]  


            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            }

            #Optuna Optimized Parameters
            params={
                "XGBRegressor":{
                    'n_estimators': 4837,
                    'learning_rate': 0.09880880217552664, 
                    'colsample_bytree': 0.5803925990843555, 
                    'subsample': 0.6840323253259143, 
                    'min_child_weight': 4, 
                    'random_state': 42
                },
                "CatBoosting Regressor":{
                    'iterations': 6221, 
                    'learning_rate': 0.011852462849694064, 
                    'depth': 3, 
                    'random_state': 42
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            best_model_rmse = min(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_rmse)
            ]
            best_model = models[best_model_name]

            logger.info(f"Best found model on both training and testing dataset")
            logger.info(f"Best Model: {best_model_name}")
            logger.info(f"Best Model Score: {best_model_rmse}")


            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            logger.info(f"R2 Score: {r2_square}")

            test_rmse = mean_squared_error(y_test, predicted, squared=False)
            logger.info(f"Test RMSE: {test_rmse}")

            y_test_pred_log = best_model.predict(X_test_final)
            y_test_pred = np.exp(y_test_pred_log)
            ypred = pd.DataFrame(y_test_pred)
            result_df = pd.concat([sample['Id'],ypred],axis=1)
            result_df.columns = ['Id', 'SalePrice']
            result_df.to_csv('artifacts/result.csv',index=False)      
            return r2_square
            
        except Exception as e:
            raise CustomException(e)