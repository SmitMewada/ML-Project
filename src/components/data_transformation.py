import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")

class DataTranformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_tranformer_obj(self):
        '''
            This function is responsible for data transformation.
        '''
        try:
            numerical_features = ["writing_score", "reading_score"]
            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            
            logging.info("Numerical standard scaling completed.")
            
            categorical_pipeline = Pipeline(
               steps= [
                   ("imputer", SimpleImputer(strategy="most-frequent")),
                   ("ohencoder", OneHotEncoder()),
                   ("scaler", StandardScaler())
               ]
            )
            
            logging.info("Categorical standard scaling completed.")
            
            preprocessor = ColumnTransformer(
               [
                   ("num_pipline", numerical_pipeline, numerical_features),
                   ("cat_pipeline", categorical_pipeline, categorical_features)
               ]
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
        
    def initiate_data_tranformation(self, train_path, test_path):
        '''
            This function initiates the data transformation.
        '''
        logging.info("Data tranformation initiated!")
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            train_df.columns = train_df.columns.str.replace(" ", "_")
            test_df.columns = test_df.columns.str.replace(" ", "_")

            logging.info("Read train and test data completed.")
            
            preprocessing_obj = self.get_data_tranformer_obj()   
            target_column = "math_score"
            numerical_features = ["writing_score", "reading_score"]
            
            input_feature_train_df = train_df.drop(columns=target_column, axis=1)
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )
            
            logging.info("Saved preprocessed data.")
        except Exception as e:
            raise CustomException(e, sys)
    