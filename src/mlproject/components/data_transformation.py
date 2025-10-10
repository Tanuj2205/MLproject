#  imports for data transformation
import sys
import os
from dataclasses import dataclass 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline   
from sklearn.impute import SimpleImputer
from mlproject.exception import CustomException
from mlproject.logger import logging
from mlproject.utils import save_object
import pickle

# Configuration class
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# Data Transformation    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        # Create the numerical and categorical pipelines
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']    
            # Pipeline for numerical columns
            num_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='median')),   
                ('scaler',StandardScaler()) 
                ])
            # Pipeline for categorical columns
            cat_pipeline = Pipeline(steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('one_hot_encoder',OneHotEncoder()), 
                ('scaler',StandardScaler(with_mean=False)) 
                ])
                
            logging.info(f'Numerical columns: {numerical_columns}')
            logging.info(f'Categorical columns: {categorical_columns}') 
            
            # ColumnTransformer is used to combine both numerical and categorical pipelines
            preprocessor = ColumnTransformer(
                [
                ('num_pipeline',num_pipeline,numerical_columns),
                ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )
            return preprocessor         
                      
        except Exception as e: 
            raise CustomException(e,sys)    
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Reading the train and test data
            logging.info('Reading train and test data')
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)    
        
            logging.info('Read train and test data completed')
            
            logging.info('Obtaining preprocessing object')
            # Get the preprocessor object
            preprocessor_obj = self.get_data_transformer_object()
            
            # Define target and numerical columns
            target_column_name = 'math_score'
            numerical_columns = ['writing_score','reading_score']   
            
            # Splitting the input and target features from training dataframe
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            # Splitting the input and target features from testing dataframe
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info(f'Applying preprocessing object on training and testing dataframe.')
            
            # Transforming the training and testing data using preprocessor object
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            
            # transforming the testing data
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            # Combining the transformed input features and target feature into a single array
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)] # np.c_ is used to concatenate two arrays
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            # Saving the preprocessor object
            logging.info(f'Saving the preprocessor object.')
         
           # from mlproject.utils import save_object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e,sys)