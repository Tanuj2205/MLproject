import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from mlproject.logger import logging
from mlproject.exception import CustomException
from mlproject.components.data_ingestion import DataIngestion
# from mlproject.components.data_ingestion import DataIngestionConfig
from mlproject.components.data_transformation import DataTransformationConfig,DataTransformation
from mlproject.components.model_tranier import ModelTrainer
from mlproject.components.model_tranier import ModelTrainerConfig

if __name__=="__main__":
    logging.info("The Execution has started...")
    
    try:
        #data_inegestion_config=DataIngestionConfig()
        data_ingestion=DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
        
        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        # data_transformation.initiate_data_transformation(train_data_path,test_data_path)
        logging.info("The Execution has ended...")
        
        # model_trainer_config=ModelTrainerConfig()
        train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        ## Model Training

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))
        
    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)    
    