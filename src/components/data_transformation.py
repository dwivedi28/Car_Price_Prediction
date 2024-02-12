from sklearn.impute import SimpleImputer  # Handling Missing Values
from sklearn.preprocessing import StandardScaler  # Handling Feature Scaling
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import category_encoders as ce
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import sys
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np

# Data Transformation config
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


# Data Ingestionconfig class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled
            categorical_cols = ['Car_Name', 'Fuel_Type', 'Seller_Type','Transmission']
            numerical_cols = ['Year', 'Present_Price', 'Kms_Driven', 'Owner']
            logging.info('Pipeline Initiated')

            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define categorical pipelines for each categorical column
            cat_pipeline_Car_Name = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('category', ce.TargetEncoder()),  # Using TargetEncoder for encoding categorical variables
                ('scaler', StandardScaler(with_mean=False))
            ])

            cat_pipeline_Fuel_Type_Seller_Type_Transmission = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehotencoder', OneHotEncoder()),  # Using OneHotEncoder for encoding categorical variables
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline_Car_Name', cat_pipeline_Car_Name, ['Car_Name']),
                ('cat_pipeline_Fuel_Type_Seller_Type_Transmission',
                 cat_pipeline_Fuel_Type_Seller_Type_Transmission,
                 ['Fuel_Type', 'Seller_Type', 'Transmission'])
            ])

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            logging.error("Error in Data Transformation")
            raise CustomException(e, sys)
    def initiate_data_transformation(self, train_path, test_path):
     try:
        # Reading train and test data
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logging.info('Read train and test data completed')
        logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
        logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

        logging.info('Obtaining preprocessing object')
        preprocessing_obj = self.get_data_transformation_object()

        target_column_name = 'Selling_Price'
        drop_columns = [target_column_name]

        # Check the columns of the train and test dataframes
        logging.info(f'Train Dataframe columns: {train_df.columns}')
        logging.info(f'Test Dataframe columns: {test_df.columns}')

        # Features into independent and dependent features
        input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
        target_feature_train_df = train_df[target_column_name]
        input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
        target_feature_test_df = test_df[target_column_name]

        # Apply the transformation
        input_feature_train_df = input_feature_train_df.reset_index(drop=True)
        input_feature_test_df = input_feature_test_df.reset_index(drop=True)

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df,target_feature_train_df)

        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        logging.info("Applying preprocessing object on training and testing datasets.")
        train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values.ravel()]
        test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

        save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

        logging.info('Preprocessor pickle is created and saved')

        return train_arr, test_arr

     except Exception as e:
        logging.error("Exception occurred in the initiate_data_transformation")
        logging.info(f"Error details: {e}")

    
        raise CustomException(e, sys)