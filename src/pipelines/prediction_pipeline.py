import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
class CustomData:
    def __init__(self, carName, year, presentPrice, kmsDriven, owner, fuelType, sellerType, transmissionType):
        self.carName = carName
        self.year = year
        self.presentPrice = presentPrice
        self.kmsDriven = kmsDriven
        self.owner = owner
        self.fuelType = fuelType
        self.sellerType = sellerType
        self.transmissionType = transmissionType

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Car_Name': [self.carName],
                'Year': [self.year],  # Include 'Year' column
                'Present_Price': [self.presentPrice],
                'Kms_Driven': [self.kmsDriven],
                'Owner': [self.owner],
                'Fuel_Type': [self.fuelType],
                'Seller_Type': [self.sellerType],
                'Transmission': [self.transmissionType]
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.error('Exception occurred in prediction pipeline')
            raise CustomException(e, sys)
