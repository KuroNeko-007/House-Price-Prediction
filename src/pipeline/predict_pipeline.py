import sys
import os
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            # Convert log-scale predictions back to dollar amounts
            preds = np.expm1(preds)
            return preds
        except Exception as e:
            raise CustomException(e)
    


class CustomData:
    def __init__(self, **kwargs):
        """
        Initialize with any number of features using keyword arguments
        For 70 features, this is more practical than defining each parameter
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def get_data_as_data_frame(self):
        try:
            # Convert all attributes to a dictionary format suitable for DataFrame
            custom_data_input_dict = {}
            for key, value in self.__dict__.items():
                custom_data_input_dict[key] = [value]
            
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e)
    
    @classmethod
    def from_dict(cls, data_dict):
        """
        Create CustomData instance from dictionary
        Useful when receiving data from forms or APIs
        """
        try:
            return cls(**data_dict)
        except Exception as e:
            raise CustomException(e)
    
    @classmethod
    def from_form_data(cls, form_data):
        """
        Create CustomData instance from Flask form data
        Handles type conversion for numeric fields
        """
        try:
            data_dict = {}
            for key, value in form_data.items():
                # Try to convert to appropriate type
                if value.isdigit():
                    data_dict[key] = int(value)
                else:
                    try:
                        data_dict[key] = float(value)
                    except ValueError:
                        data_dict[key] = value  # Keep as string
            
            return cls(**data_dict)
        except Exception as e:
            raise CustomException(e)
    
    def validate_features(self, required_features):
        """
        Validate if all required features are present
        """
        try:
            current_features = set(self.__dict__.keys())
            required_features_set = set(required_features)
            missing_features = required_features_set - current_features
            
            if missing_features:
                raise ValueError(f"Missing required features: {list(missing_features)}")
            
            return True
        except Exception as e:
            raise CustomException(e)

class FileUploadData:
    """
    Class to handle file upload and conversion to DataFrame
    """
    def __init__(self, file_object):
        self.file_object = file_object
    
    def process_uploaded_file(self):
        try:
            # Get file extension
            filename = self.file_object.filename
            file_extension = filename.split('.')[-1].lower()
            
            if file_extension == 'csv':
                df = pd.read_csv(self.file_object)
            elif file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(self.file_object)
            else:
                raise ValueError("Unsupported file format. Please upload CSV or Excel file.")
            
            print(f"File processed successfully. Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            raise CustomException(e)
    
    def validate_columns(self, df, required_columns):
        """
        Validate if the uploaded file has required columns
        """
        try:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            return True
        except Exception as e:
            raise CustomException(e)