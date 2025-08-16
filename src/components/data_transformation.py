import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from feature_engine.encoding import RareLabelEncoder, OrdinalEncoder
import warnings

from src.utils import save_object
from src.logger import logger
from src.exception import CustomException

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file = os.path.join("artifacts", "preprocessor.pkl")
    train_data_path = os.path.join("artifacts", "train.csv")
    test_data_path = os.path.join("artifacts", "test.csv")


# -----------------------------
# High NaN Column Dropper
# -----------------------------
class HighNanColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.cols_to_drop_ = None
        self.is_fitted = False
        
    def fit(self, X, y=None):
        """Identify columns with more than threshold% missing values."""
        missing_percent = X.isnull().mean()
        self.cols_to_drop_ = missing_percent[missing_percent > self.threshold].index.tolist()
        
        logger.info(f"Dropping columns with > {self.threshold*100:.0f}% missing values:")
        for col in self.cols_to_drop_:
            logger.info(f"Dropped '{col}' ({missing_percent[col]*100:.2f}% missing)")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("HighNanColumnDropper must be fitted before transform")
        
        df = X.copy()
        df = df.drop(columns=self.cols_to_drop_, errors='ignore')
        return df


# -----------------------------
# Custom Imputer
# -----------------------------
class CustomImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted = False
        self.fill_values_ = {}
        
    def fit(self, X, y=None):
        """Calculate fill values for missing data and store them."""
        df = X.copy()
        
        categorical_cols = [col for col in df.columns if df[col].dtype == 'O']
        numerical_cols = [col for col in df.columns if df[col].dtype != 'O']
        
        logger.info("Calculating fill values for missing data:")
        
        # Calculate fill values for numerical columns
        for col in numerical_cols:
            mean_val = df[col].mean()
            self.fill_values_[col] = mean_val
            logger.info(f"Numerical '{col}' will be filled with mean: {mean_val:.2f}")
        
        # Calculate fill values for categorical columns
        for col in categorical_cols:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else None
            self.fill_values_[col] = mode_val
            logger.info(f"Categorical '{col}' will be filled with mode: '{mode_val}'")
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("CustomImputer must be fitted before transform")
        
        df = X.copy()
        
        logger.info("Applying stored fill values:")
        for col, val in self.fill_values_.items():
            if col == 'SalePrice':
                continue
            if col in df.columns:
                df[col].fillna(val, inplace=True)
                logger.info(f"Filled NaNs in '{col}' with stored value: {val}")
        
        return df


# -----------------------------
# Feature Engineering
# -----------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted = False
        
    def fit(self, X, y=None):
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        df = X.copy()
        
        # Age-based features (convert years to age)
        for feature in ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt']:
            if feature in df.columns:
                df[feature] = df['YrSold'] - df[feature]

        # Total number of bathrooms
        df["TotalBaths"] = (df.get("FullBath", 0) + 0.5 * df.get("HalfBath", 0) +
                           df.get("BsmtFullBath", 0) + 0.5 * df.get("BsmtHalfBath", 0))
        df.drop(columns=["FullBath", "HalfBath", "BsmtFullBath", "BsmtHalfBath"], 
               inplace=True, errors='ignore')

        # Total square footage
        df["TotalSF"] = df.get("TotalBsmtSF", 0) + df.get("GrLivArea", 0)
        df.drop(columns=["TotalBsmtSF", "GrLivArea"], inplace=True, errors='ignore')
        
        # Log-transform selected numeric features
        num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'TotalSF']
        for feature in num_features:
            if feature in df.columns:
                df[feature] = np.log1p(df[feature])
        
        logger.info("Feature engineering completed")
        return df


# -----------------------------
# Rare Label Encoder (fits on combined train+test)
# -----------------------------
class RareEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, tol=0.01, n_categories=1, replace_with='Rare'):
        self.tol = tol
        self.n_categories = n_categories
        self.replace_with = replace_with
        self.is_fitted = False
        self.rare_encoder = None
        self.categorical_cols_ = None
        
    def fit(self, X_train, X_test=None, y=None):
        """Fit rare encoder on combined train+test data."""
        # Combine train and test for rare encoding
        if X_test is not None:
            combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)
        else:
            combined_df = X_train.copy()
            
        # Identify categorical columns
        self.categorical_cols_ = [col for col in combined_df.columns if combined_df[col].dtype == 'O']
        logger.info(f"Categorical columns for rare encoding: {self.categorical_cols_}")
        
        # Fit rare encoder on combined data
        self.rare_encoder = RareLabelEncoder(
            tol=self.tol, 
            n_categories=self.n_categories, 
            replace_with=self.replace_with, 
            variables=self.categorical_cols_
        )
        self.rare_encoder.fit(combined_df)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("RareEncoderTransformer must be fitted before transform")
        
        return self.rare_encoder.transform(X)


# -----------------------------
# Ordinal Encoder and Scaler Pipeline
# -----------------------------
class OrdinalScalingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, encoding_method='ordered'):
        self.encoding_method = encoding_method
        self.is_fitted = False
        self.ordinal_encoder = None
        self.scaler = None
        self.categorical_cols_ = None
        
    def fit(self, X, y=None):
        df = X.copy()
        
        # Identify categorical columns for ordinal encoding
        self.categorical_cols_ = [col for col in df.columns if df[col].dtype == 'O']
        logger.info(f"Columns for ordinal encoding: {self.categorical_cols_}")
        
        # Fit ordinal encoder
        self.ordinal_encoder = OrdinalEncoder(
            encoding_method=self.encoding_method, 
            variables=self.categorical_cols_
        )
        df_encoded = self.ordinal_encoder.fit_transform(df, y)
        
        # Fit scaler
        self.scaler = MinMaxScaler()
        self.scaler.fit(df_encoded)
        
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("OrdinalScalingPipeline must be fitted before transform")
        
        # Apply ordinal encoding
        df_encoded = self.ordinal_encoder.transform(X)
        
        # Apply scaling
        df_scaled = self.scaler.transform(df_encoded)
        
        return df_scaled


# -----------------------------
# Multicollinearity Dropper
# -----------------------------
class MulticollinearityDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop=None):
        if columns_to_drop is None:
            columns_to_drop = ['GarageCars', 'TotRmsAbvGrd', 'Exterior2nd']
        self.columns_to_drop = columns_to_drop
        self.is_fitted = False
        
    def fit(self, X, y=None):
        self.is_fitted = True
        return self
    
    def transform(self, X):
        if not self.is_fitted:
            raise ValueError("MulticollinearityDropper must be fitted before transform")
        
        df = X.copy()
        df = df.drop(columns=self.columns_to_drop, errors='ignore')
        logger.info(f"Dropped multicollinear columns: {self.columns_to_drop}")
        return df


# -----------------------------
# Complete Preprocessing Pipeline
# -----------------------------
class PreprocessingPipeline(BaseEstimator, TransformerMixin):
    def __init__(self, nan_threshold=0.5):
        self.nan_threshold = nan_threshold
        self.is_fitted = False
        
        # Initialize all transformers
        self.high_nan_dropper = HighNanColumnDropper(threshold=nan_threshold)
        self.imputer = CustomImputer()
        self.feature_engineer = FeatureEngineer()
        self.rare_encoder = RareEncoderTransformer()
        self.multicollinearity_dropper = MulticollinearityDropper()
        self.ordinal_scaling_pipeline = OrdinalScalingPipeline()

    def fit(self, X_train, X_test=None, y=None):
        """
        Fit the complete preprocessing pipeline.
        
        Args:
            X_train: Training features
            X_test: Test features (optional, used for rare encoding)
            y: Target variable (optional, used for ordinal encoding)
        """
        df_train = X_train.copy()
        df_test = X_test.copy() if X_test is not None else None

        # Step 1: Drop high-NaN columns
        self.high_nan_dropper.fit(df_train)
        df_train = self.high_nan_dropper.transform(df_train)
        if df_test is not None:
            df_test = self.high_nan_dropper.transform(df_test)

        # Step 2: Fit and transform with imputer
        self.imputer.fit(df_train)
        df_train = self.imputer.transform(df_train)
        if df_test is not None:
            df_test = self.imputer.transform(df_test)
        
        # Step 3: Feature engineering
        self.feature_engineer.fit(df_train)
        df_train = self.feature_engineer.transform(df_train)
        if df_test is not None:
            df_test = self.feature_engineer.transform(df_test)

        # Step 4: Fit rare encoder on combined train+test data
        self.rare_encoder.fit(df_train, df_test)
        df_train = self.rare_encoder.transform(df_train)
        
        # Step 5: Drop multicollinear columns
        self.multicollinearity_dropper.fit(df_train)
        df_train = self.multicollinearity_dropper.transform(df_train)
        
        # Step 6: Fit ordinal encoding and scaling pipeline (only on train)
        self.ordinal_scaling_pipeline.fit(df_train, y)

        self.is_fitted = True
        return self

    def transform(self, X):
        """Transform new data using fitted pipeline."""
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before transform")
            
        df = X.copy()

        # Apply all transformations in sequence
        df = self.high_nan_dropper.transform(df)
        df = self.imputer.transform(df)
        df = self.feature_engineer.transform(df)
        df = self.rare_encoder.transform(df)
        df = self.multicollinearity_dropper.transform(df)
        df_scaled = self.ordinal_scaling_pipeline.transform(df)

        return df_scaled


# -----------------------------
# Main Data Transformation Class
# -----------------------------
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        """
        Main method to perform data transformation.
        
        Args:
            train_path: Path to training data CSV
            test_path: Path to test data CSV
            
        Returns:
            train_arr: Processed training data as numpy array
            test_arr: Processed test data as numpy array  
            preprocessor_path: Path to saved preprocessor object
        """
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logger.info("Read train and test data completed")

            # Define columns
            target_column = "SalePrice"
            id_column = "Id"

            # Remove outliers from training data (specific row indices)
            outlier_rows = [1182, 1298, 1169, 224]
            train_df = train_df.drop(index=outlier_rows, errors='ignore').reset_index(drop=True)
            logger.info(f"Removed outlier rows: {outlier_rows}")

            # Separate features and target
            input_feature_train_df = train_df.drop(columns=[id_column, target_column], errors='ignore')
            target_feature_train_df = train_df[target_column].copy()
            input_feature_test_df = test_df.drop(columns=[id_column], errors='ignore')

            # Apply log transformation to target variable (training only)
            target_feature_train_df = np.log1p(target_feature_train_df)
            logger.info("Applied log transformation to target variable")

            # Create and fit preprocessing pipeline
            preprocessor = PreprocessingPipeline(nan_threshold=0.5)
            logger.info("Fitting preprocessing pipeline on training and test data")

            # Fit with both train and test data for proper rare encoding
            preprocessor.fit(input_feature_train_df, input_feature_test_df, target_feature_train_df)
            
            # Transform train and test data
            input_feature_train_arr = preprocessor.transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            logger.info("Data transformation completed")

            # Combine features with target for training data
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = input_feature_test_arr

            # Save preprocessing object
            logger.info("Saving preprocessing object")
            save_object(self.data_transformation_config.preprocessor_obj_file, preprocessor)

            return (
                train_arr, 
                test_arr, 
                self.data_transformation_config.preprocessor_obj_file
            )

        except Exception as e:
            logger.error(f"Error in data transformation: {str(e)}")
            raise CustomException(e)

