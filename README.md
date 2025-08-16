# Housing Price Predictor

A machine learning project that predicts house prices using various features and provides a web interface for predictions.

## Project Overview

This project implements a housing price prediction system using machine learning algorithms. It includes:

- **Data preprocessing and feature engineering**
- **Machine learning model training** (CatBoost)
- **Web application** for making predictions
- **REST API** for batch predictions
- **Comprehensive logging and monitoring**

## Project Structure

```
Housing-Price-Predictor/
├── src/                    # Source code
│   ├── components/         # Data processing components
│   ├── pipeline/          # Training and prediction pipelines
│   ├── exception.py       # Custom exception handling
│   └── logger.py          # Logging configuration
├── templates/             # HTML templates for web interface
├── static/               # CSS, JS, and static assets
├── notebook/             # Jupyter notebooks and data
│   ├── data/            # Dataset files
│   └── EDA.ipynb        # Exploratory Data Analysis
├── app.py               # Flask web application
├── requirements.txt     # Python dependencies
├── setup.py            # Package setup
└── .gitignore          # Git ignore rules
```

## Features

### Web Interface
- **Single Prediction**: Input individual house features to get price predictions
- **Batch Prediction**: Upload CSV files for multiple predictions
- **Results Display**: View and download prediction results

### Machine Learning
- **CatBoost Algorithm**: Gradient boosting for accurate predictions
- **Feature Engineering**: Comprehensive data preprocessing
- **Model Persistence**: Save and load trained models
- **Performance Metrics**: Model evaluation and validation

### Data Processing
- **Data Ingestion**: Handle various data formats
- **Data Transformation**: Feature scaling, encoding, and cleaning
- **Missing Value Handling**: Robust imputation strategies

## Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Housing-Price-Predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

## Usage

### Web Interface
1. Start the application: `python app.py`
2. Open your browser and go to `http://localhost:5000`
3. Choose between single prediction or batch upload

### Single Prediction
1. Fill in the house features in the web form
2. Click "Predict" to get the estimated price
3. View the prediction result

### Batch Prediction
1. Prepare a CSV file with house features
2. Upload the file through the web interface
3. Download the results as a CSV file

### API Usage
```python
import requests

response = requests.post("http://localhost:5000/predict", json=data)
prediction = response.json()["prediction"]

# Batch prediction
files = {"file": open("test_data.csv", "rb")}
response = requests.post("http://localhost:5000/predict_batch", files=files)
```

## Data Format

### Input Features
The model expects the following features (based on the Ames Housing dataset):

- **MSSubClass**: Building class
- **MSZoning**: Zoning classification
- **LotArea**: Lot size in square feet
- **Street**: Type of road access
- **Alley**: Type of alley access
- **LotShape**: General shape of property
- **LandContour**: Flatness of the property
- **Utilities**: Type of utilities available
- **LotConfig**: Lot configuration
- **LandSlope**: Slope of property
- **Neighborhood**: Physical locations within Ames city limits
- **Condition1**: Proximity to various conditions
- **Condition2**: Proximity to various conditions (if more than one is present)
- **BldgType**: Type of dwelling
- **HouseStyle**: Style of dwelling
- **OverallQual**: Overall material and finish quality
- **OverallCond**: Overall condition rating
- **YearBuilt**: Original construction date
- **YearRemodAdd**: Remodel date
- **RoofStyle**: Type of roof
- **RoofMatl**: Roof material
- **Exterior1st**: Exterior covering on house
- **Exterior2nd**: Exterior covering on house (if more than one material)
- **MasVnrType**: Masonry veneer type
- **MasVnrArea**: Masonry veneer area in square feet
- **ExterQual**: Exterior material quality
- **ExterCond**: Present condition of the material on the exterior
- **Foundation**: Type of foundation
- **BsmtQual**: Height of the basement
- **BsmtCond**: General condition of the basement
- **BsmtExposure**: Walkout or garden level basement walls
- **BsmtFinType1**: Quality of basement finished area
- **BsmtFinSF1**: Type 1 finished square feet
- **BsmtFinType2**: Quality of second finished area (if present)
- **BsmtFinSF2**: Type 2 finished square feet
- **BsmtUnfSF**: Unfinished square feet of basement area
- **TotalBsmtSF**: Total square feet of basement area
- **Heating**: Type of heating
- **HeatingQC**: Heating quality and condition
- **CentralAir**: Central air conditioning
- **Electrical**: Electrical system
- **1stFlrSF**: First Floor square feet
- **2ndFlrSF**: Second floor square feet
- **LowQualFinSF**: Low quality finished square feet (all floors)
- **GrLivArea**: Above grade (ground) living area square feet
- **BsmtFullBath**: Basement full bathrooms
- **BsmtHalfBath**: Basement half bathrooms
- **FullBath**: Full bathrooms above grade
- **HalfBath**: Half baths above grade
- **BedroomAbvGr**: Bedrooms above grade (does NOT include basement bedrooms)
- **KitchenAbvGr**: Kitchens above grade
- **KitchenQual**: Kitchen quality
- **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
- **Functional**: Home functionality (Assume typical unless deductions are warranted)
- **Fireplaces**: Number of fireplaces
- **FireplaceQu**: Fireplace quality
- **GarageType**: Garage location
- **GarageYrBlt**: Year garage was built
- **GarageFinish**: Interior finish of the garage
- **GarageCars**: Size of garage in car capacity
- **GarageArea**: Size of garage in square feet
- **GarageQual**: Garage quality
- **GarageCond**: Garage condition
- **PavedDrive**: Paved driveway
- **WoodDeckSF**: Wood deck area in square feet
- **OpenPorchSF**: Open porch area in square feet
- **EnclosedPorch**: Enclosed porch area in square feet
- **3SsnPorch**: Three season porch area in square feet
- **ScreenPorch**: Screen porch area in square feet
- **PoolArea**: Pool area in square feet
- **PoolQC**: Pool quality
- **Fence**: Fence quality
- **MiscFeature**: Miscellaneous feature not covered in other categories
- **MiscVal**: Value of miscellaneous feature
- **MoSold**: Month Sold (MM)
- **YrSold**: Year Sold (YYYY)
- **SaleType**: Type of sale
- **SaleCondition**: Condition of sale

### Output
- **Prediction**: Estimated house price in USD

## Model Training

To retrain the model:

1. **Prepare training data** in the `notebook/data/` directory
2. **Run the training pipeline**:
   ```python
   from src.pipeline.train_pipeline import TrainPipeline
   
   trainer = TrainPipeline()
   trainer.train()
   ```

3. **Model artifacts** will be saved in the `artifacts/` directory

## Dependencies

### Core Dependencies
- **Flask**: Web framework
- **CatBoost**: Gradient boosting algorithm
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning utilities

### Development Dependencies
- **Jupyter**: Interactive notebooks
- **Matplotlib**: Plotting
- **Seaborn**: Statistical visualization

## Configuration

The application uses environment variables for configuration:

- `PORT`: Application port (default: 5000)
- `DEBUG`: Debug mode (default: True)
- `LOG_LEVEL`: Logging level (default: INFO)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **Ames Housing Dataset**: Used for training and testing
- **CatBoost**: Gradient boosting library
- **Flask**: Web framework

## Support

For questions or issues, please open an issue on GitHub or contact the maintainers.
