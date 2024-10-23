# combined-miles-per-gallon


Overview
This project aims to predict the combined miles per gallon (MPG) of vehicles using a machine learning model based on various input features such as city MPG, highway MPG, number of cylinders, and other characteristics of the car. The model uses a Random Forest Regressor for predictions.

Technologies Used
Python: Programming language for data analysis and machine learning.
Pandas: Library for data manipulation and analysis.
Scikit-learn: Machine learning library for building and evaluating models.
NumPy: Library for numerical computations.
IO: For handling data input.
Dataset
The dataset used for this project can be loaded from a CSV file named car_data.csv. The dataset contains the following columns:

city_mpg: City miles per gallon.
class: Class of the vehicle (e.g., midsize car, SUV).
combination_mpg: Combined miles per gallon (target variable).
cylinders: Number of cylinders in the engine.
displacement: Engine displacement in liters.
drive: Drive type (e.g., fwd, rwd, awd).
fuel_type: Type of fuel used (e.g., gas, diesel).
highway_mpg: Highway miles per gallon.
make: Make of the vehicle (e.g., Mazda, Ford).
model: Model of the vehicle.
transmission: Type of transmission (e.g., automatic, manual).
year: Year of manufacture.
Installation
To run this project, ensure you have Python installed along with the necessary libraries. You can install the required libraries using pip:

pip install pandas scikit-learn

Usage
Load the dataset: Load the dataset from the car_data.csv file.

Preprocess the data:

Split the features and the target variable (combination_mpg).
Define numerical and categorical columns.
Create a preprocessing pipeline that includes scaling for numerical data and one-hot encoding for categorical data.
Train the model:

Split the data into training and testing sets.
Train the Random Forest Regressor model on the training data.
Make predictions:

Predict the combination_mpg for the test set.
Evaluate the model using Mean Squared Error (MSE) and R-squared metrics.
Example Prediction: Use the following input to predict the combination_mpg for a new vehicle:

new_input = pd.DataFrame({
    'city_mpg': [28],
    'class': ['small sport utility vehicle'],
    'cylinders': [4],
    'displacement': [2.5],
    'drive': ['fwd'],
    'fuel_type': ['gas'],
    'highway_mpg': [35],
    'make': ['mazda'],
    'model': ['6'],
    'transmission': ['m'],
    'year': [2014]
})
predicted_mpg = model.predict(new_input)
print(f'Predicted combination_mpg: {predicted_mpg[0]}')

Evaluation Metrics
The performance of the model is evaluated using:

Mean Squared Error (MSE): Measures the average of the squares of the errors.
R-squared (R²): Indicates the proportion of the variance in the dependent variable that's predictable from the independent variables.
