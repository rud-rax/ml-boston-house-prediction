# ml-boston-house-prediction


## Project Summary: Boston Housing Data Analysis and Predictive Modeling

### **Objective:**
The primary goal of this project is to analyze a housing dataset, preprocess the data, and build predictive models to estimate the median value of owner-occupied homes (target variable `MEDV`). The project involves data cleaning, visualization, outlier detection, feature scaling, and the evaluation of multiple regression models to identify the best-performing model for predicting housing prices.

---

### **Key Steps:**

1. **Data Import and Exploration:**
   - The dataset (`HousingData.csv`) is loaded into a pandas DataFrame.
   - Initial exploration is performed using `df.describe()` to understand the distribution and summary statistics of the data.

2. **Handling Missing Values:**
   - Columns with missing values are identified and visualized using joint plots to understand their relationship with the target variable (`MEDV`).
   - Missing values are filled using interpolation methods (polynomial, nearest) and mode filling for categorical data.
   - The cleaned dataset is saved to a new CSV file (`hd1.csv`).

3. **Data Visualization:**
   - Pairplots and joint plots are created to visualize relationships between features and the target variable.
   - Violin plots are used to visualize the distribution of features in the dataset.

4. **Outlier Detection and Removal:**
   - The Isolation Forest algorithm is used to detect and remove outliers, assuming 20% of the data points are outliers.
   - Data before and after outlier removal is visualized using pairplots.

5. **Feature Scaling:**
   - Features are standardized using `StandardScaler` to have a mean of 0 and a standard deviation of 1.
   - The scaled data is visualized using a violin plot.

6. **Model Training and Evaluation:**
   - Multiple regression models are defined, including:
     - Linear Regression
     - K-Nearest Neighbors (KNN) with different configurations
     - Random Forest with various criteria and estimators
     - Gradient Boosting with different loss functions
   - A specific model is selected, and 5-fold cross-validation is performed using the R² score.
   - The model is trained on the training data and evaluated on the test data using metrics such as:
     - Mean Squared Error (MSE)
     - Root Mean Squared Error (RMSE)
     - Mean Absolute Error (MAE)
     - R² Score

7. **Logging Results:**
   - Model performance metrics are stored in a dictionary and appended to a JSON file (`logs/metrics.json`) for future reference.

8. **Visualizing Predictions:**
   - A scatter plot is created to compare actual vs. predicted values of the target variable.
   - The plot is saved as an image in the `plots` directory.

9. **Saving the Model:**
   - The trained model is saved as a pickle file in the `models` directory for future use.

---

### **Key Findings:**
- The dataset contained missing values in several columns, which were successfully handled using interpolation and mode filling.
- Outliers were detected and removed using the Isolation Forest algorithm, improving the quality of the dataset.
- Feature scaling ensured that all features were on a similar scale, which is crucial for many machine learning algorithms.
- Multiple regression models were evaluated, and their performance was logged for comparison.
- The best-performing model was saved for future predictions.

---

### **Technologies Used:**
- **Python Libraries:**
  - `pandas` and `numpy` for data manipulation.
  - `matplotlib` and `seaborn` for data visualization.
  - `scikit-learn` for preprocessing, modeling, and evaluation.
  - `pickle` for saving the trained model.
- **Machine Learning Models:**
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Random Forest
  - Gradient Boosting

---

### **Future Work:**
- **Hyperparameter Tuning:** Perform hyperparameter tuning for the models to further improve performance.
- **Feature Engineering:** Explore additional feature engineering techniques to enhance model accuracy.
- **Model Comparison:** Compare the performance of more advanced models, such as Support Vector Regression (SVR) or Neural Networks.
- **Deployment:** Deploy the best-performing model as a web service or API for real-time predictions.

---

### **Conclusion:**
This project provides a comprehensive analysis of a housing dataset and demonstrates the process of building and evaluating predictive models for estimating housing prices. The project highlights the importance of data preprocessing, visualization, and model evaluation in achieving accurate predictions. The final model can be used for further analysis or deployed in a real-world application to predict housing prices based on given features.


---

## Summary of model performance

| Model | Mean R² Score | Std Dev R² | MSE | RMSE | MAE | R² |
|---|---|---|---|---|---|---|
| LinearRegression() | 0.783 | 0.032 | 11.068 | 3.327 | 2.553 | 0.76 |
| KNeighborsRegressor(n_neighbors=3) | 0.822 | 0.033 | 10.841 | 3.293 | 2.16 | 0.76 |
| KNeighborsRegressor(n_neighbors=3, weights='distance') | 0.828 | 0.030 | 10.663 | 3.265 | 2.074 | 0.77 |
| KNeighborsRegressor() | 0.798 | 0.038 | 11.862 | 3.444 | 2.241 | 0.74 |
| KNeighborsRegressor(weights='distance') | 0.812 | 0.033 | 11.022 | 3.320 | 2.106 | 0.76 |
| KNeighborsRegressor(n_neighbors=10) | 0.799 | 0.043 | 11.885 | 3.447 | 2.410 | 0.74 |
| KNeighborsRegressor(n_neighbors=10, weights='distance') | 0.812 | 0.040 | 10.915 | 3.304 | 2.226 | 0.76 |
| RandomForestRegressor(n_estimators=10, random_state=42) | 0.812 | 0.053 | 8.684 | 2.947 | 2.092 | 0.81 |
| RandomForestRegressor(criterion='absolute_error', n_estimators=10, random_state=42) | 0.820 | 0.054 | 8.465 | 2.909 | 2.086 | 0.82 |
| RandomForestRegressor(criterion='friedman_mse', n_estimators=10, random_state=42) | 0.817 | 0.050 | 8.908 | 2.985 | 2.132 | 0.81 |
| RandomForestRegressor(n_estimators=20, random_state=42) | 0.833 | 0.045 | 8.209 | 2.865 | 2.100 | 0.82 |
| RandomForestRegressor(criterion='absolute_error', n_estimators=20, random_state=42) | 0.843 | 0.040 | 7.652 | 2.766 | 1.983 | 0.83 |
| RandomForestRegressor(criterion='friedman_mse', n_estimators=20, random_state=42) | 0.835 | 0.047 | 8.150 | 2.855 | 2.095 | 0.82 |
| RandomForestRegressor(n_estimators=50, random_state=42) | 0.844 | 0.040 | 7.253 | 2.693 | 1.978 | 0.84 |
| RandomForestRegressor(criterion='absolute_error', n_estimators=50, random_state=42) | 0.849 | 0.034 | 7.331 | 2.708 | 1.961 | 0.84 |
| RandomForestRegressor(criterion='friedman_mse', n_estimators=50, random_state=42) | 0.845 | 0.042 | 7.617 | 2.760 | 2.001 | 0.83 |
| RandomForestRegressor(random_state=42) | 0.850 | 0.040 | 7.077 | 2.660 | 1.979 | 0.85 |
| RandomForestRegressor(criterion='absolute_error', random_state=42) | 0.852 | 0.039 | 7.301 | 2.702 | 1.995 | 0.84 |
| RandomForestRegressor(criterion='friedman_mse', random_state=42) | 0.850 | 0.040 | 7.176 | 2.679 | 1.969 | 0.84 |
| GradientBoostingRegressor() | 0.852 | 0.043 | 7.051 | 2.655 | 1.995 | 0.85 |
| GradientBoostingRegressor(loss='absolute_error') | 0.851 | 0.038 | 7.544 | 2.747 | 2.058 | 0.84 |
| GradientBoostingRegressor(loss='huber') | 0.867 | 0.037 | 6.724 | 2.593 | 1.834 | 0.85 |
| GradientBoostingRegressor(loss='quantile') | 0.568 | 0.150 | 18.626 | 4.316 | 3.541 | 0.60 |


## Regression Plots for Best Models

![alt text](plots/LinearRegression().png)

![alt text](plots/KNeighborsRegressor(weights='distance').png)

![alt text](plots/RandomForestRegressor(random_state=42).png)

![alt text](plots/GradientBoostingRegressor(loss='huber').png)

