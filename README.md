# ml-boston-house-prediction

---

##Summary of model performance

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