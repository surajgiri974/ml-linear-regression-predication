import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load Excel file
file_path = 'information.xlsx'
sheet_name = 'Sheet1'
df = pd.read_excel(file_path, sheet_name=sheet_name)

# # Convert duration columns to total seconds
# df['Total Duration Gross'] = pd.to_timedelta(df['Total Duration Gross']).dt.total_seconds()
# df['Total Duration Net'] = pd.to_timedelta(df['Total Duration Net']).dt.total_seconds()

# Select features and target
feature_columns = ['Record Count', 'Total Duration Gross', 'Total Duration Net']
target_column = 'Final'
# Convert duration columns to total seconds
df['Total Duration Gross'] = pd.to_timedelta(df['Total Duration Gross']).dt.total_seconds()
df['Total Duration Net'] = pd.to_timedelta(df['Total Duration Net']).dt.total_seconds()

# Drop rows with missing values
df.dropna(subset=feature_columns + [target_column], inplace=True)

X = df[feature_columns]
y = df[target_column]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print("Mean Squared Error:", mse)
print("R² Score:", r2)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
