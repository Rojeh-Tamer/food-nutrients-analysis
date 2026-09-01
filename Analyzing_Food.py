import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the dataset

df = pd.read_csv("nutrients_csvfile.csv")

# Data Exploration

# print(df.head())
# print(df.info())
# print(df.describe())
# print(df['Grams'].unique())
# print(df['Calories'].unique())
# print(df['Protein'].unique())
# print(df['Fat'].unique())
# print(df['Fiber'].unique())
# print(df['Carbs'].unique())

# Data preprocessing

# 1. Casting the 'Grams', 'Calories', 'Protein', 'Fat', 'Sat.Fat', 'Carbs', 'Fiber' to numeric
df = df.drop(columns=['Measure'])
df['Grams'] = df['Grams'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Calories'] = df['Calories'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Protein'] = df['Protein'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Fat'] = df['Fat'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Sat.Fat'] = df['Sat.Fat'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Carbs'] = df['Carbs'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()
df['Fiber'] = df['Fiber'].str.replace(',', '').str.replace('t', '0').str.replace('a', '0').str.strip()

df['Grams'] = pd.to_numeric(df['Grams'], errors='coerce')
df['Calories'] = pd.to_numeric(df['Calories'], errors='coerce')
df['Protein'] = pd.to_numeric(df['Protein'], errors='coerce')
df['Fat'] = pd.to_numeric(df['Fat'], errors='coerce')
df['Sat.Fat'] = pd.to_numeric(df['Sat.Fat'], errors='coerce')
df['Carbs'] = pd.to_numeric(df['Carbs'], errors='coerce')
df['Fiber'] = pd.to_numeric(df['Fiber'], errors='coerce')
# df.info()
# df.describe()

# 2. Data cleaning

# print(df.isnull().sum())
df['Calories'] = df['Calories'].fillna(df['Calories'].median())
df['Fat'] = df['Fat'].fillna(df['Fat'].median())
df['Sat.Fat'] = df['Sat.Fat'].fillna(df['Sat.Fat'].median())

# print(df.isnull().sum())

def remove_outliers(df, column):
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df

df = remove_outliers(df, 'Calories')
df = remove_outliers(df, 'Protein')
df = remove_outliers(df, 'Carbs')
df = remove_outliers(df, 'Fiber')
df = remove_outliers(df, 'Fat')
df = remove_outliers(df, 'Sat.Fat')

# print(df.duplicated().sum())
df = df.drop_duplicates()
# print(df.duplicated().sum())

# 3. Data transformation

df = pd.get_dummies(df, columns=['Category'], drop_first=True)
# print(df.head())


# Data Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Fat', y='Calories', alpha=0.5, color='blue')
plt.title('Fat vs Calories (Scatter Plot)')
plt.xlabel('Fat (g)')
plt.ylabel('Calories')
plt.show()

plt.figure(figsize=(9, 5))
sns.boxplot(data=df[['Calories', 'Protein', 'Fat', 'Carbs']])
plt.title('Outliers Check for Main Nutritional Values')
plt.ylabel('Values')
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(df['Calories'], kde=True, color='skyblue', bins=20)
plt.title('Distribution of Calories')
plt.xlabel('Calories')
plt.ylabel('Frequency (Count)')
plt.show()

plt.figure(figsize=(8, 6))
numeric_cols = ['Calories', 'Protein', 'Fat', 'Sat.Fat', 'Carbs', 'Fiber', 'Grams']
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='Blues', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()
# Data Modelling

# 1. Splitting Features and Target
x = df.drop(columns=['Food', 'Calories'])
y = df['Calories']

# 2. Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Feature Scaling
scaler = MinMaxScaler()

# 4. Training the model

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"R² Score: {r2:.4f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")