#%% Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


#%% Reading csv files and transforming them into dataframes
journal_entries = pd.read_csv('journal_entries.csv')
sleep_data = pd.read_csv('sleeps.csv')
workouts_data = pd.read_csv('workouts.csv')
physiological_cycles = pd.read_csv('physiological_cycles.csv')

#%% Merging all datasets into one with custom suffixes to avoid column name conflicts
merged_df = pd.merge(journal_entries, sleep_data, on=['Cycle start time', 'Cycle end time'], how='outer', suffixes=('', '_sleep'))
merged_df = pd.merge(merged_df, workouts_data, on=['Cycle start time', 'Cycle end time'], how='outer', suffixes=('', '_workouts'))
merged_df = pd.merge(merged_df, physiological_cycles, on=['Cycle start time', 'Cycle end time'], how='outer', suffixes=('', '_physio'))

#%% After merging, check for any duplicated columns and decide whether to keep, rename, or drop them
print(merged_df.columns)  # This will help you identify duplicated columns)

# Data Preprocessing
#%% Converting datetime columns
merged_df['Cycle start time'] = pd.to_datetime(merged_df['Cycle start time'])
merged_df['Cycle end time'] = pd.to_datetime(merged_df['Cycle end time'])

# Handling missing values
#%% Identify numeric columns in the DataFrame
numeric_cols = merged_df.select_dtypes(include=np.number).columns

#%% Calculate medians only for these numeric columns
medians = merged_df[numeric_cols].median()

#%% Fill missing values in numeric columns with their respective medians
merged_df[numeric_cols] = merged_df[numeric_cols].fillna(medians)

# EDA and Visualization
#%% Distribution of 'Recovery Score'
plt.figure(figsize=(10, 6))
sns.histplot(merged_df['Recovery score %'], kde=True, bins=30)
plt.title('Distribution of Recovery Score')
plt.show()

#%% Exclude non-numeric columns before computing the correlation matrix
numeric_cols = merged_df.select_dtypes(include=[np.number])
#%% Compute the correlation matrix using only numeric columns
corr_matrix = numeric_cols.corr()
#%%Plotting the heatmap of the correlation matrix
plt.figure(figsize=(20, 30))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

#%% Pairplot for selected features
sns.pairplot(merged_df[['Recovery score %', 'Sleep performance %', 'Activity Strain', 'Energy burned (cal)_physio']])
plt.show()

#%% Feature Engineering
X = merged_df.select_dtypes(include=[np.number]).drop('Recovery score %', axis=1)
y = merged_df['Recovery score %']

#%% Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#%% Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

#%% Model Building
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

#%% Predictions and Evaluation
y_pred = model.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R-squared: {r2_score(y_test, y_pred)}")

#%% Displaying feature importances
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.show()