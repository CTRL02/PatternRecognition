import re
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score




data = pd.read_csv('ApartmentRentPrediction.csv')

columns_to_drop = ['id', 'category', 'body', 'currency', 'fee',
                   'price', 'title']

data.drop(columns=columns_to_drop, inplace=True)


# drop Monthly|Weekly Type from price_type
data = data[data['price_type'] != 'Monthly|Weekly']
# Remove weekly/monthly from price_display
def remove_after_space(value):
    if re.search(r'\s\w', value):
        return value.split()[0]
    return value

data['price_display'] = data['price_display'].apply(remove_after_space)


# Remove $ and ,
def remove_symbols(value):
    return value.replace('$', '').replace(',', '')


data['price_display'] = data['price_display'].apply(remove_symbols)

# Compute all values to rent per month
for index, row in data.iterrows():
    if row['price_type'] == 'Weekly':
        data.loc[index, 'price_display'] = str(int(row['price_display']) * 4)

data.drop(columns='price_type', inplace=True)
data['price_display'] = data['price_display'].astype('int64')

data['bathrooms'].fillna(data['bathrooms'].mode()[0], inplace=True)
data['amenities'].fillna(data['amenities'].mode()[0], inplace=True)
data['bedrooms'].fillna(data['bedrooms'].mode()[0], inplace=True)
data['bathrooms'] = data['bathrooms'].astype('int64')
data['bedrooms'] = data['bedrooms'].astype('int64')
data['square_feet'].fillna(data['square_feet'].mean(), inplace=True)
data['has_photo'].fillna(data['has_photo'].mode()[0], inplace=True)
data['pets_allowed'].fillna(data['pets_allowed'].mode()[0], inplace=True)
data['source'].fillna(data['source'].mode()[0], inplace=True)
data['state'].fillna(data['state'].mode()[0], inplace=True)
data['latitude'].fillna(data['latitude'].mode()[0], inplace=True)
data['longitude'].fillna(data['longitude'].mode()[0], inplace=True)
data['price_display'].fillna(data['price_display'].mean(), inplace=True)

mode_address = data['address'].mode()[0]
data['address'] = data['address'].fillna(mode_address)
data['street_name'] = data['address'].str.extract(r'([A-Za-z\s]+)', expand=False).str.strip()
data['address'] = data['address'].str.extract(r'(\d+)')
data['address'] = data['address'].astype('int64')
label_encoder = LabelEncoder()
data['street_name'] = label_encoder.fit_transform(data['street_name'])

# label encoding for categorical features
cols = ('cityname', 'state', 'has_photo', 'pets_allowed', 'source')
encoder = LabelEncoder()
for col in cols:
    data[col] = encoder.fit_transform(data[col])

# Transfer time column to actual years
data['time'] = pd.to_datetime(data['time'], unit='s')
current_datetime = datetime.now()
data['time'] = (current_datetime - data['time']).dt.days / 365.25

text_column = data['amenities']
text_column = text_column.astype(str)
words = [word.strip() for text in text_column for word in text.split(',')]
unique_words = set(words)
word_dict = {}
temp = pd.DataFrame()
for word in unique_words:
    data[word] = 0
    temp[word] = 0

for index, value in data['amenities'].items():
    amenities_list = value.split(',')
    for word in amenities_list:
        word = word.strip()
        if word in data.columns:
            data.at[index, word] = 1
            temp.at[index, word] = 1

data.drop(columns=['amenities'], inplace=True)

# Outliers for price
mean_price = np.mean(data['price_display'])
std_price = np.std(data['price_display'])
threshold = 3 * std_price
outlier_mask = (data['price_display'] - mean_price).abs() > threshold
data = data[~outlier_mask]

from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()
# Scaling square_feet
scaled_column = data['square_feet'].values.reshape(-1, 1)
scaled_column = scaler.fit_transform(scaled_column)
data['square_feet'] = scaled_column

# Scaling state
scaled_column = data['state'].values.reshape(-1, 1)
scaled_column = scaler.fit_transform(scaled_column)
data['state'] = scaled_column


# Visualizing features after correlation analysis

corre = data.corr()
best_features = corre.index[abs(corre['price_display']) >= 0.1]
plt.figure(figsize=(35, 23))
sns.heatmap(corre, annot=True)
plt.show()

best_features = best_features.drop('price_display')
best_features = list(best_features)
best_features.append('latitude')
best_features = pd.Index(best_features)
print('best features : ')
print(best_features)
X = data[best_features]
Y = data['price_display']

# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# # Define XGBoost regressor and set hyperparameters
# xgb_model = xgb.XGBRegressor(objective ='reg:squarederror',
#                              colsample_bytree = 0.3,
#                              learning_rate = 1,
#                              max_depth = 2,
#                              alpha = 5,
#                              n_estimators = 900)
# from sklearn.metrics import accuracy_score

# xgb_model.fit(X_train, y_train)
# y_pred = xgb_model.predict(X_test)
# rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# print("RMSE for xgboost:", rmse)
# r2 = r2_score(y_test, y_pred)
# print("R^2 score for xg boost accuracy :", r2*100)