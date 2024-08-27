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
from pdpbox import pdp



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




X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True,random_state=10)
model = linear_model.LinearRegression()

model.fit(X_train, y_train)
#
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
#
train_rmse = metrics.mean_squared_error(y_train, y_train_pred, squared=False)
test_rmse = metrics.mean_squared_error(y_test, y_test_pred, squared=False)
r2_linear = r2_score(y_test, y_test_pred)
print(train_rmse)
print(test_rmse)
print("R^2 score accuracy linear:", r2_linear*100)

feature_name = 'bedrooms'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns, feature=feature_name)
fig, axes = pdp.pdp_plot(pdp_dist, feature_name)
plt.show()
feature_name = 'bathrooms'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns, feature=feature_name)
fig, axes = pdp.pdp_plot(pdp_dist, feature_name)
plt.show()
feature_name = 'square_feet'
pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns, feature=feature_name)
fig, axes = pdp.pdp_plot(pdp_dist, feature_name)
plt.show()
plt.legend()



##############################################################

# Polynomial Model


# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3,shuffle=True,random_state=10)
# poly_features = PolynomialFeatures(degree=3)

# # transforms the existing features to higher degree features.
# X_train_poly = poly_features.fit_transform(X_train)

# # fit the transformed features to Linear Regression
# poly_model1 = linear_model.LinearRegression()
# scores = cross_val_score(poly_model1, X_train_poly, y_train, scoring='neg_mean_squared_error', cv=5)
# model_1_score = abs(scores.mean())
# poly_model1.fit(X_train_poly, y_train)
# print("model 1 cross validation score is "+ str(model_1_score))
# poly_model1.fit(X_train_poly, y_train)

# # predicting on training data-set
# y_train_predicted = poly_model1.predict(X_train_poly)
# ypred = poly_model1.predict(poly_features.transform(X_test))

# # predicting on test data-set
# prediction = poly_model1.predict(poly_features.fit_transform(X_test))

# print('Mean Square Error', metrics.mean_squared_error(y_test, prediction))