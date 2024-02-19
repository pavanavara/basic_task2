Python 3.12.1 (tags/v3.12.1:2305ca5, Dec  7 2023, 22:03:25) [MSC v.1937 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> import numpy as np
>>> import pandas as pd
>>> import matplotlib.pyplot as plt
>>> import seaborn as sns
impo
>>> import mpl_toolkits
>>> data=pd.read_csv("C:\\Users\\LOHIDAS\\Downloads\\house-price-prediction-master (1)\\house-price-prediction-master\\kc_house_data.csv")
>>> data.head()
           id             date     price  ...     long  sqft_living15  sqft_lot15
0  7129300520  20141013T000000  221900.0  ... -122.257           1340        5650
1  6414100192  20141209T000000  538000.0  ... -122.319           1690        7639
2  5631500400  20150225T000000  180000.0  ... -122.233           2720        8062
3  2487200875  20141209T000000  604000.0  ... -122.393           1360        5000
4  1954400510  20150218T000000  510000.0  ... -122.045           1800        7503

[5 rows x 21 columns]
>>> data.describe()
                 id         price  ...  sqft_living15     sqft_lot15
count  2.161300e+04  2.161300e+04  ...   21613.000000   21613.000000
mean   4.580302e+09  5.400881e+05  ...    1986.552492   12768.455652
std    2.876566e+09  3.671272e+05  ...     685.391304   27304.179631
min    1.000102e+06  7.500000e+04  ...     399.000000     651.000000
25%    2.123049e+09  3.219500e+05  ...    1490.000000    5100.000000
50%    3.904930e+09  4.500000e+05  ...    1840.000000    7620.000000
75%    7.308900e+09  6.450000e+05  ...    2360.000000   10083.000000
max    9.900000e+09  7.700000e+06  ...    6210.000000  871200.000000

[8 rows x 20 columns]
data['bedrooms'].value_counts().plot(kind='bar')
<Axes: xlabel='bedrooms'>
plt.title('number of Bedroom')
Text(0.5, 1.0, 'number of Bedroom')
plt.xlabel('Bedrooms')
Text(0.5, 0, 'Bedrooms')
plt.ylabel('Count')
Text(0, 0.5, 'Count')
sns.despine
<function despine at 0x000001AE3CC8F9C0>
plt.figure(figsize=(10,10))
<Figure size 1000x1000 with 0 Axes>
sns.jointplot(x=data.lat.values, y=data.long.values, size=10)
<seaborn.axisgrid.JointGrid object at 0x000001AE4D4628D0>
plt.ylabel('Longitude', fontsize=12)
Text(37.722222222222214, 0.5, 'Longitude')
plt.xlabel('Latitude', fontsize=12)
Text(0.5, 36.72222222222221, 'Latitude')
plt.show()
plt1 = plt.figure(figsize=(10,10))
sns.despine
<function despine at 0x000001AE3CC8F9C0>
plt.scatter(data.price,data.sqft_living)
<matplotlib.collections.PathCollection object at 0x000001AE52D8F7A0>
plt.title("Price vs Square Feet")
Text(0.5, 1.0, 'Price vs Square Feet')
plt.show()
plt.scatter(data.price,data.long)
<matplotlib.collections.PathCollection object at 0x000001AE52D562A0>
plt.title("Price vs Location of the area")
Text(0.5, 1.0, 'Price vs Location of the area')
plt.show()
plt.scatter(data.price,data.lat)
<matplotlib.collections.PathCollection object at 0x000001AE50174830>
plt.xlabel("Price")
Text(0.5, 0, 'Price')
plt.ylabel('Latitude')
Text(0, 0.5, 'Latitude')
plt.title("Latitude vs Price")
Text(0.5, 1.0, 'Latitude vs Price')
plt.show()
plt.scatter(data.bedrooms,data.price)
<matplotlib.collections.PathCollection object at 0x000001AE52820770>
plt.title("Bedroom and Price ")
Text(0.5, 1.0, 'Bedroom and Price ')
plt.xlabel("Bedrooms")
Text(0.5, 0, 'Bedrooms')
plt.ylabel("Price")
Text(0, 0.5, 'Price')
plt.show()
sns.despine
<function despine at 0x000001AE3CC8F9C0>
plt.scatter((data['sqft_living']+data['sqft_basement']),data['price'])
<matplotlib.collections.PathCollection object at 0x000001AE528AF7A0>
plt.show()
plt.scatter(data.waterfront,data.price)
<matplotlib.collections.PathCollection object at 0x000001AE5265A780>
plt.title("Waterfront vs Price ( 0= no waterfront)")
Text(0.5, 1.0, 'Waterfront vs Price ( 0= no waterfront)')
plt.show()
train1 = data.drop(['id', 'price'],axis=1)
train1.head()
              date  bedrooms  bathrooms  ...     long  sqft_living15  sqft_lot15
0  20141013T000000         3       1.00  ... -122.257           1340        5650
1  20141209T000000         3       2.25  ... -122.319           1690        7639
2  20150225T000000         2       1.00  ... -122.233           2720        8062
3  20141209T000000         4       3.00  ... -122.393           1360        5000
4  20150218T000000         3       2.00  ... -122.045           1800        7503

[5 rows x 19 columns]
data.floors.value_counts().plot(kind='bar')
<Axes: xlabel='floors'>
plt.show()
plt.scatter(data.floors,data.price)
<matplotlib.collections.PathCollection object at 0x000001AE501EB710>
plt.show()
plt.scatter(data.condition,data.price)
<matplotlib.collections.PathCollection object at 0x000001AE52823710>
plt.show()
plt.scatter(data.zipcode,data.price)
<matplotlib.collections.PathCollection object at 0x000001AE52D65100>
plt.title("Which is the pricey location by zipcode?")
Text(0.5, 1.0, 'Which is the pricey location by zipcode?')
plt.show()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
labels = data['price']
conv_dates = [1 if values == 2014 else 0 for values in data.date ]
data['date'] = conv_dates
train1 = data.drop(['id', 'price'],axis=1)
from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(train1 , labels , test_size = 0.10,random_state =2)
from sklearn.ensemble import GradientBoostingRegressor
reg.fit(x_train,y_train)
LinearRegression()
reg.score(x_test,y_test)
0.7320342760357731
from sklearn import ensemble
clf = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 5, min_samples_split = 2,
          learning_rate = 0.1, loss = 'ls')
clf = GradientBoostingRegressor(loss='squared_error')
clf.fit(x_train, y_train)
GradientBoostingRegressor()
clf.score(x_test,y_test)
0.8947629640003595
params = {'n_estimators': 100}
KeyboardInterrupt
t_sc = np.zeros((params['n_estimators']), dtype=np.float64)
y_pred = reg.predict(x_test)
for i,y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i]=clf.loss_(y_test,y_pred)

    
Traceback (most recent call last):
  File "<pyshell#73>", line 2, in <module>
    t_sc[i]=clf.loss_(y_test,y_pred)
AttributeError: 'GradientBoostingRegressor' object has no attribute 'loss_'. Did you mean: 'loss'?
clf = GradientBoostingRegressor(loss='ls')
clf.fit(X_train, y_train)
Traceback (most recent call last):
  File "<pyshell#75>", line 1, in <module>
    clf.fit(X_train, y_train)
NameError: name 'X_train' is not defined. Did you mean: 'x_train'?
clf.fit(x_train, y_train)
Traceback (most recent call last):
  File "<pyshell#76>", line 1, in <module>
    clf.fit(x_train, y_train)
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\base.py", line 1344, in wrapper
    estimator._validate_params()
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\base.py", line 666, in _validate_params
    validate_parameter_constraints(
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\utils\_param_validation.py", line 95, in validate_parameter_constraints
    raise InvalidParameterError(
sklearn.utils._param_validation.InvalidParameterError: The 'loss' parameter of GradientBoostingRegressor must be a str among {'quantile', 'absolute_error', 'huber', 'squared_error'}. Got 'ls' instead.
for i,y_pred in enumerate(clf.staged_predict(x_test)):
    t_sc[i]=clf.ls(y_test,y_pred)


Warning (from warnings module):
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\base.py", line 486
    warnings.warn(
UserWarning: X has feature names, but GradientBoostingRegressor was fitted without feature names
Traceback (most recent call last):
  File "<pyshell#78>", line 1, in <module>
    for i,y_pred in enumerate(clf.staged_predict(x_test)):
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_gb.py", line 2144, in staged_predict
    for raw_predictions in self._staged_raw_predict(X):
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_gb.py", line 994, in _staged_raw_predict
    raw_predictions = self._raw_predict_init(X)
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_gb.py", line 947, in _raw_predict_init
    self._check_initialized()
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\ensemble\_gb.py", line 610, in _check_initialized
    check_is_fitted(self)
  File "C:\Users\LOHIDAS\AppData\Local\Programs\Python\Python312\Lib\site-packages\sklearn\utils\validation.py", line 1544, in check_is_fitted
    raise NotFittedError(msg % {"name": type(estimator).__name__})
sklearn.exceptions.NotFittedError: This GradientBoostingRegressor instance is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.
testsc = np.arange((params['n_estimators']))+1
plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
plt.subplot(1, 2, 1)
<Axes: >
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
Traceback (most recent call last):
  File "<pyshell#82>", line 1, in <module>
    plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
AttributeError: 'GradientBoostingRegressor' object has no attribute 'train_score_'
clf = GradientBoostingRegressor(warm_start=True)
train_scores = []
test_scores = []
for i in range(100):  # Example: iterate for 100 iterations
    clf.fit(X_train, y_train)
    train_score = clf.train_score_[-1]  # Get the last training score
    test_score = clf.loss_(y_test, clf.predict(X_test))  # Calculate test score
    train_scores.append(train_score)
    test_scores.append(test_score)

    
Traceback (most recent call last):
  File "<pyshell#87>", line 2, in <module>
    clf.fit(X_train, y_train)
NameError: name 'X_train' is not defined. Did you mean: 'x_train'?
for i in range(100):  # Example: iterate for 100 iterations
    clf.fit(x_train, y_train)
    train_score = clf.train_score_[-1]  # Get the last training score
    test_score = clf.loss_(y_test, clf.predict(x_test))  # Calculate test score
    train_scores.append(train_score)
    test_scores.append(test_score)

    
GradientBoostingRegressor(warm_start=True)
Traceback (most recent call last):
  File "<pyshell#89>", line 4, in <module>
    test_score = clf.loss_(y_test, clf.predict(x_test))  # Calculate test score
AttributeError: 'GradientBoostingRegressor' object has no attribute 'loss_'. Did you mean: 'loss'?


plt.plot(testsc,t_sc,'r-',label = 'set dev test')
[<matplotlib.lines.Line2D object at 0x000001AE54D8E6C0>]
plt.show()
plt.plot(testsc,clf.train_score_,'b-',label= 'Set dev train')
[<matplotlib.lines.Line2D object at 0x000001AE57F589B0>]
plt.plot(testsc,t_sc,'r-',label = 'set dev test')
[<matplotlib.lines.Line2D object at 0x000001AE57F653D0>]
plt.show()
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
pca = PCA()
pca.fit_transform(scale(train1))
array([[-2.64785461e+00, -4.54699955e-02, -3.16665762e-01, ...,
        -7.94687728e-02, -3.18219319e-16,  0.00000000e+00],
       [-2.34485164e-01,  1.68297114e+00, -7.61521725e-01, ...,
         9.81487761e-01, -2.01119690e-14, -0.00000000e+00],
       [-2.57007792e+00, -6.14344122e-01,  3.49292423e-01, ...,
        -1.38570764e-01,  3.54280486e-15,  0.00000000e+00],
       ...,
       [-2.41985641e+00, -1.10027662e+00, -1.46293798e+00, ...,
         9.66785881e-01,  7.33823083e-17, -0.00000000e+00],
       [ 3.32183025e-01, -1.88043103e+00, -1.04412760e+00, ...,
        -3.97449542e-01, -4.34087160e-17,  0.00000000e+00],
       [-2.43180432e+00, -1.08505981e+00, -1.47248379e+00, ...,
         9.53674385e-01,  7.35227481e-17, -0.00000000e+00]])
