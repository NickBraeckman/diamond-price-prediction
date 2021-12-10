# train.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.compose import make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

def get_metrics(y_true: np.array, y_pred: np.array) -> dict:
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {'r2': r2,'mse': mse}

rnd_state = 998544

# Collect data
data = pd.read_csv('../data/raw/diamonds.csv', index_col='Unnamed: 0')
data.rename(columns={'Unnamed: 0': 'id'})
data['price_per_carat'] = data.price / data.carat

# Prepare data
y = data.price_per_carat
X = data.drop(['price_per_carat', 'price', 'carat'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y,
    test_size=0.15,
    random_state=rnd_state
)

"""X_train, X_valid, y_train, y_valid = train_test_split(
    X_tr_va,
    y_tr_va,
    test_size=0.15,
    random_state=rnd_state
)"""

print('train_size:', len(X_train.to_numpy()))
print('test_size:', len(X_test.to_numpy()))
#print('validation_size:', len(X_valid.to_numpy()))

ordinal_encoder = make_column_transformer(
    (
        OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan),
        make_column_selector(dtype_include=['category', 'object'], dtype_exclude=['bool', 'float64'])
    ),
    remainder='passthrough'
)

estimators = []

estimators.append(dict(
    name='LR', 
    estimator=LinearRegression(), 
    param_grid={}
))
estimators.append(dict(
    name='RF', 
    estimator=RandomForestRegressor(random_state=rnd_state), 
    param_grid={'estimator__n_estimators': [100, 500, 800]}
))
estimators.append(dict(
    name='GB', 
    estimator=GradientBoostingRegressor(random_state=rnd_state), 
    param_grid={
        'estimator__n_estimators': [100, 500, 800], 
        'estimator__learning_rate': [0.01, 0.05, 0.1]
    }
))

# Model training
for est in estimators:
    print('----', est['name'], '----')
    pipe = Pipeline([
        ('encoder', ordinal_encoder),
        ('estimator', est['estimator'])
    ])

    clf = GridSearchCV(pipe, est['param_grid'], scoring='neg_mean_squared_error', n_jobs=-1)
    clf.fit(X_train, y_train)
    print('Best parmaters: ', clf.best_params_)

    # Model evaluation
    y_true, y_pred = y_test, clf.predict(X_test)
    metrics = get_metrics(y_true, y_pred)

    for (k, v) in metrics.items():
        print('{}: {}'.format(k, v))