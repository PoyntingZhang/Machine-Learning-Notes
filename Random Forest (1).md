# Random Forest \:Theory

![](https://img-blog.csdnimg.cn/20200710141430652.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3pob25nX2RkYmI=,size_16,color_FFFFFF,t_70#pic_center)

## Bagging or Boosting

Random forest is a kind of bagging algorithm.&#x20;

Bagging is a random sampling algorithm. The difference between it and boosting is that bagging algorithm's decision trees are independent of each other.

## &#x20;The Randomness

1.  random forest grows each decision tree from the randomly drawn sample
2.  random forest's splitting nodes are each time picked from a random subset of the total features.

    The overall performance is:

    *   &#x20;it reduces the total variance with de-correlation&#x20;
    *   it introduced a little bias(due to approximately one third of the total sample has not been drawn)

        &#x20;

## Pros and Cons

1.  training speed can be highly boosted by parallel processing
2.  prevent over fitting&#x20;
3.  can return the contribution of each feature to the final prediction

# RandomForestClassifier

### parameters

1.  n\_estimators : the numbers of the decision trees
2.  bootstrap: whether sampling with replacement or not, the default value is True
3.  oob\_score: whether assess the model with the out of bag sample, default is False
4.  max\_features: the maximum number of features considered each time constructing a splitting node
5.  max\_depth\:maximum depth of the decision tree
6.  min-samples-leaf
7.  min-samples-split
8.  max-leaf-nodes
9.  min-impurity-decrease
10. criterion: the criterion of node splitting, te default is Gini while we can also set it to entropy

    ### import the packages

```python
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

```

### common methods

```python
rfc=RandomForestClassifier()
rfc.decision_path(X) # return the decision path in the random forest
rfc.fit(X,y) #train the model using training set X with label y
rfc.predict(x) # predict the targets for the test set x
rfc.predict_proba(X) # predict the pseudo-probability for X
rfc.get_params(deep=True) # check the parameters


```

### GridSearchCV（网格调参）

```python
rfc=RandomForestClassifier()
parameters={'n_estimators':range(30,80,10),'max_depth':range(3,10,2),'min_samples_leaf':[5,6,7],'max_features':[1,2,3]}
grid_rfc=GridSearchCV(rfc,parameters,scoring='f1_macro')
# f1_macro refers to the "macro-averaged F1-score" that can be used as the scoring metric for model evaluation.The F1-score is a metric that combines precision and recall into a single value, and it ranges from 0 to 1, with 1 being the best score.
grid_rfc.fit(X,y)
grid_rfc.best_params_
grid_rfc.best_score_



```

The `GridSearchCV` object that is returned has the following key attributes and methods:

*   `grid_rfc.best_params_`: This returns a dictionary containing the hyperparameter values that resulted in the best macro-averaged F1-score during the grid search.
*   `grid_rfc.best_score_`: This returns the actual best macro-averaged F1-score found during the grid search.
*   `grid_rfc.cv_results_`: This returns a dictionary containing detailed information about the cross-validation results for each set of hyperparameters tried.
*   `grid_rfc.fit(X, y)`: This method is used to fit the grid search model to the training data `X` and `y`.
*   `grid_rfc.predict(X)`: This method can be used to make predictions on new data `X` using the best model found during the grid search.

### Re-prediction using the new training parameters

```python
rfc_param=RandomForestClassifier(n_estimators=50,max_depth=3,max_features=2,min_samples_leaf=7)
rfc_param.fit(X_train,y_train)
pred=rfc_param.predict(X_test)
print(metrics.classification_report(pred,y_test))

```

### Read important features

```python
feature_important=rfc_param.feature_importances_
feature_name=load_iris().feature_names
plt.barh(range(len(feature_name)),feature_important,tick_label=feature_name)

```

### When the data set  is large and with complex features

#### 1. grid search for n\_estimators&#x20;

```python
param_test1 = {'n_estimators':range(10,71,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100,
                                  min_samples_leaf=20,max_depth=8,max_features='sqrt' ,random_state=10), 
                       param_grid = param_test1, scoring='roc_auc',cv=5)
gsearch1.fit(X,y)
gsearch1.best_params_

```

#### 2.grid search for `max_`*`depth` and `min_`*`samples_split`

```python
param_test2 = {'max_depth':range(3,14,2), 'min_samples_split':range(50,201,20)}
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, 
                                  min_samples_leaf=20,max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
gsearch2.fit(X,y)
gsearch1.best_params_

```

#### 3. grid search for `min`*`_samples_split` and `mins_sample_split`*

```python
param_test3 = {'min_samples_split':range(80,150,20), 'min_samples_leaf':range(10,60,10)}
gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13,
                                  max_features='sqrt' ,oob_score=True, random_state=10),
   param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
gsearch3.fit(X,y)
gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
```

#### 4. grid search for `max_features`

```python
param_test4 = {'max_features':range(3,11,2)}
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20 ,oob_score=True, random_state=10),
   param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
gsearch4.fit(X,y)
gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

```

#### 5. regression with the best parameters

```python
rf2 = RandomForestClassifier(n_estimators= 60, max_depth=13, min_samples_split=120,
                                  min_samples_leaf=20,max_features=7 ,oob_score=True, random_state=10)
rf2.fit(X,y)
print rf2.oob_score_

```

# RandomForestRegressor

```python
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


X,y = load_boston(return_X_y=True)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2020)



rfr = RandomForestRegressor()
rfr.fit(X_train,y_train)

pred = rfr.predict(X_test)
print(metrics.mean_squared_error(pred,y_test))
# RandomForest 特征重要性
# 获取重要性
feat_important = rfr.feature_importances_
# 特征名
feat_name = load_boston().feature_names
plt.barh(range(len(feat_name)),feat_important,tick_label=feat_name)

```

