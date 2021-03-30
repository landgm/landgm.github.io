---
title: "Logistic Regression"
excerpt: "로지스틱 회귀분석"
date: '2021-03-30'
categories : study
tags : [logistic,regression,mle]
use_math : true
---



#### Logistic Regression


* 로지스틱 회귀분석은 반응변수가 범주형변수일 때 사용이 가능합니다.(범주 2개 이상 가능)
* 설명변수는 범주형과 연속형 둘다 사용이 가능합니다.
* $E(y) =  {exp (\beta_0 + \beta_1) \over 1+exp(\beta_0+\beta_1 x)}$
* $E(y) =  p 확률을 의미합니다.$  성공하는 경우에 y =1 실패하는 경우에 y=0이라고 두면 p는 성공할 확률을 나타냅니다. p는 0~1사이의 값으로 분석가가 임의로 cutoff value를 설정하여서 cutoff value 밑은 y= 0 위는 y=1을 줍니다.

---


* $p' = {ln \left (p\over1-p \right ) } = {ln \left( E(y)\over 1-E(y) \right )} = ln \left( {{exp (\beta_0 + \beta_1) \over 1+exp(\beta_0+\beta_1 x)} \over {1 \over 1+exp(\beta_0+\beta_1 x)}} \right ) =  \beta_0+\beta_1 x $

* 이러한 변환을 로지스틱변환이라고 부릅니다.

#### 계수 추정법
* 3번 째 수식 $1- W_{beta}(x)입니다$
* ![로지스틱1](https://github.com/landgm/image/blob/master/img/logistic1.jpg?raw=true)
* ![로지스틱2](https://github.com/landgm/image/blob/master/img/logistic2.jpg?raw=true)

#### Odds(오즈)

* 사건이 일어날 확률 p = ${exp (\beta_0 + \beta_1) \over 1+exp(\beta_0+\beta_1 x)} 일때 odds  = {p \over 1-p}$ 
* 사건이 일어날 확률 /사건이 일어나지 않을 확률
* **odds**가 클 수록 사건이 일어날 확률도 커진다.
* 설명변수 $x_1의 단위가 하나 증가할 때의 사건이 일어날 오즈가 e^{\beta_1} *100 \%가 되는것이다$ 


#### titanic 실습



```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
```


```python
import warnings

warnings.filterwarnings('ignore')
```


```python
train= pd.read_csv("C:/Users/landg/Downloads/titanic/train.csv")
test= pd.read_csv("C:/Users/landg/Downloads/titanic/test.csv")

```


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  891 non-null    int64  
     1   Survived     891 non-null    int64  
     2   Pclass       891 non-null    int64  
     3   Name         891 non-null    object 
     4   Sex          891 non-null    object 
     5   Age          714 non-null    float64
     6   SibSp        891 non-null    int64  
     7   Parch        891 non-null    int64  
     8   Ticket       891 non-null    object 
     9   Fare         891 non-null    float64
     10  Cabin        204 non-null    object 
     11  Embarked     889 non-null    object 
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.7+ KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  418 non-null    int64  
     1   Pclass       418 non-null    int64  
     2   Name         418 non-null    object 
     3   Sex          418 non-null    object 
     4   Age          332 non-null    float64
     5   SibSp        418 non-null    int64  
     6   Parch        418 non-null    int64  
     7   Ticket       418 non-null    object 
     8   Fare         417 non-null    float64
     9   Cabin        91 non-null     object 
     10  Embarked     418 non-null    object 
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB



```python
def reduce_mem_usage(df):
    """ 
    iterate through all the columns of a dataframe and 
    modify the data type to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f}MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max <\
                  np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max <\
                   np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max <\
                   np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max <\
                   np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max <\
                   np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max <\
                   np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            else:
                pass
        else:
            df[col] = df[col].astype('category')
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f}MB')
    print(f'Decreased by {100*((start_mem - end_mem)/start_mem):.1f}%')
    
    return df
```


```python
train = reduce_mem_usage(train)
```

    Memory usage of dataframe is 0.08MB
    Memory usage after optimization is: 0.09MB
    Decreased by -12.9%



```python
train['Sex_clean'] = train['Sex'].astype('category').cat.codes
test['Sex_clean'] = test['Sex'].astype('category').cat.codes
```


```python
train['Sex_clean']
```




    0      1
    1      0
    2      0
    3      0
    4      1
          ..
    886    1
    887    0
    888    0
    889    1
    890    1
    Name: Sex_clean, Length: 891, dtype: int8




```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 13 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   PassengerId  891 non-null    int16   
     1   Survived     891 non-null    int8    
     2   Pclass       891 non-null    int8    
     3   Name         891 non-null    category
     4   Sex          891 non-null    category
     5   Age          714 non-null    float16 
     6   SibSp        891 non-null    int8    
     7   Parch        891 non-null    int8    
     8   Ticket       891 non-null    category
     9   Fare         891 non-null    float16 
     10  Cabin        204 non-null    category
     11  Embarked     889 non-null    category
     12  Sex_clean    891 non-null    int8    
    dtypes: category(5), float16(2), int16(1), int8(5)
    memory usage: 95.3 KB



```python
train['Embarked'].isnull().sum()
```




    2




```python
test['Embarked'].isnull().sum()
```




    0




```python
train['Embarked'].value_counts()
```




    S    644
    C    168
    Q     77
    Name: Embarked, dtype: int64




```python
train['Embarked'].fillna('S', inplace = True)
```


```python
train['Embarked_clean'] = train['Embarked'].astype('category').cat.codes
test['Embarked_clean'] = test['Embarked'].astype('category').cat.codes
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Sex_clean</th>
      <th>Embarked_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.250000</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.312500</td>
      <td>C85</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.925781</td>
      <td>NaN</td>
      <td>S</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.093750</td>
      <td>C123</td>
      <td>S</td>
      <td>0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.046875</td>
      <td>NaN</td>
      <td>S</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>




```python
train['Family'] = 1 + train['SibSp'] + train['Parch']
test['Family'] = 1 + test['SibSp'] + test['Parch']

```


```python
train['Solo'] = (train['Family']==1)
```


```python
test['Solo'] = (test['Family']==1)
```


```python
train['FareBin_4'] = pd.qcut(train['Fare'],4)
test['FareBin_4'] = pd.qcut(test['Fare'],4)
```


```python
train['FareBin_4'].value_counts()
```




    (7.91, 14.453]    224
    (-0.001, 7.91]    223
    (31.0, 512.5]     222
    (14.453, 31.0]    222
    Name: FareBin_4, dtype: int64




```python
train['Title'] = train['Name'].str.extract(' ([A-Za-z]+)\.',  expand= False)
```


```python
test['Title'] = test['Name'].str.extract(' ([A-Za-z]+)\.',  expand= False)
```


```python
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capy','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Other')
```


```python
train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
```


```python
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Other')
```


```python
test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')
```


```python
train['Title_clean'] = train['Title'].astype('category').cat.codes
test['Title_clean'] = test['Title'].astype('category').cat.codes
```


```python
train.groupby("Title")["Age"].transform("median")
```




    0      30.0
    1      35.0
    2      21.0
    3      35.0
    4      30.0
           ... 
    886    48.0
    887    21.0
    888    21.0
    889    30.0
    890    30.0
    Name: Age, Length: 891, dtype: float16




```python
train['Age'].plot.hist(bins=range(10,101,10),figsize=[15,8])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1d5a9776e48>




![png](output_35_1.png)



```python
train['Age'].fillna(train.groupby("Title")["Age"].transform("median"),inplace = True)
test['Age'].fillna(test.groupby("Title")["Age"].transform("median"),inplace = True)
```


```python
train.loc[train['Age'] <= 16, 'Age_clean'] = 0
train.loc[(train['Age'] > 16) & (train['Age'] <=26), 'Age_clean'] = 1
train.loc[(train['Age'] > 26) & (train['Age'] <=36), 'Age_clean'] = 2
train.loc[(train['Age'] > 36) & (train['Age'] <=62), 'Age_clean'] = 3
train.loc[(train['Age'] > 62), 'Age_clean'] = 4
```


```python
test.loc[test['Age'] <= 16, 'Age_clean'] = 0
test.loc[(test['Age'] > 16) & (test['Age'] <=26), 'Age_clean'] = 1
test.loc[(test['Age'] > 26) & (test['Age'] <=36), 'Age_clean'] = 2
test.loc[(test['Age'] > 36) & (test['Age'] <=62), 'Age_clean'] = 3
test.loc[(test['Age'] > 62), 'Age_clean'] = 4
```


```python
train['Fare'].fillna(train.groupby("Pclass")["Fare"].transform("median"), inplace = True)
test['Fare'].fillna(test.groupby("Pclass")["Fare"].transform("median"), inplace = True)

```


```python
pd.qcut(train['Fare'], 4)
```




    0      (-0.001, 7.91]
    1       (31.0, 512.5]
    2      (7.91, 14.453]
    3       (31.0, 512.5]
    4      (7.91, 14.453]
                ...      
    886    (7.91, 14.453]
    887    (14.453, 31.0]
    888    (14.453, 31.0]
    889    (14.453, 31.0]
    890    (-0.001, 7.91]
    Name: Fare, Length: 891, dtype: category
    Categories (4, interval[float64]): [(-0.001, 7.91] < (7.91, 14.453] < (14.453, 31.0] < (31.0, 512.5]]




```python
train.loc[train['Fare'] <= 17, 'Fare_clean'] = 0
train.loc[(train['Fare'] > 17) & (train['Fare'] <=30), 'Fare_clean'] = 1
train.loc[(train['Fare'] > 30) & (train['Fare'] <=100), 'Fare_clean'] = 2
train.loc[(train['Fare'] > 100), 'Fare_clean'] = 3

```


```python
train['Fare_clean'] = train['Fare_clean'].astype(int)
```


```python
test.loc[test['Fare'] <= 17, 'Fare_clean'] = 0
test.loc[(test['Fare'] > 17) & (test['Fare'] <=30), 'Fare_clean'] = 1
test.loc[(test['Fare'] > 30) & (test['Fare'] <=100), 'Fare_clean'] = 2
test.loc[(test['Fare'] > 100), 'Fare_clean'] = 3
test['Fare_clean'] = test['Fare_clean'].astype(int)
```


```python
train['Cabin'].str[:1].value_counts()
```




    C    59
    B    47
    D    33
    E    32
    A    15
    F    13
    G     4
    T     1
    Name: Cabin, dtype: int64




```python
mapping = {
    'A' :0,
    'B' :0.4,
    'C' : 0.8,
    'D' : 1.2,
    'E' : 1.6,
    'F' : 2.0,
    'G' : 2.4,
    'T' : 2.8
    
}
```


```python
train['Cabin_clean'] = train['Cabin'].str[:1]
```


```python
train['Cabin_clean'] = train['Cabin_clean'].map(mapping)
```


```python
train[['Pclass','Cabin_clean']].head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Cabin_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>3</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>2</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.groupby('Pclass')['Cabin_clean'].median()
```




    Pclass
    1    0.8
    2    1.8
    3    2.0
    Name: Cabin_clean, dtype: float64




```python
train['Cabin_clean'].fillna(train.groupby('Pclass')['Cabin_clean'].transform('median'),inplace = True)
```


```python
train['Cabin_clean']
```




    0      2.0
    1      0.8
    2      2.0
    3      0.8
    4      2.0
          ... 
    886    1.8
    887    0.4
    888    2.0
    889    0.8
    890    2.0
    Name: Cabin_clean, Length: 891, dtype: float64




```python
test['Cabin_clean'] = test['Cabin'].str[:1]
test['Cabin_clean'] = test['Cabin_clean'].map(mapping)
test['Cabin_clean'].fillna(test.groupby('Pclass')['Cabin_clean'].transform('median'), inplace=True)
```


```python
feature = [
    'Pclass',
    'SibSp',
    'Parch',
    'Sex_clean',
    'Embarked_clean',
    'Family',
    'Solo',
    'Title_clean',
    'Age_clean',
    'Fare_clean',
    'Cabin_clean'
]
```


```python
label = [
    'Survived'
]
```


```python
data = train[feature]
target = train[label]
```


```python
k_fold = KFold(n_splits=10, shuffle = True, random_state=0)
```


```python
x_train, x_test, y_train, y_test = train_test_split(data, target, random_state=0)
```


```python
x_train.shape, x_test.shape, y_train.shape, y_test.shape 
```




    ((668, 11), (223, 11), (668, 1), (223, 1))



#### Logistic Regression 해석



```python
import statsmodels.api as sm
```


```python
model = sm.Logit(y_train, x_train.astype(float))
results = model.fit(method = "newton") 

```

    Optimization terminated successfully.
             Current function value: 0.439386
             Iterations 7



```python
x_train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 668 entries, 105 to 684
    Data columns (total 11 columns):
     #   Column          Non-Null Count  Dtype  
    ---  ------          --------------  -----  
     0   Pclass          668 non-null    int8   
     1   SibSp           668 non-null    int8   
     2   Parch           668 non-null    int8   
     3   Sex_clean       668 non-null    int8   
     4   Embarked_clean  668 non-null    int8   
     5   Family          668 non-null    int8   
     6   Solo            668 non-null    bool   
     7   Title_clean     668 non-null    int8   
     8   Age_clean       668 non-null    float64
     9   Fare_clean      668 non-null    int32  
     10  Cabin_clean     668 non-null    float64
    dtypes: bool(1), float64(2), int32(1), int8(7)
    memory usage: 23.5 KB



```python
x_train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Sex_clean</th>
      <th>Embarked_clean</th>
      <th>Family</th>
      <th>Solo</th>
      <th>Title_clean</th>
      <th>Age_clean</th>
      <th>Fare_clean</th>
      <th>Cabin_clean</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
      <td>2.0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>68</th>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>0</td>
      <td>2</td>
      <td>7</td>
      <td>False</td>
      <td>2</td>
      <td>1.0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>253</th>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
      <td>False</td>
      <td>3</td>
      <td>2.0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>320</th>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>True</td>
      <td>3</td>
      <td>1.0</td>
      <td>0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>706</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>True</td>
      <td>4</td>
      <td>3.0</td>
      <td>0</td>
      <td>1.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.summary()
```




<table class="simpletable">
<caption>Logit Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>       <td>Survived</td>     <th>  No. Observations:  </th>  <td>   668</td>  
</tr>
<tr>
  <th>Model:</th>                 <td>Logit</td>      <th>  Df Residuals:      </th>  <td>   657</td>  
</tr>
<tr>
  <th>Method:</th>                 <td>MLE</td>       <th>  Df Model:          </th>  <td>    10</td>  
</tr>
<tr>
  <th>Date:</th>            <td>Sun, 28 Mar 2021</td> <th>  Pseudo R-squ.:     </th>  <td>0.3413</td>  
</tr>
<tr>
  <th>Time:</th>                <td>18:52:55</td>     <th>  Log-Likelihood:    </th> <td> -293.51</td> 
</tr>
<tr>
  <th>converged:</th>             <td>True</td>       <th>  LL-Null:           </th> <td> -445.58</td> 
</tr>
<tr>
  <th>Covariance Type:</th>     <td>nonrobust</td>    <th>  LLR p-value:       </th> <td>2.079e-59</td>
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>           <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Pclass</th>         <td>   -1.1096</td> <td>    0.263</td> <td>   -4.214</td> <td> 0.000</td> <td>   -1.626</td> <td>   -0.593</td>
</tr>
<tr>
  <th>SibSp</th>          <td>   -6.3773</td> <td>    0.909</td> <td>   -7.012</td> <td> 0.000</td> <td>   -8.160</td> <td>   -4.595</td>
</tr>
<tr>
  <th>Parch</th>          <td>   -5.9752</td> <td>    0.888</td> <td>   -6.728</td> <td> 0.000</td> <td>   -7.716</td> <td>   -4.235</td>
</tr>
<tr>
  <th>Sex_clean</th>      <td>   -2.6116</td> <td>    0.229</td> <td>  -11.428</td> <td> 0.000</td> <td>   -3.059</td> <td>   -2.164</td>
</tr>
<tr>
  <th>Embarked_clean</th> <td>   -0.1870</td> <td>    0.140</td> <td>   -1.338</td> <td> 0.181</td> <td>   -0.461</td> <td>    0.087</td>
</tr>
<tr>
  <th>Family</th>         <td>    5.7611</td> <td>    0.876</td> <td>    6.573</td> <td> 0.000</td> <td>    4.043</td> <td>    7.479</td>
</tr>
<tr>
  <th>Solo</th>           <td>   -0.8793</td> <td>    0.328</td> <td>   -2.679</td> <td> 0.007</td> <td>   -1.523</td> <td>   -0.236</td>
</tr>
<tr>
  <th>Title_clean</th>    <td>   -0.2110</td> <td>    0.152</td> <td>   -1.392</td> <td> 0.164</td> <td>   -0.508</td> <td>    0.086</td>
</tr>
<tr>
  <th>Age_clean</th>      <td>   -0.4346</td> <td>    0.138</td> <td>   -3.143</td> <td> 0.002</td> <td>   -0.706</td> <td>   -0.164</td>
</tr>
<tr>
  <th>Fare_clean</th>     <td>    0.0126</td> <td>    0.198</td> <td>    0.063</td> <td> 0.949</td> <td>   -0.376</td> <td>    0.401</td>
</tr>
<tr>
  <th>Cabin_clean</th>    <td>    0.2644</td> <td>    0.350</td> <td>    0.756</td> <td> 0.450</td> <td>   -0.421</td> <td>    0.950</td>
</tr>
</table>




```python
#계수들
results.params
```




    Pclass           -1.109590
    SibSp            -6.377258
    Parch            -5.975151
    Sex_clean        -2.611591
    Embarked_clean   -0.187041
    Family            5.761088
    Solo             -0.879342
    Title_clean      -0.210982
    Age_clean        -0.434641
    Fare_clean        0.012587
    Cabin_clean       0.264359
    dtype: float64




```python
##계수들의 오즈비 
np.exp(results.params)
```




    Pclass              0.329694
    SibSp               0.001700
    Parch               0.002541
    Sex_clean           0.073418
    Embarked_clean      0.829410
    Family            317.693867
    Solo                0.415056
    Title_clean         0.809789
    Age_clean           0.647497
    Fare_clean          1.012667
    Cabin_clean         1.302596
    dtype: float64



#### 오즈비 해석

* Pclass 단위가 1 증가할 때마다 생존할 오즈가 0.32배라는 의미입니다.
* solo인 사람이 solo가 아닌 사람보다 생존할 오즈가 0.41배 입니다.

#### 모형의 적합도 평가

* 모든 변수를 포함한 모형을 포화모형이라고 합니다. 이때의 가능도 값을 L1이라고 두겠습니다.
* 의미없는 설명변수를 제거하고 적합한 모형의 가능도 값을 L2라고 하겠습니다.
* 이때 Likelihood ratio statistics, LRS는 -2log(L1/L2)입니다.
* LRS이 0에 가까울수록 적합이 잘됐다고 판단할 수 있습니다.
* LRS = Deviance(이탈도) = -2loglikelihood
* 이탈도는 카이제곱 모형을 따르게 되며 자유도는 각 모형의 자유도를 뺀 값이다.
* Cox & Snell, Nagelkerke, pseudo-Rsquare는 낮을수록 좋습니다.
* Hosmer-Lemeshow's goodness-of-fit test: 모형이 적합한지를 테스트를 합니다. 표본수가 커야하고 귀무가설이 모형이 적합하다입니다. 

#### 회귀계수 검정

* Wald통계량 이용
* $W_j =  \left(  {\hat{\beta_j} \over \sqrt{\hat{Var({\hat\beta_j})}}} \right )^2$ 
* 자유도가 1인 카이제곱 통계량을 따른다.

#### References

* [https://www.slideshare.net/JeonghunYoon/04-logistic-regression](https://www.slideshare.net/JeonghunYoon/04-logistic-regression)
* [https://m.blog.naver.com/libido1014/120122772781](https://m.blog.naver.com/libido1014/120122772781)
* [https://todayisbetterthanyesterday.tistory.com/11](https://todayisbetterthanyesterday.tistory.com/11)
* 패스트캠퍼스
* 회귀분석 -박성현


```python

```
