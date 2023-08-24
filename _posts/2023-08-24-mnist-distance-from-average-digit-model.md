# MNIST Distance from Average Digit Model
This post is the main parts of a jupyter notebook (instantiated on [Kaggle](https://www.kaggle.com/)) in which I made my first submission to a Kaggle competition. The competition was to build a Digit Classifier based on the MNIST data set, and to make it as accurate as possible on the competition test data.  

I created a model that was not deep learning, or any type of machine learning for that matter. It was a simple model that:
- took the average of the pixel values for each digit class (0-9) in the training data
- and got it's predictions (of the classes of input images), by chacking which of the 'average digits' they were closest to.
## The Kaggle Notebook
```python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
```python
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```

`setup_comp` is from `fastkaggle` library, it get's the path to the data for competition. If not on kaggle it will: download it, and also it will install any of the modules passed to it as strings.


```python
comp = 'digit-recognizer'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
```


```python
from fastai.vision.all import *
```
let's check what's in the data.


```python
path.ls()
```




    (#3) [Path('../input/digit-recognizer/sample_submission.csv'),Path('../input/digit-recognizer/train.csv'),Path('../input/digit-recognizer/test.csv')]



We have a `train.csv` and a `test.csv`. `test.csv` is what we use for submission. So it looks like we will be creating our validation set, as well as the training set, from `train.csv`.

let's look at the data.


```python
df = pd.read_csv(path/'train.csv')
df
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
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>41995</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41996</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41997</th>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41998</th>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>41999</th>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>42000 rows × 785 columns</p>
</div>



It is just as described in the competition guidelines.

Lets split this into training and validation data.  
We will split by rows.  
We will use 80% for training and 20% for validation. 80% of 42,000 is 33,600 so that will be our split index.




```python
train_data = df.iloc[:33_600,:]
valid_data = df.iloc[33_600:,:]

len(train_data)/42000,len(valid_data)/42000
```




    (0.8, 0.2)



Our pixel values can be anywhere between 0 and 255. For good practice, and ease of use later, we'll normalise all these values by dividing by 255 so that they are all values between 0 and 1.

For the training data:


```python
pixel_value_columns = train_data.iloc[:,1:]
label_value_column = train_data.iloc[:,:1]

pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)
train_data = pd.concat([label_value_column, pixel_value_columns], axis=1)

train_data.describe()
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
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>33600.000000</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>...</td>
      <td>33600.000000</td>
      <td>33600.000000</td>
      <td>33600.000000</td>
      <td>33600.000000</td>
      <td>33600.000000</td>
      <td>33600.000000</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
      <td>33600.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.459881</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000801</td>
      <td>0.000454</td>
      <td>0.000255</td>
      <td>0.000086</td>
      <td>0.000037</td>
      <td>0.000007</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.885525</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.024084</td>
      <td>0.017751</td>
      <td>0.013733</td>
      <td>0.007516</td>
      <td>0.005349</td>
      <td>0.001326</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.996078</td>
      <td>0.996078</td>
      <td>0.992157</td>
      <td>0.992157</td>
      <td>0.956863</td>
      <td>0.243137</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 785 columns</p>
</div>



And for the validation data:


```python
pixel_value_columns = valid_data.iloc[:,1:]
label_value_column = valid_data.iloc[:,:1]

pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)
valid_data = pd.concat([label_value_column, pixel_value_columns], axis=1)

valid_data.describe()
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
      <th>label</th>
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>8400.000000</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>...</td>
      <td>8400.000000</td>
      <td>8400.000000</td>
      <td>8400.000000</td>
      <td>8400.000000</td>
      <td>8400.000000</td>
      <td>8400.000000</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
      <td>8400.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.443690</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.001098</td>
      <td>0.000482</td>
      <td>0.000137</td>
      <td>0.000050</td>
      <td>0.000190</td>
      <td>0.000027</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.896668</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.027281</td>
      <td>0.019766</td>
      <td>0.008370</td>
      <td>0.003489</td>
      <td>0.012709</td>
      <td>0.002482</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.992157</td>
      <td>0.996078</td>
      <td>0.627451</td>
      <td>0.309804</td>
      <td>0.996078</td>
      <td>0.227451</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 785 columns</p>
</div>



## Baseline model: distance from mean
We'll create a baseline model. For this we will take the average value of each pixel, across all the images (in vector form in our data) for each label (each digit from 0-9). This average will act as our 'ideal' version of the corresponding digit. We can use these averages and take the one that's 'closest' to an input to be the correct label for that input.


```python
all_training_zeros = train_data.loc[df['label'] == 0]
all_training_ones = train_data.loc[df['label'] == 1]
all_training_twos = train_data.loc[df['label'] == 2]
all_training_threes = train_data.loc[df['label'] == 3]
all_training_fours = train_data.loc[df['label'] == 4]
all_training_fives = train_data.loc[df['label'] == 5]
all_training_sixes = train_data.loc[df['label'] == 6]
all_training_sevens = train_data.loc[df['label'] == 7]
all_training_eights = train_data.loc[df['label'] == 8]
all_training_nines = train_data.loc[df['label'] == 9]

### the below values are excluding the labels of each of the respective items
### (the mean calculated on pixel values only)
average_zero = all_training_zeros.iloc[:,1:].mean(0)
average_one = all_training_ones.iloc[:,1:].mean(0)
average_two = all_training_twos.iloc[:,1:].mean(0)
average_three = all_training_threes.iloc[:,1:].mean(0)
average_four = all_training_fours.iloc[:,1:].mean(0)
average_five = all_training_fives.iloc[:,1:].mean(0)
average_six = all_training_sixes.iloc[:,1:].mean(0)
average_seven = all_training_sevens.iloc[:,1:].mean(0)
average_eight = all_training_eights.iloc[:,1:].mean(0)
average_nine = all_training_nines.iloc[:,1:].mean(0)

averages = [average_zero, average_one, average_two, average_three, average_four, average_five, average_six, average_seven, average_eight, average_nine]
```

we can check our distances using root mean squared error. let's see how that works on a few random items.


```python
### excluding the label value from our calculation
first_zero = all_training_zeros.iloc[0,1:]

((first_zero-average_zero)**2).mean()**0.5
```




    0.2529256872506997



We can see that this is very close (a small difference), as expected. Now let's see how far the same zero item is from the average four.


```python
((first_zero-average_four)**2).mean()**0.5
```




    0.387471198659217



A much further distance, also as expected.

So now, for each item, we can check it's distance from each of the average items and then take our prediction to be whichever distance is the shortest.

let's define a rmse function to use on our dataframes.


```python
def df_rmse(a,b):
    return ((a-b)**2).mean()**0.5
```

Letes create a list of the labels.


```python
labels = [0,1,2,3,4,5,6,7,8,9]

```

And now the function (**our model**) that will return the label with the lowest distance.


```python
def get_label_prediction(x):
    distances = []
    for average in averages:
        distances.append(df_rmse(x, average))
    return labels[distances.index(min(distances))]
```


```python
get_label_prediction(first_zero)
```




    0



let's give this a run against our validation set and see what proportion of the predictions made are actually correct.


```python
def is_correct(pred,actual):
    return pred == actual
```

Let's produce a list of predictions against actuals.


```python
correct_or_not_list = [is_correct(get_label_prediction(row[1][1:]), row[1][0]) for row in valid_data.iterrows()]

correct_or_not_list[:10]
```




    [True, True, True, True, True, True, True, True, False, False]



We can workout our accuracy on this validation data by:  
first converting all `True` booleans to `1` and all `False` booleans to `0`,  
and then taking the mean of all values to see what proportion of our predictions were correct.


```python
def get_accuracy(data):
    correct_or_not_list = [is_correct(get_label_prediction(row[1][1:]), row[1][0]) for row in data.iterrows()]
    boolean_number_list = [int(boolean) for boolean in correct_or_not_list]
    
    return np.array(boolean_number_list).mean()
```


```python
get_accuracy(valid_data)
```




    0.8126190476190476



81.3%. I deem this as a good bassline. We can run this model on the test data provided and submit it to the kaggle competition with an aim to beat it in future iterations.

Let's first take a look at the test data to see if it is the same format as our training data.


```python
df = pd.read_csv(path/'test.csv')
df.describe()
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
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>...</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.164607</td>
      <td>0.073214</td>
      <td>0.028036</td>
      <td>0.011250</td>
      <td>0.006536</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>5.473293</td>
      <td>3.616811</td>
      <td>1.813602</td>
      <td>1.205211</td>
      <td>0.807475</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>253.000000</td>
      <td>254.000000</td>
      <td>193.000000</td>
      <td>187.000000</td>
      <td>119.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 784 columns</p>
</div>



Let's also normalise the pixel values, like we did before.


```python
pixel_value_columns = df.iloc[:,1:]
label_value_column = df.iloc[:,:1]

pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)
test_data = pd.concat([label_value_column, pixel_value_columns], axis=1)

test_data.describe()
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
      <th>pixel0</th>
      <th>pixel1</th>
      <th>pixel2</th>
      <th>pixel3</th>
      <th>pixel4</th>
      <th>pixel5</th>
      <th>pixel6</th>
      <th>pixel7</th>
      <th>pixel8</th>
      <th>pixel9</th>
      <th>...</th>
      <th>pixel774</th>
      <th>pixel775</th>
      <th>pixel776</th>
      <th>pixel777</th>
      <th>pixel778</th>
      <th>pixel779</th>
      <th>pixel780</th>
      <th>pixel781</th>
      <th>pixel782</th>
      <th>pixel783</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>...</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.000000</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
      <td>28000.0</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000646</td>
      <td>0.000287</td>
      <td>0.000110</td>
      <td>0.000044</td>
      <td>0.000026</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.021464</td>
      <td>0.014184</td>
      <td>0.007112</td>
      <td>0.004726</td>
      <td>0.003167</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.992157</td>
      <td>0.996078</td>
      <td>0.756863</td>
      <td>0.733333</td>
      <td>0.466667</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 784 columns</p>
</div>



Now let's create our predictions for each of the test images.


```python
predictions = [get_label_prediction(row[1]) for row in test_data.iterrows()]
```


```python
predictions[:10]
```




    [2, 0, 9, 4, 3, 7, 0, 3, 0, 3]



Let's take a looks at how our submission of these predictions should look.


```python
ss = pd.read_csv(path/'sample_submission.csv')

ss
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
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27996</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27997</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27998</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27999</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>28000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>



So let's utilise this sample submission by keeping the id's and inserting our predictions into the "Label" column.


```python
results = pd.Series(predictions, name="Label")
ss['Label'] = results

ss
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
      <th>ImageId</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27995</th>
      <td>27996</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27996</th>
      <td>27997</td>
      <td>7</td>
    </tr>
    <tr>
      <th>27997</th>
      <td>27998</td>
      <td>3</td>
    </tr>
    <tr>
      <th>27998</th>
      <td>27999</td>
      <td>9</td>
    </tr>
    <tr>
      <th>27999</th>
      <td>28000</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>28000 rows × 2 columns</p>
</div>



Now let's create a csv to submit.


```python
ss.to_csv('subm.csv', index=False)
!head subm.csv
```

    ImageId,Label
    1,2
    2,0
    3,9
    4,4
    5,3
    6,7
    7,0
    8,3
    9,0


Looks good! now we can submit this to kaggle.  

We can do it straight from this note book if we are running it on Kaggle, otherwise we can use the API  
In this case I did it directly using the kaggle notebook UI and selecting my 'subm.csv' file from the output folder there (see these [guidelines](https://www.kaggle.com/code/ryanholbrook/create-your-first-submission?scriptVersionId=46745751&cellId=49))

After submitting the results, a score of 0.80750 was given. Meaning it was in the top 80th percintile of scores! (the top being 1.00000). Not a bad result at all for such a simple model. Next thing to try will be a simple neural net...
