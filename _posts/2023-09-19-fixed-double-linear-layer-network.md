#FIXED Double Linear Layer Network
## Introduction
In the [previous notebook](https://gurtaj1.github.io/blog/2023/09/17/fixed-linear-model-with-mnist-data.html), I applied some core concepts I had learned in creating and training models to my, then failing, linear equation model on the MNIST data set.  
The optimisation I had made were to do with how many activations I was producing, per image, and what the shape of my data was whenver it had needed processing.

The updated linear equation model was very successful and I am now going to apply the same optimisations to my 2 linear layer model.


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```


```python
# install fastkaggle if not available
try: import fastkaggle
except ModuleNotFoundError:
    !pip install -Uq fastkaggle

from fastkaggle import *
```


```python
comp = 'digit-recognizer'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /Users/gaz/.kaggle/kaggle.json'



```python
from fastai.vision.all import *
```


```python
path.ls()
```




    (#3) [Path('digit-recognizer/test.csv'),Path('digit-recognizer/train.csv'),Path('digit-recognizer/sample_submission.csv')]




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




```python
train_data_split = df.iloc[:33_600,:]
valid_data_split = df.iloc[33_600:,:]

len(train_data_split)/42000,len(valid_data_split)/42000
```




    (0.8, 0.2)




```python
pixel_value_columns = train_data_split.iloc[:,1:]
label_value_column = train_data_split.iloc[:,:1]

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




```python
pixel_value_columns_tensor = torch.tensor(train_data.iloc[:,1:].values).float()
label_value_column_tensor = torch.tensor(train_data.iloc[:,:1].values).float()

train_ds = list(zip(pixel_value_columns_tensor,label_value_column_tensor))
```

We'll make this a function, so that we can do the same again for our validation data.


```python
train_dl = DataLoader(train_ds, batch_size=256)
train_xb,train_yb = first(train_dl)

train_xb.shape,train_xb.shape
```




    (torch.Size([256, 784]), torch.Size([256, 784]))




```python
def dataset_from_dataframe(dframe):
    pixel_value_columns = dframe.iloc[:,1:]
    label_value_column = dframe.iloc[:,:1]

    pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)

    pixel_value_columns_tensor = torch.tensor(train_data.iloc[:,1:].values).float()
    label_value_column_tensor = torch.tensor(train_data.iloc[:,:1].values).float()

    return list(zip(pixel_value_columns_tensor, label_value_column_tensor))
```


```python
valid_ds = dataset_from_dataframe(valid_data_split)

valid_dl = DataLoader(valid_ds, batch_size=256)
```

To ease my mind and help spot places where I could be making errors, i'll make a function that can visually show a particular input (digit image) to me.


```python
def show_image(item):
    item = item.view(28,28) * 255
    plt.gray()
    plt.imshow(item, interpolation='nearest')
    plt.show()
```

Now, for my sanity, i'll test an images in `train_xb`.


```python
show_image(train_xb[0])
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_17_0.png)
    



```python
def init_params(size): return (torch.rand(size) - 0.5).requires_grad_()
```


```python
# 30 lots of 784 length weight arrays, therefore 30 outputs
w1 = init_params((30,784))
b1 = init_params((30, 1))
# 10 lots of 30 length weight arrays, therefore 10 outputs
w2 = init_params((10,30))
b2 = init_params((10, 1))
```


```python
def simple_nn(batch):
    res = w1@batch.T + b1
    res = F.relu(res)
    res = w2@res + b2
    return F.softmax(res, dim=0)
```


```python
first_batch_predictions = simple_nn(train_xb)

first_batch_predictions.shape
```




    torch.Size([10, 256])



Let's see what we get from the first column (results of the first image).


```python
first_batch_predictions[:,0]
```




    tensor([1.0476e-03, 5.4595e-03, 9.7319e-02, 2.4645e-02, 4.4612e-03, 9.0264e-02,
            2.5988e-04, 7.6168e-01, 1.2942e-02, 1.9236e-03],
           grad_fn=<SelectBackward0>)



Just to confirm that we did the softmax call across the right dimenstion, let's ensure all these 10 values now add up to `1`.


```python
sum(first_batch_predictions[:,0])
```




    tensor(1., grad_fn=<AddBackward0>)



## Loss Function


```python
number_of_classes = 10

def one_hot(yb):
    batch_size = len(yb)
    one_hot_yb = torch.zeros(batch_size, number_of_classes)
    x_coordinates_array = torch.arange(len(one_hot_yb))
    # used `.squeeze()` becasue yb originally has the size (batch_size, 1) and we just want a size of (batch_size). ([1, 2, 3, ...] instead of [[1], [2], [3], ...])
    # used `.long()` because: "tensors used as indices must be long, int, byte or bool tensors"
    y_coordinates_array = yb.squeeze().long()
    # set to `1.` rather than `1` because: Index put requires the source and destination dtypes match, got Float for the destination and Long for the source.
    one_hot_yb[x_coordinates_array, y_coordinates_array] = torch.tensor(1.)
    
    return one_hot_yb.T
```


```python
def rmse(a, b):
    b = one_hot(b)
    mse = nn.MSELoss()
    loss = torch.sqrt(mse(a, b))
    
    return loss
```


```python
one_hot(train_yb),one_hot(train_yb).shape,one_hot(train_yb)[:,0],train_yb[0]
```




    (tensor([[0., 1., 0.,  ..., 0., 0., 0.],
             [1., 0., 1.,  ..., 0., 0., 1.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             ...,
             [0., 0., 0.,  ..., 0., 1., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.],
             [0., 0., 0.,  ..., 0., 0., 0.]]),
     torch.Size([10, 256]),
     tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]),
     tensor([1.]))



Let's run it on our test batch predictions from above.


```python
rmse(first_batch_predictions, train_yb)
```




    tensor(0.3866, grad_fn=<SqrtBackward0>)



## Trainability


```python
def calc_grad(batch_inputs, batch_labels, batch_model):
    batch_preds = batch_model(batch_inputs)
    loss = rmse(batch_preds, batch_labels)
    loss.backward()
```


```python
def train_epoch(dl, batch_model, params, lr):
    for xb,yb in dl:
        calc_grad(xb, yb, batch_model)
        for p in params:
            pdata1 = p.data
            p.data -= p.grad*lr
            pdata2 = p.data
            p.grad.zero_()
```

## Validation and Metric


```python
def get_predicted_label(pred):
    #returns index of highest value in tensor, which convenietnly also is directly the the digit/label that it corresponds to
    return torch.argmax(pred)
```


```python
get_predicted_label(torch.tensor([0,4,3,2,6,1]))
```




    tensor(4)



Let's test this on some predictions from `first_batch_predictions` to ensure that we are getting sensible values (values from 0-9)


```python
get_predicted_label(first_batch_predictions[:,0]),get_predicted_label(first_batch_predictions[:,1]),get_predicted_label(first_batch_predictions[:,3]),get_predicted_label(first_batch_predictions[:,5]),get_predicted_label(first_batch_predictions[:,33])
```




    (tensor(7), tensor(3), tensor(3), tensor(3), tensor(3))



### Accuracy


```python
def batch_accuracy(preds, yb):
    #remember each column in our preds is an indivudual prediction, so we transpose preds in order to iterate through each precition in our list comprehension below
    preds = torch.tensor([get_predicted_label(pred) for pred in preds.T])
    # is_correct is a tensor of True and False values
    is_correct = preds==yb.squeeze()
    # now we turn all True values into 1 and all False values into 0, then return the mean of those values
    return is_correct.float().mean()
```


```python
batch_accuracy(simple_nn(train_xb[:100]),train_yb[:100])
```




    tensor(0.0800)




```python
def validate_epoch(dl, batch_model):
    accuracies = [batch_accuracy(batch_model(xb),yb) for xb,yb in dl]
    # turn list of tensors into one single tensor of stacked values, so that we can then calculate the mean across all those values
    stacked_tensor = torch.stack(accuracies)
    mean_tensor = stacked_tensor.mean()
    # round method only works on value within tensor so we use item() to get it (and then round to four decimal places)
    return round(mean_tensor.item(), 4)
```


```python
validate_epoch(valid_dl, simple_nn)
```




    0.0644



## Train for Number of Epochs


```python
lr = 0.01
params = w1,b1,w2,b2
```


```python
train_epoch(train_dl, simple_nn, params, lr)
validate_epoch(valid_dl, simple_nn)
```




    0.082



Now let's attempt at training our model over 500 more epochs and see if it improves.


```python
for i in range(500):
    train_epoch(train_dl, simple_nn, params, lr)
    # run validate_epoch on every 50th iteration
    if i % 50 == 0:
        print(validate_epoch(valid_dl, simple_nn), ' ')
```

    0.1113  
    0.5865  
    0.7313  
    0.7669  
    0.8314  
    0.8682  
    0.8825  
    0.8929  
    0.8995  
    0.9053  


A ~91% accuracy on our validation data. For the single linear function model in the [previous notebook](https://gurtaj1.github.io/blog/2023/09/17/fixed-linear-model-with-mnist-data.html) we got an ~87% accuracy in the same number of epochs. So we are getting a higher accuracy with this two layer network but one key thing to notice is that huge jump in accuracy from the first printout to the second, this is a drastic improvement in trainability on the single layer linear model in the previous notebook.  

I'll now run this on the test data and submit it to the kaggle competition in order to see if it's any good...

### Submission


```python
test_df = pd.read_csv(path/'test.csv')

test_df.describe()
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



Now we format it into what our model expects. Since we don't have labels for this data we'll make a batch of inputs only (the batch will be the whole of the test data)


```python
test_tensor = torch.tensor(test_df.values)/255

test_tensor.shape
```




    torch.Size([28000, 784])



Let's take a look at the first image, just for sanity.


```python
show_image(test_tensor[0])
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_56_0.png)
    


Now let's create a function that produces a meaningful output (digit prediction) for each image, using our model.


```python
def predict(batch):
    preds = simple_nn(batch)
    # convert tensor to numpy value
    preds = [get_predicted_label(pred).numpy() for pred in preds.T]
    
    return preds
```


```python
preds = predict(test_tensor)

preds[:5]
```




    [array(2), array(0), array(9), array(9), array(2)]




```python
pred_labels_series = pd.Series(preds, name="Label")

pred_labels_series
```




    0        2
    1        0
    2        9
    3        9
    4        2
            ..
    27995    9
    27996    7
    27997    3
    27998    9
    27999    2
    Name: Label, Length: 28000, dtype: object




```python
sample_submission = pd.read_csv(path/'sample_submission.csv')
sample_submission
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




```python
len(test_df)
```




    28000




```python
sample_submission['Label'] = pred_labels_series

sample_submission
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
      <td>9</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2</td>
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




```python
# this outputs the actual file
sample_submission.to_csv('subm.csv', index=False)
#this shows the head (first few lines)
!head subm.csv
```

    ImageId,Label
    1,2
    2,0
    3,9
    4,9
    5,2
    6,7
    7,0
    8,3
    9,0



```python
!kaggle competitions list
```


```python
!kaggle competitions files digit-recognizer 
```


```python
!kaggle competitions submit -c digit-recognizer  -f ./subm.csv -m "going from single linear function model to two-linear-layer model"

```

This received a score of 0.90142, an improvement on the previous model yet again! 

## Conclusion
Using just the things learnt in the [previous notebook](https://gurtaj1.github.io/blog/2023/09/17/fixed-linear-model-with-mnist-data.html) we have managed to get our, previously non-working, 2 linear layer model to work, and with better results than the single layer (as one would expect).  

The next step will be to try out our 2 linear layer model that utilises PyTorch's `nn` modules and see if any of the things we have learnt thusfar will help us to get that working too. This will be done in a separate notebook.
