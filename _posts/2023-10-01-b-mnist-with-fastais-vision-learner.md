# MNIST with fastai's vision_learner
## Introduction
Previously I have attempted several different models for tackling this MNIST digit recognition problem (see the Kaggle competition [here](https://www.kaggle.com/competitions/digit-recognizer/overview)). The best model I had so far was the most recent which used PyTorch's `nn.Conv2d` to manually construct a CNN architecture.  

In this notebook I will attempt to create a model using all of the tools available from the fastai library, `vision_learner` being one of those. I will also use a pretrained model `resnet18`. There are certainly better models out there, at the time of writing this, but I know this one to be quickly trainable as well as successful on recognising images that are far more complex then black and white hand written digits.

Let's begin by setting thing's up in this notebook..


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

`setup_comp` is from `fastkaggle` library, it get's the path to the data for competition. If not on kaggle it will: download it, and also it will install any of the modules passed to it as strings.


```python
comp = 'digit-recognizer'

path = setup_comp(comp, install='fastai "timm>=0.6.2.dev0"')
```


```python
from fastai.vision.all import *
```

let's check whats in the data.


```python
path.ls()
```




    (#3) [Path('digit-recognizer/test.csv'),Path('digit-recognizer/train.csv'),Path('digit-recognizer/sample_submission.csv')]



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

## Data Preperation
Lets split this data into training and validation data.  
We will split by rows.  
We will use 80% for training and 20% for validation. 80% of 42,000 is 33,600 so that will be our split index.


```python
train_data_split = df.iloc[:33_600,:]
valid_data_split = df.iloc[33_600:,:]

len(train_data_split)/42000,len(valid_data_split)/42000
```




    (0.8, 0.2)



Our pixel values can be anywhere between 0 and 255. For good practice, and ease of use later, we'll normalise all these values by dividing by 255 so that they are all values between 0 and 1.


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



Let's further process our data so that it is ready for this new type of model we are using. Note that:
- we will use 28*28 pixel matrices for our images rather than 783 pixel vectors
  - because convolutions are done on matrices
- we also add a dimension of 1 as the first dimension (view(1,28,28)) because Conv2d takes in 'channels' for each image
  - 3d images would have 2 channels, 1 for each colour (RGB).
  - since we are only dealing with black and white images (one colour) we will deal with only one channel


```python
pixel_value_columns = train_data_split.iloc[:,1:]
label_value_column = train_data_split.iloc[:,:1]

pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)

pixel_value_columns_tensor = torch.tensor(train_data.iloc[:,1:].values).float()
# here we change from image vectors to image matrices (via `.view`) and put it in our three channels (via `.expand`) ( we want three channels since resnet18 expects 3 channel images (RGB))
pixel_value_matrices_tensor = [row.view(28,28).expand(3,28,28) for row in pixel_value_columns_tensor]

label_value_column_tensor = torch.tensor(train_data.iloc[:,:1].values).float()

# F.cross_entropy requires that the labels are tensor of scalar. label values cannot be `FloatTensor` if they are classes (discrete values), must be cast to `LongTensor` (`LongTensor` is synonymous with integer)
train_ds = list(zip(pixel_value_matrices_tensor, label_value_column_tensor.squeeze().type(torch.LongTensor)))

train_dl = DataLoader(train_ds, batch_size=256)
```

Lets do the same for the validation data.


```python
pixel_value_columns = valid_data_split.iloc[:,1:]
label_value_column = valid_data_split.iloc[:,:1]

pixel_value_columns = pixel_value_columns.apply(lambda x: x/255)

pixel_value_columns_tensor = torch.tensor(train_data.iloc[:,1:].values).float()
    # NOTE how in .expand we can use `-1` instead of `28` to say 'keep this dimension the same size as it is originally'
pixel_value_matrices_tensor = [row.view(28,28).expand(3,-1,-1) for row in pixel_value_columns_tensor]

label_value_column_tensor = torch.tensor(train_data.iloc[:,:1].values).float()

valid_ds = list(zip(pixel_value_matrices_tensor, label_value_column_tensor.squeeze().type(torch.LongTensor)))

valid_dl = DataLoader(valid_ds, batch_size=256)
```


```python
dls = DataLoaders(train_dl,valid_dl)
```

Let's see if we now have the format we need.


```python
valid_xb,valid_yb = first(valid_dl)
valid_xb.shape,valid_yb.shape
```




    (torch.Size([256, 3, 28, 28]), torch.Size([256]))



So we have:
- tuples of (x, y),
- and for x we have
  - batches of 256,
  - images of 3 channels (even though it is a black and white image we tripled the channels as resnet18 works with RGB/3-channel images),
  - of size 28 by 28 pixels.
- and for y we have
  - batches of 256 of,
  - single values (the ground truth/digit label).

NOTE: that the way in which we trippled our channels was very inneficient. First we converted them to numpy arrays inorder to change their combination back to a tensor in the end. There must be a more efficient way of doing this that I am yet to learn.

To ease my mind and help spot places where I could be making errors, i'll make a function that can visually show a particular input (digit image) to me.


```python
def show_image(item):
    plt.gray()
    plt.imshow(item, interpolation='nearest')
    plt.show()
```


```python
# we access the first channel of the first image of our first batch of validation data
show_image(valid_xb[0][0])
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_27_0.png)
    


## Model Creation and Training
Now let's see what we can do with this data using `resnet18` architecture CNN model with fastai's `vision_learner`.  

Note that we will also use `fine_tune` here rather than `fit` or `fit_one_cycle` since we don't want to retrain the whole model, just the head (last layers) in order to utilise its pre-existing trained wieghts but then specify their use towards our specific goal.


```python
learn = vision_learner(dls, resnet18, metrics=accuracy, n_out=10, normalize=False, loss_func=F.cross_entropy)
```


```python
learn.fine_tune(1)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.278895</td>
      <td>0.619535</td>
      <td>0.802321</td>
      <td>06:30</td>
    </tr>
  </tbody>
</table>




<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>




<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.268740</td>
      <td>0.103634</td>
      <td>0.968423</td>
      <td>12:43</td>
    </tr>
  </tbody>
</table>


0.968423 sounds like a very good accuracy indeed. But, going off previous attempts, this is not reason alone to get excited and start expecting a great result on the leaderbaord. The only way to gauge that will be to make a submission, so let's do that!

## Submitting Results
First let's pre process the test data like we did with our training data, since this is how our model expects it.


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



Like our traning data, the test data has values for each pixel in each image. Unlike our test data it does not include any labels for any of the images. Therefore our data preperation will take this difference into account. We will create a `DataLoader` in order to feed our data into the model *but* we since DataLoaders require labels and we don't have any, we will just create some dummy `0` value labels instead.


```python
def dataloader_from_test_dataframe(dframe):
    pixel_value_columns = dframe.apply(lambda x: x/255)

    pixel_value_columns_tensor = torch.tensor(pixel_value_columns.values).float()
    pixel_value_matrices_tensor = [row.view(28,28).expand(3,-1,-1) for row in pixel_value_columns_tensor]

    dummy_label_value_column_tensor = torch.zeros(len(pixel_value_columns_tensor)).float()

    ds = list(zip(pixel_value_matrices_tensor, dummy_label_value_column_tensor.squeeze().type(torch.LongTensor)))

    return DataLoader(ds, batch_size=len(pixel_value_columns_tensor))
```


```python
test_dl = dataloader_from_test_dataframe(test_df)
```


```python
test_xb,test_yb = first(test_dl)
test_xb.shape,test_yb.shape
```




    (torch.Size([28000, 3, 28, 28]), torch.Size([28000]))




```python
show_image(test_xb[0][0]),show_image(test_xb[1][0])
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_38_0.png)
    



    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_38_1.png)
    





    (None, None)



Notice how we didn't include the final step of creating a dataloader here, like we did for our training and validation data earlier. That's because, for some reason, we now need to use the `test_dl` method on the `dls` object of our `learn`. I think it is something to do with how I have done this all manaully in a way that we normally wouldn't in real practice. (for the sake of learning). For example there is the fact that I earlier used `DataLoader` on the training and validation data instead of `ImageDataLoaders` or `DataBlock` with the appropriate input and output types declared (`ImageBlock` and `CategoryBlock` respectively).  

The methods used below were acquired from [this comment](https://forums.fast.ai/t/not-able-to-export-learner-failing-with-attributeerror-list-object-has-no-attribute-new-empty/81803?u=gurtaj) in the fastai forums.


```python
preds = learn.get_preds(dl=test_dl)
```



<style>
    /* Turns off some styling */
    progress {
        /* gets rid of default border in Firefox and Opera. */
        border: none;
        /* Needs to be in here for Safari polyfill so background images work as expected. */
        background-size: auto;
    }
    progress:not([value]), progress:not([value])::-webkit-progress-bar {
        background: repeating-linear-gradient(45deg, #7e7e7e, #7e7e7e 10px, #5c5c5c 10px, #5c5c5c 20px);
    }
    .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
        background: #F44336;
    }
</style>








```python
preds_x,preds_y = preds
preds_x.shape,preds_y.shape
```




    (torch.Size([28000, 10]), torch.Size([28000]))



We are getting tuples as our output. The first item in each tuple is the activation that corresponds to each possible class that we are classifying our inputs by (each possible digit). The second item in the tuple is just always 0 as this is the dummy label we added to our DataLoader. Let's confirm by looking at a few.


```python
preds_x[0],preds_y[0],preds_x[1],preds_y[1],preds_x[2],preds_y[2],preds_x[3],preds_y[3],preds_x[4],preds_y[4]
```




    (tensor([-2.1849, -0.6895, 13.6192,  4.5025, -4.8089,  3.7261, -5.1013, -3.5475,
             -0.4706, -3.9076]),
     tensor(0),
     tensor([ 7.8694, -2.0303, -3.1039, -1.5830, -1.8573, -2.4699,  0.6730, -2.8973,
             -0.3747,  0.0887]),
     tensor(0),
     tensor([-2.2344, -3.3648, -0.4304,  0.4682, -3.0089,  1.7690, -1.3859, -2.9892,
              5.0062,  2.8473]),
     tensor(0),
     tensor([ 0.2836, -2.3423,  1.0665,  1.1372, -2.7025,  2.5808,  2.2428, -2.8127,
              1.8188, -1.4930]),
     tensor(0),
     tensor([-4.4768, -2.5752,  3.9028, 12.8070, -2.8616,  2.3295, -2.0011, -2.2258,
             -2.3868, -3.6948]),
     tensor(0))



The negative activation values are throwing me off as I did not expect it, non the less I will go ahead and take the index of the highes activation value in each prediction, as our predicted class.


```python
def get_predicted_label(pred):
    #returns index of highest value in tensor, which convenietnly also is directly the the digit/label that it corresponds to
    return torch.argmax(pred)
```


```python
# .numpy() turns the scalar tensors into normal scalar variable in python
predictions_list = [get_predicted_label(pred).numpy() for pred in preds_x]

predictions_list[:5]
```




    [array(2), array(0), array(8), array(5), array(3)]




```python
pred_labels = pd.Series(predictions_list, name="Label")

pred_labels
```




    0        2
    1        0
    2        8
    3        5
    4        3
            ..
    27995    9
    27996    7
    27997    3
    27998    9
    27999    2
    Name: Label, Length: 28000, dtype: object




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




```python
ss['Label'] = pred_labels

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
      <td>8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>5</td>
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



Looks good! now we can submit this to kaggle.  

We can do it straight from this note book if we are running it on Kaggle, otherwise we can use the API  
In this case I did it directly using the kaggle notebook UI and selecting my 'subm.csv' file from the output folder there (see these [guidelines](https://www.kaggle.com/code/ryanholbrook/create-your-first-submission?scriptVersionId=46745751&cellId=49))


```python
# this outputs the actual file
ss.to_csv('subm.csv', index=False)
#this shows the head (first few lines)
!head subm.csv
```

    ImageId,Label
    1,2
    2,0
    3,8
    4,5
    5,3
    6,7
    7,0
    8,3
    9,0



```python
!kaggle competitions submit -c digit-recognizer  -f ./subm.csv -m "vision learner"
```

This got a score of 0.9471 which is just the tiniest bit below the score we got for the CNN we made by manually constructing our architecture using PyTorch module `nn.Conv2d` (see [the previous notebook](https://gurtaj1.github.io/blog/2023/10/01/convolutional-network-using-nn.conv2d.html))

## Conclusion
fastai's `vision_learner` and `fine_tune`, allowed us the benefit of using a pre-made, tried and proven, architecture, along with its pre-learned weights.  

With singificantly less training time than that which was required for the [previous nn.Conv2d model](https://gurtaj1.github.io/blog/2023/10/01/convolutional-network-using-nn.conv2d.html), and still receiving almost the same score, it can be seen that using `vision_learner` has improved our efficiency by a great deal.
