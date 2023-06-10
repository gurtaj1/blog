# An Attempt at an Image Finder (Object Detection)
### The Goal
I had an image that i was using as my profile image on certain social media. This image was actually a small crop from an original full image. Since using this crop I had never thought to make a note of where the full image is. I did have a few old folders of images that mainly needed deleting (whatsapp media) but i know the full image was sent to me and may indeed be somewhere in one of these folders. 

That's when it occured to me that rather than go through the 1000's of images one by one, I could surely just develop an image recognition model that would be able to detect any image that includes the crop image that I was using as a profile picture.


### Proposed Method 1
I had just finished going through chapter four of ['Deep Learning for Coders with fastai and Pytorch' by Jeremy Howard and Sylvian Gugger](https://github.com/fastai/fastbook/blob/master/04_mnist_basics.ipynb). In this chapter it is stated that a single nonlinearity with, two linear layers, is enough to approximate any function. With this in mind I thought why not give such a model the problem I am trying to solve with finding my full image.

### Proposed Method 2
As you can see in my [last blog post](https://gurtaj1.github.io/blog/2021/09/04/first_post_icon_classifier.html) I created an image classifier model, with very minimal effort, using fastai and PyTorch. When doing this I saw that there are many, pre-trained, models readily available for me to use and it wouldn't take much effort in terms of writing up the code at all. The same pre-trained model could be chosen, and then further trained on my specific data in order to fulfil my specific purpose.

Perhaps the hardest part in both methods, is preparing the data for the model to be trained with. I decided what I would do is get a set of my own photos. The original copies would all be **negatives**. But then I will make a copy of them all and all the copies will have the crop image that I am trying to detect, added to them. these edited copies will be the **positives**.

#### TL;DR
Both methods produced a model that was not able to fulfill the purpose I wanted to use it for. I concluded that I would need to study further in the fastai course and read further in the book.

---

I began by installing and importing the necessary packages.


```python
#hide
# !mamba install -c fastchan fastai
# !mamba install -c fastchan nbdev
```


```python
from fastai.vision.all import *
import pathlib
```

## Data Preparation

Then I proceeded to import my image data and get it into a format that is ready to be utilised easily by my models. The format used is actually one that fastai uses (see `DataLoaders` further below).

>Note: in the code block below, `get_image_files(path/'training/negative')` does the same thing that `(path/'training/negative').ls()` also does.


```python
path = pathlib.Path().resolve()

training_negatives = get_image_files(path/'training/negative')
training_positives = get_image_files(path/'training/positive')
```

Since all images are of different sizes, this is not good for a model. We must normalise them to make them all uniform. I opted to resize them all to 128 pixels by 128 pixels.


```python
im = Image.open(training_positives[2]).resize((128,128))
```

The shape of `im_tens` below shows that each image is 3 lots of a 128 by 128 matrix of pixels. That's one matrix for each color R, G, and B


```python
im_tens = tensor(im)
im_tens.shape
```




    torch.Size([128, 128, 3])



Let's plot, for each of the three colours, the shades for each pixel value, for a subsection of the image.


```python
df_red = pd.DataFrame(im_tens[87:112,99:119,0])
df_red.style.set_properties(**{'font-size':'6pt'}).background_gradient('Reds')
```




<style type="text/css">
#T_3398e_row0_col0, #T_3398e_row0_col4, #T_3398e_row0_col5, #T_3398e_row0_col6, #T_3398e_row0_col7, #T_3398e_row0_col8, #T_3398e_row1_col2, #T_3398e_row1_col3, #T_3398e_row2_col1, #T_3398e_row4_col11, #T_3398e_row4_col12, #T_3398e_row7_col10, #T_3398e_row8_col9, #T_3398e_row10_col13, #T_3398e_row10_col14, #T_3398e_row11_col15, #T_3398e_row12_col16, #T_3398e_row12_col18, #T_3398e_row12_col19, #T_3398e_row13_col17 {
  font-size: 6pt;
  background-color: #67000d;
  color: #f1f1f1;
}
#T_3398e_row0_col1 {
  font-size: 6pt;
  background-color: #8e0912;
  color: #f1f1f1;
}
#T_3398e_row0_col2, #T_3398e_row3_col1 {
  font-size: 6pt;
  background-color: #9c0d14;
  color: #f1f1f1;
}
#T_3398e_row0_col3, #T_3398e_row2_col2 {
  font-size: 6pt;
  background-color: #6f020e;
  color: #f1f1f1;
}
#T_3398e_row0_col9, #T_3398e_row4_col0, #T_3398e_row14_col19, #T_3398e_row19_col17 {
  font-size: 6pt;
  background-color: #fb7050;
  color: #f1f1f1;
}
#T_3398e_row0_col10 {
  font-size: 6pt;
  background-color: #fff0e9;
  color: #000000;
}
#T_3398e_row0_col11, #T_3398e_row2_col8, #T_3398e_row2_col12, #T_3398e_row2_col13, #T_3398e_row3_col10, #T_3398e_row3_col14, #T_3398e_row3_col15, #T_3398e_row3_col18, #T_3398e_row3_col19, #T_3398e_row4_col6, #T_3398e_row4_col9, #T_3398e_row5_col5, #T_3398e_row5_col8, #T_3398e_row6_col4, #T_3398e_row6_col7, #T_3398e_row7_col3, #T_3398e_row11_col0, #T_3398e_row20_col1, #T_3398e_row21_col0, #T_3398e_row21_col3, #T_3398e_row23_col2, #T_3398e_row23_col16, #T_3398e_row23_col17 {
  font-size: 6pt;
  background-color: #fff5f0;
  color: #000000;
}
#T_3398e_row0_col12, #T_3398e_row1_col8, #T_3398e_row1_col13, #T_3398e_row4_col7, #T_3398e_row9_col12, #T_3398e_row20_col4, #T_3398e_row21_col14, #T_3398e_row24_col8 {
  font-size: 6pt;
  background-color: #fcbba1;
  color: #000000;
}
#T_3398e_row0_col13, #T_3398e_row6_col19, #T_3398e_row16_col8 {
  font-size: 6pt;
  background-color: #ed392b;
  color: #f1f1f1;
}
#T_3398e_row0_col14, #T_3398e_row12_col13 {
  font-size: 6pt;
  background-color: #e32f27;
  color: #f1f1f1;
}
#T_3398e_row0_col15, #T_3398e_row11_col7 {
  font-size: 6pt;
  background-color: #fc8464;
  color: #f1f1f1;
}
#T_3398e_row0_col16, #T_3398e_row13_col2, #T_3398e_row21_col13 {
  font-size: 6pt;
  background-color: #fdd2bf;
  color: #000000;
}
#T_3398e_row0_col17, #T_3398e_row5_col16, #T_3398e_row10_col2, #T_3398e_row10_col3, #T_3398e_row10_col19, #T_3398e_row12_col11, #T_3398e_row17_col3 {
  font-size: 6pt;
  background-color: #fcb99f;
  color: #000000;
}
#T_3398e_row0_col18, #T_3398e_row20_col15 {
  font-size: 6pt;
  background-color: #fca588;
  color: #000000;
}
#T_3398e_row0_col19, #T_3398e_row23_col5 {
  font-size: 6pt;
  background-color: #fcaa8d;
  color: #000000;
}
#T_3398e_row1_col0 {
  font-size: 6pt;
  background-color: #a50f15;
  color: #f1f1f1;
}
#T_3398e_row1_col1, #T_3398e_row8_col8 {
  font-size: 6pt;
  background-color: #920a13;
  color: #f1f1f1;
}
#T_3398e_row1_col4 {
  font-size: 6pt;
  background-color: #6d010e;
  color: #f1f1f1;
}
#T_3398e_row1_col5, #T_3398e_row2_col0 {
  font-size: 6pt;
  background-color: #860811;
  color: #f1f1f1;
}
#T_3398e_row1_col6 {
  font-size: 6pt;
  background-color: #ac1117;
  color: #f1f1f1;
}
#T_3398e_row1_col7, #T_3398e_row6_col0, #T_3398e_row10_col5, #T_3398e_row13_col9, #T_3398e_row17_col14 {
  font-size: 6pt;
  background-color: #f85d42;
  color: #f1f1f1;
}
#T_3398e_row1_col9, #T_3398e_row8_col13, #T_3398e_row12_col0 {
  font-size: 6pt;
  background-color: #fee8dd;
  color: #000000;
}
#T_3398e_row1_col10 {
  font-size: 6pt;
  background-color: #fff2ec;
  color: #000000;
}
#T_3398e_row1_col11, #T_3398e_row8_col7 {
  font-size: 6pt;
  background-color: #fb7555;
  color: #f1f1f1;
}
#T_3398e_row1_col12, #T_3398e_row10_col4, #T_3398e_row11_col4, #T_3398e_row14_col13, #T_3398e_row17_col12, #T_3398e_row17_col13 {
  font-size: 6pt;
  background-color: #f7593f;
  color: #f1f1f1;
}
#T_3398e_row1_col14, #T_3398e_row8_col3, #T_3398e_row19_col1, #T_3398e_row22_col0 {
  font-size: 6pt;
  background-color: #fff0e8;
  color: #000000;
}
#T_3398e_row1_col15, #T_3398e_row2_col18, #T_3398e_row2_col19, #T_3398e_row4_col14, #T_3398e_row6_col14, #T_3398e_row9_col1, #T_3398e_row15_col1, #T_3398e_row23_col18 {
  font-size: 6pt;
  background-color: #fee4d8;
  color: #000000;
}
#T_3398e_row1_col16, #T_3398e_row4_col17, #T_3398e_row18_col3, #T_3398e_row20_col14 {
  font-size: 6pt;
  background-color: #fdc9b3;
  color: #000000;
}
#T_3398e_row1_col17, #T_3398e_row2_col9, #T_3398e_row3_col11 {
  font-size: 6pt;
  background-color: #fcb499;
  color: #000000;
}
#T_3398e_row1_col18, #T_3398e_row7_col2, #T_3398e_row15_col2, #T_3398e_row21_col7 {
  font-size: 6pt;
  background-color: #fdd7c6;
  color: #000000;
}
#T_3398e_row1_col19, #T_3398e_row19_col12, #T_3398e_row19_col14, #T_3398e_row20_col12, #T_3398e_row24_col2 {
  font-size: 6pt;
  background-color: #fedbcc;
  color: #000000;
}
#T_3398e_row2_col3 {
  font-size: 6pt;
  background-color: #7c0510;
  color: #f1f1f1;
}
#T_3398e_row2_col4, #T_3398e_row9_col9 {
  font-size: 6pt;
  background-color: #9f0e14;
  color: #f1f1f1;
}
#T_3398e_row2_col5 {
  font-size: 6pt;
  background-color: #c8171c;
  color: #f1f1f1;
}
#T_3398e_row2_col6, #T_3398e_row3_col13, #T_3398e_row8_col16 {
  font-size: 6pt;
  background-color: #fc8d6d;
  color: #f1f1f1;
}
#T_3398e_row2_col7, #T_3398e_row16_col2, #T_3398e_row18_col19, #T_3398e_row19_col13, #T_3398e_row20_col8 {
  font-size: 6pt;
  background-color: #fedfd0;
  color: #000000;
}
#T_3398e_row2_col10, #T_3398e_row5_col12 {
  font-size: 6pt;
  background-color: #e43027;
  color: #f1f1f1;
}
#T_3398e_row2_col11, #T_3398e_row8_col0 {
  font-size: 6pt;
  background-color: #fcae92;
  color: #000000;
}
#T_3398e_row2_col14, #T_3398e_row2_col17, #T_3398e_row18_col10, #T_3398e_row19_col3 {
  font-size: 6pt;
  background-color: #fedecf;
  color: #000000;
}
#T_3398e_row2_col15, #T_3398e_row9_col0, #T_3398e_row20_col3 {
  font-size: 6pt;
  background-color: #fed9c9;
  color: #000000;
}
#T_3398e_row2_col16, #T_3398e_row14_col1, #T_3398e_row20_col11 {
  font-size: 6pt;
  background-color: #fee5d9;
  color: #000000;
}
#T_3398e_row3_col0 {
  font-size: 6pt;
  background-color: #b61319;
  color: #f1f1f1;
}
#T_3398e_row3_col2 {
  font-size: 6pt;
  background-color: #7e0610;
  color: #f1f1f1;
}
#T_3398e_row3_col3 {
  font-size: 6pt;
  background-color: #980c13;
  color: #f1f1f1;
}
#T_3398e_row3_col4 {
  font-size: 6pt;
  background-color: #d42121;
  color: #f1f1f1;
}
#T_3398e_row3_col5, #T_3398e_row12_col3 {
  font-size: 6pt;
  background-color: #fca98c;
  color: #000000;
}
#T_3398e_row3_col6, #T_3398e_row21_col18 {
  font-size: 6pt;
  background-color: #fee2d5;
  color: #000000;
}
#T_3398e_row3_col7, #T_3398e_row7_col6, #T_3398e_row20_col0 {
  font-size: 6pt;
  background-color: #fff4ef;
  color: #000000;
}
#T_3398e_row3_col8, #T_3398e_row11_col11 {
  font-size: 6pt;
  background-color: #fc9c7d;
  color: #000000;
}
#T_3398e_row3_col9 {
  font-size: 6pt;
  background-color: #fa6648;
  color: #f1f1f1;
}
#T_3398e_row3_col12, #T_3398e_row7_col11 {
  font-size: 6pt;
  background-color: #f03d2d;
  color: #f1f1f1;
}
#T_3398e_row3_col16 {
  font-size: 6pt;
  background-color: #ffece3;
  color: #000000;
}
#T_3398e_row3_col17, #T_3398e_row4_col5, #T_3398e_row4_col19, #T_3398e_row7_col8, #T_3398e_row10_col16, #T_3398e_row19_col19, #T_3398e_row22_col16 {
  font-size: 6pt;
  background-color: #fee3d6;
  color: #000000;
}
#T_3398e_row4_col1, #T_3398e_row6_col17 {
  font-size: 6pt;
  background-color: #f96044;
  color: #f1f1f1;
}
#T_3398e_row4_col2, #T_3398e_row18_col17 {
  font-size: 6pt;
  background-color: #f75b40;
  color: #f1f1f1;
}
#T_3398e_row4_col3, #T_3398e_row7_col9, #T_3398e_row12_col5, #T_3398e_row13_col5, #T_3398e_row14_col12, #T_3398e_row16_col11, #T_3398e_row17_col9 {
  font-size: 6pt;
  background-color: #fb7757;
  color: #f1f1f1;
}
#T_3398e_row4_col4 {
  font-size: 6pt;
  background-color: #fca78b;
  color: #000000;
}
#T_3398e_row4_col8, #T_3398e_row16_col4, #T_3398e_row20_col17 {
  font-size: 6pt;
  background-color: #fc9070;
  color: #000000;
}
#T_3398e_row4_col10, #T_3398e_row11_col3, #T_3398e_row14_col2, #T_3398e_row21_col17, #T_3398e_row23_col8 {
  font-size: 6pt;
  background-color: #fdcebb;
  color: #000000;
}
#T_3398e_row4_col13, #T_3398e_row9_col8 {
  font-size: 6pt;
  background-color: #bb141a;
  color: #f1f1f1;
}
#T_3398e_row4_col15, #T_3398e_row23_col3 {
  font-size: 6pt;
  background-color: #ffeee6;
  color: #000000;
}
#T_3398e_row4_col16, #T_3398e_row5_col15, #T_3398e_row6_col2, #T_3398e_row8_col5, #T_3398e_row19_col6, #T_3398e_row19_col8, #T_3398e_row19_col9, #T_3398e_row20_col9, #T_3398e_row21_col11 {
  font-size: 6pt;
  background-color: #fedccd;
  color: #000000;
}
#T_3398e_row4_col18, #T_3398e_row10_col18 {
  font-size: 6pt;
  background-color: #fee0d2;
  color: #000000;
}
#T_3398e_row5_col0 {
  font-size: 6pt;
  background-color: #f96245;
  color: #f1f1f1;
}
#T_3398e_row5_col1, #T_3398e_row11_col12, #T_3398e_row16_col12 {
  font-size: 6pt;
  background-color: #f5523a;
  color: #f1f1f1;
}
#T_3398e_row5_col2, #T_3398e_row9_col19 {
  font-size: 6pt;
  background-color: #fc9b7c;
  color: #000000;
}
#T_3398e_row5_col3, #T_3398e_row6_col3, #T_3398e_row17_col2, #T_3398e_row19_col11 {
  font-size: 6pt;
  background-color: #fee7dc;
  color: #000000;
}
#T_3398e_row5_col4 {
  font-size: 6pt;
  background-color: #fee5d8;
  color: #000000;
}
#T_3398e_row5_col6, #T_3398e_row6_col6, #T_3398e_row7_col5, #T_3398e_row11_col1, #T_3398e_row18_col11, #T_3398e_row23_col12, #T_3398e_row24_col4, #T_3398e_row24_col6 {
  font-size: 6pt;
  background-color: #fdc7b2;
  color: #000000;
}
#T_3398e_row5_col7, #T_3398e_row22_col6 {
  font-size: 6pt;
  background-color: #fdc6b0;
  color: #000000;
}
#T_3398e_row5_col9, #T_3398e_row17_col0, #T_3398e_row18_col1 {
  font-size: 6pt;
  background-color: #feeae1;
  color: #000000;
}
#T_3398e_row5_col10, #T_3398e_row7_col17, #T_3398e_row17_col16 {
  font-size: 6pt;
  background-color: #e22e27;
  color: #f1f1f1;
}
#T_3398e_row5_col11 {
  font-size: 6pt;
  background-color: #be151a;
  color: #f1f1f1;
}
#T_3398e_row5_col13, #T_3398e_row9_col4, #T_3398e_row13_col4, #T_3398e_row14_col11, #T_3398e_row15_col4 {
  font-size: 6pt;
  background-color: #fb7353;
  color: #f1f1f1;
}
#T_3398e_row5_col14 {
  font-size: 6pt;
  background-color: #fee9df;
  color: #000000;
}
#T_3398e_row5_col17, #T_3398e_row5_col19, #T_3398e_row17_col11, #T_3398e_row18_col16 {
  font-size: 6pt;
  background-color: #fc8565;
  color: #f1f1f1;
}
#T_3398e_row5_col18, #T_3398e_row14_col18, #T_3398e_row15_col7, #T_3398e_row15_col12 {
  font-size: 6pt;
  background-color: #fb694a;
  color: #f1f1f1;
}
#T_3398e_row6_col1, #T_3398e_row11_col10, #T_3398e_row24_col13 {
  font-size: 6pt;
  background-color: #fc7f5f;
  color: #f1f1f1;
}
#T_3398e_row6_col5, #T_3398e_row8_col1, #T_3398e_row21_col12, #T_3398e_row23_col7 {
  font-size: 6pt;
  background-color: #fdd3c1;
  color: #000000;
}
#T_3398e_row6_col8, #T_3398e_row10_col1, #T_3398e_row19_col10 {
  font-size: 6pt;
  background-color: #ffefe8;
  color: #000000;
}
#T_3398e_row6_col9, #T_3398e_row7_col13, #T_3398e_row9_col14, #T_3398e_row20_col6, #T_3398e_row23_col13, #T_3398e_row23_col14 {
  font-size: 6pt;
  background-color: #fdd4c2;
  color: #000000;
}
#T_3398e_row6_col10, #T_3398e_row13_col19 {
  font-size: 6pt;
  background-color: #cf1c1f;
  color: #f1f1f1;
}
#T_3398e_row6_col11 {
  font-size: 6pt;
  background-color: #f34a36;
  color: #f1f1f1;
}
#T_3398e_row6_col12, #T_3398e_row7_col12, #T_3398e_row11_col5, #T_3398e_row13_col3, #T_3398e_row13_col12, #T_3398e_row14_col3 {
  font-size: 6pt;
  background-color: #fb6e4e;
  color: #f1f1f1;
}
#T_3398e_row6_col13, #T_3398e_row9_col16, #T_3398e_row17_col4, #T_3398e_row17_col5, #T_3398e_row19_col15, #T_3398e_row21_col19, #T_3398e_row23_col6 {
  font-size: 6pt;
  background-color: #fcc3ab;
  color: #000000;
}
#T_3398e_row6_col15, #T_3398e_row18_col7, #T_3398e_row24_col10 {
  font-size: 6pt;
  background-color: #fcc1a8;
  color: #000000;
}
#T_3398e_row6_col16, #T_3398e_row20_col16 {
  font-size: 6pt;
  background-color: #fca689;
  color: #000000;
}
#T_3398e_row6_col18 {
  font-size: 6pt;
  background-color: #e53228;
  color: #f1f1f1;
}
#T_3398e_row7_col0 {
  font-size: 6pt;
  background-color: #fb6c4c;
  color: #f1f1f1;
}
#T_3398e_row7_col1, #T_3398e_row19_col4 {
  font-size: 6pt;
  background-color: #fcc2aa;
  color: #000000;
}
#T_3398e_row7_col4, #T_3398e_row12_col2 {
  font-size: 6pt;
  background-color: #fee3d7;
  color: #000000;
}
#T_3398e_row7_col7, #T_3398e_row9_col2 {
  font-size: 6pt;
  background-color: #fff1ea;
  color: #000000;
}
#T_3398e_row7_col14, #T_3398e_row8_col6, #T_3398e_row20_col19, #T_3398e_row22_col1, #T_3398e_row22_col9, #T_3398e_row22_col13, #T_3398e_row23_col15 {
  font-size: 6pt;
  background-color: #fdcbb6;
  color: #000000;
}
#T_3398e_row7_col15 {
  font-size: 6pt;
  background-color: #fc9576;
  color: #000000;
}
#T_3398e_row7_col16, #T_3398e_row11_col16 {
  font-size: 6pt;
  background-color: #fc8262;
  color: #f1f1f1;
}
#T_3398e_row7_col18 {
  font-size: 6pt;
  background-color: #af1117;
  color: #f1f1f1;
}
#T_3398e_row7_col19 {
  font-size: 6pt;
  background-color: #d21f20;
  color: #f1f1f1;
}
#T_3398e_row8_col2, #T_3398e_row9_col13, #T_3398e_row16_col0 {
  font-size: 6pt;
  background-color: #ffede5;
  color: #000000;
}
#T_3398e_row8_col4, #T_3398e_row19_col18, #T_3398e_row21_col5 {
  font-size: 6pt;
  background-color: #fcb89e;
  color: #000000;
}
#T_3398e_row8_col10 {
  font-size: 6pt;
  background-color: #75030f;
  color: #f1f1f1;
}
#T_3398e_row8_col11, #T_3398e_row24_col19 {
  font-size: 6pt;
  background-color: #dc2924;
  color: #f1f1f1;
}
#T_3398e_row8_col12, #T_3398e_row17_col7 {
  font-size: 6pt;
  background-color: #fc8666;
  color: #f1f1f1;
}
#T_3398e_row8_col14 {
  font-size: 6pt;
  background-color: #fcc4ad;
  color: #000000;
}
#T_3398e_row8_col15, #T_3398e_row14_col0 {
  font-size: 6pt;
  background-color: #fc9373;
  color: #000000;
}
#T_3398e_row8_col17 {
  font-size: 6pt;
  background-color: #f14331;
  color: #f1f1f1;
}
#T_3398e_row8_col18 {
  font-size: 6pt;
  background-color: #b21218;
  color: #f1f1f1;
}
#T_3398e_row8_col19 {
  font-size: 6pt;
  background-color: #b81419;
  color: #f1f1f1;
}
#T_3398e_row9_col3, #T_3398e_row17_col6, #T_3398e_row18_col13 {
  font-size: 6pt;
  background-color: #fcb398;
  color: #000000;
}
#T_3398e_row9_col5 {
  font-size: 6pt;
  background-color: #ee3a2c;
  color: #f1f1f1;
}
#T_3398e_row9_col6, #T_3398e_row11_col8 {
  font-size: 6pt;
  background-color: #e02c26;
  color: #f1f1f1;
}
#T_3398e_row9_col7 {
  font-size: 6pt;
  background-color: #d01d1f;
  color: #f1f1f1;
}
#T_3398e_row9_col10 {
  font-size: 6pt;
  background-color: #b71319;
  color: #f1f1f1;
}
#T_3398e_row9_col11 {
  font-size: 6pt;
  background-color: #f44d38;
  color: #f1f1f1;
}
#T_3398e_row9_col15, #T_3398e_row22_col5, #T_3398e_row22_col15, #T_3398e_row23_col4, #T_3398e_row24_col9 {
  font-size: 6pt;
  background-color: #fcb296;
  color: #000000;
}
#T_3398e_row9_col17, #T_3398e_row18_col8 {
  font-size: 6pt;
  background-color: #fcbda4;
  color: #000000;
}
#T_3398e_row9_col18, #T_3398e_row18_col9, #T_3398e_row24_col5 {
  font-size: 6pt;
  background-color: #fcbea5;
  color: #000000;
}
#T_3398e_row10_col0, #T_3398e_row22_col17, #T_3398e_row24_col3 {
  font-size: 6pt;
  background-color: #fee6da;
  color: #000000;
}
#T_3398e_row10_col6 {
  font-size: 6pt;
  background-color: #fb7151;
  color: #f1f1f1;
}
#T_3398e_row10_col7, #T_3398e_row10_col12, #T_3398e_row14_col4, #T_3398e_row14_col9, #T_3398e_row15_col9, #T_3398e_row24_col17 {
  font-size: 6pt;
  background-color: #fb6d4d;
  color: #f1f1f1;
}
#T_3398e_row10_col8 {
  font-size: 6pt;
  background-color: #cc191e;
  color: #f1f1f1;
}
#T_3398e_row10_col9, #T_3398e_row13_col15 {
  font-size: 6pt;
  background-color: #d92523;
  color: #f1f1f1;
}
#T_3398e_row10_col10 {
  font-size: 6pt;
  background-color: #f24734;
  color: #f1f1f1;
}
#T_3398e_row10_col11, #T_3398e_row17_col18 {
  font-size: 6pt;
  background-color: #fc8060;
  color: #f1f1f1;
}
#T_3398e_row10_col15 {
  font-size: 6pt;
  background-color: #f34935;
  color: #f1f1f1;
}
#T_3398e_row10_col17 {
  font-size: 6pt;
  background-color: #fee1d4;
  color: #000000;
}
#T_3398e_row11_col2 {
  font-size: 6pt;
  background-color: #fcad90;
  color: #000000;
}
#T_3398e_row11_col6, #T_3398e_row12_col10, #T_3398e_row14_col5, #T_3398e_row14_col7 {
  font-size: 6pt;
  background-color: #fb7b5b;
  color: #f1f1f1;
}
#T_3398e_row11_col9 {
  font-size: 6pt;
  background-color: #f5533b;
  color: #f1f1f1;
}
#T_3398e_row11_col13 {
  font-size: 6pt;
  background-color: #ab1016;
  color: #f1f1f1;
}
#T_3398e_row11_col14, #T_3398e_row14_col17 {
  font-size: 6pt;
  background-color: #900a12;
  color: #f1f1f1;
}
#T_3398e_row11_col17, #T_3398e_row15_col3, #T_3398e_row15_col5, #T_3398e_row15_col6, #T_3398e_row24_col14 {
  font-size: 6pt;
  background-color: #fb7d5d;
  color: #f1f1f1;
}
#T_3398e_row11_col18 {
  font-size: 6pt;
  background-color: #f03f2e;
  color: #f1f1f1;
}
#T_3398e_row11_col19 {
  font-size: 6pt;
  background-color: #ea362a;
  color: #f1f1f1;
}
#T_3398e_row12_col1, #T_3398e_row17_col10 {
  font-size: 6pt;
  background-color: #fc8767;
  color: #f1f1f1;
}
#T_3398e_row12_col4 {
  font-size: 6pt;
  background-color: #fa6849;
  color: #f1f1f1;
}
#T_3398e_row12_col6, #T_3398e_row13_col10 {
  font-size: 6pt;
  background-color: #fc8b6b;
  color: #f1f1f1;
}
#T_3398e_row12_col7, #T_3398e_row13_col0 {
  font-size: 6pt;
  background-color: #fc9d7f;
  color: #000000;
}
#T_3398e_row12_col8 {
  font-size: 6pt;
  background-color: #d82422;
  color: #f1f1f1;
}
#T_3398e_row12_col9, #T_3398e_row16_col18 {
  font-size: 6pt;
  background-color: #f4503a;
  color: #f1f1f1;
}
#T_3398e_row12_col12 {
  font-size: 6pt;
  background-color: #fb7858;
  color: #f1f1f1;
}
#T_3398e_row12_col14, #T_3398e_row14_col15 {
  font-size: 6pt;
  background-color: #e12d26;
  color: #f1f1f1;
}
#T_3398e_row12_col15, #T_3398e_row16_col15 {
  font-size: 6pt;
  background-color: #a60f15;
  color: #f1f1f1;
}
#T_3398e_row12_col17 {
  font-size: 6pt;
  background-color: #8c0912;
  color: #f1f1f1;
}
#T_3398e_row13_col1, #T_3398e_row17_col19, #T_3398e_row22_col14 {
  font-size: 6pt;
  background-color: #fcbfa7;
  color: #000000;
}
#T_3398e_row13_col6, #T_3398e_row15_col10, #T_3398e_row16_col6 {
  font-size: 6pt;
  background-color: #fc9474;
  color: #000000;
}
#T_3398e_row13_col7, #T_3398e_row13_col11, #T_3398e_row21_col15 {
  font-size: 6pt;
  background-color: #fca285;
  color: #000000;
}
#T_3398e_row13_col8 {
  font-size: 6pt;
  background-color: #e63328;
  color: #f1f1f1;
}
#T_3398e_row13_col13, #T_3398e_row16_col14 {
  font-size: 6pt;
  background-color: #f24633;
  color: #f1f1f1;
}
#T_3398e_row13_col14 {
  font-size: 6pt;
  background-color: #ec382b;
  color: #f1f1f1;
}
#T_3398e_row13_col16, #T_3398e_row15_col17 {
  font-size: 6pt;
  background-color: #b01217;
  color: #f1f1f1;
}
#T_3398e_row13_col18 {
  font-size: 6pt;
  background-color: #ca181d;
  color: #f1f1f1;
}
#T_3398e_row14_col6 {
  font-size: 6pt;
  background-color: #fc8969;
  color: #f1f1f1;
}
#T_3398e_row14_col8, #T_3398e_row15_col8 {
  font-size: 6pt;
  background-color: #eb372a;
  color: #f1f1f1;
}
#T_3398e_row14_col10 {
  font-size: 6pt;
  background-color: #fc9879;
  color: #000000;
}
#T_3398e_row14_col14, #T_3398e_row15_col14, #T_3398e_row16_col13 {
  font-size: 6pt;
  background-color: #f34c37;
  color: #f1f1f1;
}
#T_3398e_row14_col16 {
  font-size: 6pt;
  background-color: #c2161b;
  color: #f1f1f1;
}
#T_3398e_row15_col0, #T_3398e_row19_col7, #T_3398e_row20_col7, #T_3398e_row20_col13, #T_3398e_row21_col2, #T_3398e_row22_col18 {
  font-size: 6pt;
  background-color: #fedaca;
  color: #000000;
}
#T_3398e_row15_col11, #T_3398e_row24_col16 {
  font-size: 6pt;
  background-color: #fb7a5a;
  color: #f1f1f1;
}
#T_3398e_row15_col13 {
  font-size: 6pt;
  background-color: #f44f39;
  color: #f1f1f1;
}
#T_3398e_row15_col15 {
  font-size: 6pt;
  background-color: #cb181d;
  color: #f1f1f1;
}
#T_3398e_row15_col16, #T_3398e_row17_col15 {
  font-size: 6pt;
  background-color: #c5171c;
  color: #f1f1f1;
}
#T_3398e_row15_col18, #T_3398e_row24_col12 {
  font-size: 6pt;
  background-color: #fc8f6f;
  color: #000000;
}
#T_3398e_row15_col19, #T_3398e_row16_col19, #T_3398e_row24_col1 {
  font-size: 6pt;
  background-color: #fca183;
  color: #000000;
}
#T_3398e_row16_col1 {
  font-size: 6pt;
  background-color: #fee8de;
  color: #000000;
}
#T_3398e_row16_col3, #T_3398e_row16_col5, #T_3398e_row23_col0 {
  font-size: 6pt;
  background-color: #fc9777;
  color: #000000;
}
#T_3398e_row16_col7 {
  font-size: 6pt;
  background-color: #fb7252;
  color: #f1f1f1;
}
#T_3398e_row16_col9 {
  font-size: 6pt;
  background-color: #f96346;
  color: #f1f1f1;
}
#T_3398e_row16_col10 {
  font-size: 6pt;
  background-color: #fc8161;
  color: #f1f1f1;
}
#T_3398e_row16_col16 {
  font-size: 6pt;
  background-color: #d11e1f;
  color: #f1f1f1;
}
#T_3398e_row16_col17 {
  font-size: 6pt;
  background-color: #b91419;
  color: #f1f1f1;
}
#T_3398e_row17_col1, #T_3398e_row20_col10, #T_3398e_row21_col1, #T_3398e_row21_col10, #T_3398e_row22_col10 {
  font-size: 6pt;
  background-color: #feeae0;
  color: #000000;
}
#T_3398e_row17_col8 {
  font-size: 6pt;
  background-color: #f85f43;
  color: #f1f1f1;
}
#T_3398e_row17_col17 {
  font-size: 6pt;
  background-color: #d52221;
  color: #f1f1f1;
}
#T_3398e_row18_col0, #T_3398e_row20_col2, #T_3398e_row22_col3 {
  font-size: 6pt;
  background-color: #ffece4;
  color: #000000;
}
#T_3398e_row18_col2 {
  font-size: 6pt;
  background-color: #ffeee7;
  color: #000000;
}
#T_3398e_row18_col4, #T_3398e_row20_col5, #T_3398e_row24_col7 {
  font-size: 6pt;
  background-color: #fdcab5;
  color: #000000;
}
#T_3398e_row18_col5, #T_3398e_row18_col6, #T_3398e_row19_col5, #T_3398e_row21_col6 {
  font-size: 6pt;
  background-color: #fdccb8;
  color: #000000;
}
#T_3398e_row18_col12, #T_3398e_row21_col4 {
  font-size: 6pt;
  background-color: #fcaf93;
  color: #000000;
}
#T_3398e_row18_col14 {
  font-size: 6pt;
  background-color: #fcb69b;
  color: #000000;
}
#T_3398e_row18_col15 {
  font-size: 6pt;
  background-color: #fc8a6a;
  color: #f1f1f1;
}
#T_3398e_row18_col18, #T_3398e_row20_col18 {
  font-size: 6pt;
  background-color: #fdc5ae;
  color: #000000;
}
#T_3398e_row19_col0, #T_3398e_row19_col2, #T_3398e_row22_col2 {
  font-size: 6pt;
  background-color: #fff2eb;
  color: #000000;
}
#T_3398e_row19_col16 {
  font-size: 6pt;
  background-color: #fcb79c;
  color: #000000;
}
#T_3398e_row21_col8, #T_3398e_row22_col11 {
  font-size: 6pt;
  background-color: #fed8c7;
  color: #000000;
}
#T_3398e_row21_col9, #T_3398e_row22_col8, #T_3398e_row22_col12, #T_3398e_row23_col11 {
  font-size: 6pt;
  background-color: #fdcdb9;
  color: #000000;
}
#T_3398e_row21_col16 {
  font-size: 6pt;
  background-color: #fcb095;
  color: #000000;
}
#T_3398e_row22_col4 {
  font-size: 6pt;
  background-color: #fca082;
  color: #000000;
}
#T_3398e_row22_col7, #T_3398e_row23_col19 {
  font-size: 6pt;
  background-color: #fdd1be;
  color: #000000;
}
#T_3398e_row22_col19 {
  font-size: 6pt;
  background-color: #fdd5c4;
  color: #000000;
}
#T_3398e_row23_col1 {
  font-size: 6pt;
  background-color: #fc9272;
  color: #000000;
}
#T_3398e_row23_col9 {
  font-size: 6pt;
  background-color: #fdd0bc;
  color: #000000;
}
#T_3398e_row23_col10 {
  font-size: 6pt;
  background-color: #fee7db;
  color: #000000;
}
#T_3398e_row24_col0 {
  font-size: 6pt;
  background-color: #fc9e80;
  color: #000000;
}
#T_3398e_row24_col11 {
  font-size: 6pt;
  background-color: #fcab8f;
  color: #000000;
}
#T_3398e_row24_col15 {
  font-size: 6pt;
  background-color: #f14130;
  color: #f1f1f1;
}
#T_3398e_row24_col18 {
  font-size: 6pt;
  background-color: #db2824;
  color: #f1f1f1;
}
</style>
<table id="T_3398e">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_3398e_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_3398e_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_3398e_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_3398e_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_3398e_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_3398e_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_3398e_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_3398e_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_3398e_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_3398e_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_3398e_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_3398e_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_3398e_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_3398e_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_3398e_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_3398e_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_3398e_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_3398e_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_3398e_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_3398e_level0_col19" class="col_heading level0 col19" >19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_3398e_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_3398e_row0_col0" class="data row0 col0" >255</td>
      <td id="T_3398e_row0_col1" class="data row0 col1" >236</td>
      <td id="T_3398e_row0_col2" class="data row0 col2" >230</td>
      <td id="T_3398e_row0_col3" class="data row0 col3" >251</td>
      <td id="T_3398e_row0_col4" class="data row0 col4" >254</td>
      <td id="T_3398e_row0_col5" class="data row0 col5" >252</td>
      <td id="T_3398e_row0_col6" class="data row0 col6" >244</td>
      <td id="T_3398e_row0_col7" class="data row0 col7" >229</td>
      <td id="T_3398e_row0_col8" class="data row0 col8" >181</td>
      <td id="T_3398e_row0_col9" class="data row0 col9" >108</td>
      <td id="T_3398e_row0_col10" class="data row0 col10" >62</td>
      <td id="T_3398e_row0_col11" class="data row0 col11" >45</td>
      <td id="T_3398e_row0_col12" class="data row0 col12" >77</td>
      <td id="T_3398e_row0_col13" class="data row0 col13" >128</td>
      <td id="T_3398e_row0_col14" class="data row0 col14" >123</td>
      <td id="T_3398e_row0_col15" class="data row0 col15" >78</td>
      <td id="T_3398e_row0_col16" class="data row0 col16" >49</td>
      <td id="T_3398e_row0_col17" class="data row0 col17" >47</td>
      <td id="T_3398e_row0_col18" class="data row0 col18" >52</td>
      <td id="T_3398e_row0_col19" class="data row0 col19" >49</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_3398e_row1_col0" class="data row1 col0" >226</td>
      <td id="T_3398e_row1_col1" class="data row1 col1" >234</td>
      <td id="T_3398e_row1_col2" class="data row1 col2" >254</td>
      <td id="T_3398e_row1_col3" class="data row1 col3" >255</td>
      <td id="T_3398e_row1_col4" class="data row1 col4" >251</td>
      <td id="T_3398e_row1_col5" class="data row1 col5" >238</td>
      <td id="T_3398e_row1_col6" class="data row1 col6" >214</td>
      <td id="T_3398e_row1_col7" class="data row1 col7" >143</td>
      <td id="T_3398e_row1_col8" class="data row1 col8" >79</td>
      <td id="T_3398e_row1_col9" class="data row1 col9" >53</td>
      <td id="T_3398e_row1_col10" class="data row1 col10" >61</td>
      <td id="T_3398e_row1_col11" class="data row1 col11" >112</td>
      <td id="T_3398e_row1_col12" class="data row1 col12" >125</td>
      <td id="T_3398e_row1_col13" class="data row1 col13" >76</td>
      <td id="T_3398e_row1_col14" class="data row1 col14" >42</td>
      <td id="T_3398e_row1_col15" class="data row1 col15" >44</td>
      <td id="T_3398e_row1_col16" class="data row1 col16" >53</td>
      <td id="T_3398e_row1_col17" class="data row1 col17" >49</td>
      <td id="T_3398e_row1_col18" class="data row1 col18" >38</td>
      <td id="T_3398e_row1_col19" class="data row1 col19" >32</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_3398e_row2_col0" class="data row2 col0" >240</td>
      <td id="T_3398e_row2_col1" class="data row2 col1" >255</td>
      <td id="T_3398e_row2_col2" class="data row2 col2" >250</td>
      <td id="T_3398e_row2_col3" class="data row2 col3" >245</td>
      <td id="T_3398e_row2_col4" class="data row2 col4" >229</td>
      <td id="T_3398e_row2_col5" class="data row2 col5" >201</td>
      <td id="T_3398e_row2_col6" class="data row2 col6" >122</td>
      <td id="T_3398e_row2_col7" class="data row2 col7" >69</td>
      <td id="T_3398e_row2_col8" class="data row2 col8" >45</td>
      <td id="T_3398e_row2_col9" class="data row2 col9" >79</td>
      <td id="T_3398e_row2_col10" class="data row2 col10" >127</td>
      <td id="T_3398e_row2_col11" class="data row2 col11" >87</td>
      <td id="T_3398e_row2_col12" class="data row2 col12" >36</td>
      <td id="T_3398e_row2_col13" class="data row2 col13" >41</td>
      <td id="T_3398e_row2_col14" class="data row2 col14" >55</td>
      <td id="T_3398e_row2_col15" class="data row2 col15" >49</td>
      <td id="T_3398e_row2_col16" class="data row2 col16" >38</td>
      <td id="T_3398e_row2_col17" class="data row2 col17" >33</td>
      <td id="T_3398e_row2_col18" class="data row2 col18" >33</td>
      <td id="T_3398e_row2_col19" class="data row2 col19" >28</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_3398e_row3_col0" class="data row3 col0" >213</td>
      <td id="T_3398e_row3_col1" class="data row3 col1" >230</td>
      <td id="T_3398e_row3_col2" class="data row3 col2" >243</td>
      <td id="T_3398e_row3_col3" class="data row3 col3" >234</td>
      <td id="T_3398e_row3_col4" class="data row3 col4" >193</td>
      <td id="T_3398e_row3_col5" class="data row3 col5" >104</td>
      <td id="T_3398e_row3_col6" class="data row3 col6" >66</td>
      <td id="T_3398e_row3_col7" class="data row3 col7" >46</td>
      <td id="T_3398e_row3_col8" class="data row3 col8" >92</td>
      <td id="T_3398e_row3_col9" class="data row3 col9" >112</td>
      <td id="T_3398e_row3_col10" class="data row3 col10" >59</td>
      <td id="T_3398e_row3_col11" class="data row3 col11" >84</td>
      <td id="T_3398e_row3_col12" class="data row3 col12" >137</td>
      <td id="T_3398e_row3_col13" class="data row3 col13" >95</td>
      <td id="T_3398e_row3_col14" class="data row3 col14" >38</td>
      <td id="T_3398e_row3_col15" class="data row3 col15" >33</td>
      <td id="T_3398e_row3_col16" class="data row3 col16" >33</td>
      <td id="T_3398e_row3_col17" class="data row3 col17" >30</td>
      <td id="T_3398e_row3_col18" class="data row3 col18" >24</td>
      <td id="T_3398e_row3_col19" class="data row3 col19" >17</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_3398e_row4_col0" class="data row4 col0" >136</td>
      <td id="T_3398e_row4_col1" class="data row4 col1" >143</td>
      <td id="T_3398e_row4_col2" class="data row4 col2" >150</td>
      <td id="T_3398e_row4_col3" class="data row4 col3" >139</td>
      <td id="T_3398e_row4_col4" class="data row4 col4" >106</td>
      <td id="T_3398e_row4_col5" class="data row4 col5" >62</td>
      <td id="T_3398e_row4_col6" class="data row4 col6" >43</td>
      <td id="T_3398e_row4_col7" class="data row4 col7" >91</td>
      <td id="T_3398e_row4_col8" class="data row4 col8" >97</td>
      <td id="T_3398e_row4_col9" class="data row4 col9" >42</td>
      <td id="T_3398e_row4_col10" class="data row4 col10" >78</td>
      <td id="T_3398e_row4_col11" class="data row4 col11" >189</td>
      <td id="T_3398e_row4_col12" class="data row4 col12" >199</td>
      <td id="T_3398e_row4_col13" class="data row4 col13" >152</td>
      <td id="T_3398e_row4_col14" class="data row4 col14" >51</td>
      <td id="T_3398e_row4_col15" class="data row4 col15" >38</td>
      <td id="T_3398e_row4_col16" class="data row4 col16" >44</td>
      <td id="T_3398e_row4_col17" class="data row4 col17" >41</td>
      <td id="T_3398e_row4_col18" class="data row4 col18" >35</td>
      <td id="T_3398e_row4_col19" class="data row4 col19" >29</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_3398e_row5_col0" class="data row5 col0" >145</td>
      <td id="T_3398e_row5_col1" class="data row5 col1" >152</td>
      <td id="T_3398e_row5_col2" class="data row5 col2" >107</td>
      <td id="T_3398e_row5_col3" class="data row5 col3" >59</td>
      <td id="T_3398e_row5_col4" class="data row5 col4" >60</td>
      <td id="T_3398e_row5_col5" class="data row5 col5" >38</td>
      <td id="T_3398e_row5_col6" class="data row5 col6" >85</td>
      <td id="T_3398e_row5_col7" class="data row5 col7" >84</td>
      <td id="T_3398e_row5_col8" class="data row5 col8" >45</td>
      <td id="T_3398e_row5_col9" class="data row5 col9" >51</td>
      <td id="T_3398e_row5_col10" class="data row5 col10" >128</td>
      <td id="T_3398e_row5_col11" class="data row5 col11" >159</td>
      <td id="T_3398e_row5_col12" class="data row5 col12" >144</td>
      <td id="T_3398e_row5_col13" class="data row5 col13" >106</td>
      <td id="T_3398e_row5_col14" class="data row5 col14" >47</td>
      <td id="T_3398e_row5_col15" class="data row5 col15" >48</td>
      <td id="T_3398e_row5_col16" class="data row5 col16" >60</td>
      <td id="T_3398e_row5_col17" class="data row5 col17" >66</td>
      <td id="T_3398e_row5_col18" class="data row5 col18" >68</td>
      <td id="T_3398e_row5_col19" class="data row5 col19" >61</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_3398e_row6_col0" class="data row6 col0" >148</td>
      <td id="T_3398e_row6_col1" class="data row6 col1" >122</td>
      <td id="T_3398e_row6_col2" class="data row6 col2" >59</td>
      <td id="T_3398e_row6_col3" class="data row6 col3" >59</td>
      <td id="T_3398e_row6_col4" class="data row6 col4" >39</td>
      <td id="T_3398e_row6_col5" class="data row6 col5" >74</td>
      <td id="T_3398e_row6_col6" class="data row6 col6" >85</td>
      <td id="T_3398e_row6_col7" class="data row6 col7" >45</td>
      <td id="T_3398e_row6_col8" class="data row6 col8" >50</td>
      <td id="T_3398e_row6_col9" class="data row6 col9" >65</td>
      <td id="T_3398e_row6_col10" class="data row6 col10" >135</td>
      <td id="T_3398e_row6_col11" class="data row6 col11" >129</td>
      <td id="T_3398e_row6_col12" class="data row6 col12" >115</td>
      <td id="T_3398e_row6_col13" class="data row6 col13" >72</td>
      <td id="T_3398e_row6_col14" class="data row6 col14" >51</td>
      <td id="T_3398e_row6_col15" class="data row6 col15" >58</td>
      <td id="T_3398e_row6_col16" class="data row6 col16" >68</td>
      <td id="T_3398e_row6_col17" class="data row6 col17" >79</td>
      <td id="T_3398e_row6_col18" class="data row6 col18" >82</td>
      <td id="T_3398e_row6_col19" class="data row6 col19" >84</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_3398e_row7_col0" class="data row7 col0" >139</td>
      <td id="T_3398e_row7_col1" class="data row7 col1" >73</td>
      <td id="T_3398e_row7_col2" class="data row7 col2" >64</td>
      <td id="T_3398e_row7_col3" class="data row7 col3" >41</td>
      <td id="T_3398e_row7_col4" class="data row7 col4" >62</td>
      <td id="T_3398e_row7_col5" class="data row7 col5" >83</td>
      <td id="T_3398e_row7_col6" class="data row7 col6" >44</td>
      <td id="T_3398e_row7_col7" class="data row7 col7" >50</td>
      <td id="T_3398e_row7_col8" class="data row7 col8" >60</td>
      <td id="T_3398e_row7_col9" class="data row7 col9" >105</td>
      <td id="T_3398e_row7_col10" class="data row7 col10" >162</td>
      <td id="T_3398e_row7_col11" class="data row7 col11" >134</td>
      <td id="T_3398e_row7_col12" class="data row7 col12" >115</td>
      <td id="T_3398e_row7_col13" class="data row7 col13" >64</td>
      <td id="T_3398e_row7_col14" class="data row7 col14" >63</td>
      <td id="T_3398e_row7_col15" class="data row7 col15" >72</td>
      <td id="T_3398e_row7_col16" class="data row7 col16" >83</td>
      <td id="T_3398e_row7_col17" class="data row7 col17" >96</td>
      <td id="T_3398e_row7_col18" class="data row7 col18" >98</td>
      <td id="T_3398e_row7_col19" class="data row7 col19" >94</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_3398e_row8_col0" class="data row8 col0" >92</td>
      <td id="T_3398e_row8_col1" class="data row8 col1" >59</td>
      <td id="T_3398e_row8_col2" class="data row8 col2" >39</td>
      <td id="T_3398e_row8_col3" class="data row8 col3" >48</td>
      <td id="T_3398e_row8_col4" class="data row8 col4" >95</td>
      <td id="T_3398e_row8_col5" class="data row8 col5" >68</td>
      <td id="T_3398e_row8_col6" class="data row8 col6" >83</td>
      <td id="T_3398e_row8_col7" class="data row8 col7" >131</td>
      <td id="T_3398e_row8_col8" class="data row8 col8" >169</td>
      <td id="T_3398e_row8_col9" class="data row8 col9" >179</td>
      <td id="T_3398e_row8_col10" class="data row8 col10" >159</td>
      <td id="T_3398e_row8_col11" class="data row8 col11" >144</td>
      <td id="T_3398e_row8_col12" class="data row8 col12" >103</td>
      <td id="T_3398e_row8_col13" class="data row8 col13" >52</td>
      <td id="T_3398e_row8_col14" class="data row8 col14" >66</td>
      <td id="T_3398e_row8_col15" class="data row8 col15" >73</td>
      <td id="T_3398e_row8_col16" class="data row8 col16" >79</td>
      <td id="T_3398e_row8_col17" class="data row8 col17" >88</td>
      <td id="T_3398e_row8_col18" class="data row8 col18" >97</td>
      <td id="T_3398e_row8_col19" class="data row8 col19" >103</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_3398e_row9_col0" class="data row9 col0" >60</td>
      <td id="T_3398e_row9_col1" class="data row9 col1" >43</td>
      <td id="T_3398e_row9_col2" class="data row9 col2" >34</td>
      <td id="T_3398e_row9_col3" class="data row9 col3" >100</td>
      <td id="T_3398e_row9_col4" class="data row9 col4" >140</td>
      <td id="T_3398e_row9_col5" class="data row9 col5" >172</td>
      <td id="T_3398e_row9_col6" class="data row9 col6" >179</td>
      <td id="T_3398e_row9_col7" class="data row9 col7" >180</td>
      <td id="T_3398e_row9_col8" class="data row9 col8" >154</td>
      <td id="T_3398e_row9_col9" class="data row9 col9" >163</td>
      <td id="T_3398e_row9_col10" class="data row9 col10" >143</td>
      <td id="T_3398e_row9_col11" class="data row9 col11" >128</td>
      <td id="T_3398e_row9_col12" class="data row9 col12" >77</td>
      <td id="T_3398e_row9_col13" class="data row9 col13" >48</td>
      <td id="T_3398e_row9_col14" class="data row9 col14" >59</td>
      <td id="T_3398e_row9_col15" class="data row9 col15" >63</td>
      <td id="T_3398e_row9_col16" class="data row9 col16" >56</td>
      <td id="T_3398e_row9_col17" class="data row9 col17" >46</td>
      <td id="T_3398e_row9_col18" class="data row9 col18" >45</td>
      <td id="T_3398e_row9_col19" class="data row9 col19" >54</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_3398e_row10_col0" class="data row10 col0" >46</td>
      <td id="T_3398e_row10_col1" class="data row10 col1" >28</td>
      <td id="T_3398e_row10_col2" class="data row10 col2" >86</td>
      <td id="T_3398e_row10_col3" class="data row10 col3" >96</td>
      <td id="T_3398e_row10_col4" class="data row10 col4" >156</td>
      <td id="T_3398e_row10_col5" class="data row10 col5" >152</td>
      <td id="T_3398e_row10_col6" class="data row10 col6" >139</td>
      <td id="T_3398e_row10_col7" class="data row10 col7" >135</td>
      <td id="T_3398e_row10_col8" class="data row10 col8" >146</td>
      <td id="T_3398e_row10_col9" class="data row10 col9" >138</td>
      <td id="T_3398e_row10_col10" class="data row10 col10" >120</td>
      <td id="T_3398e_row10_col11" class="data row10 col11" >107</td>
      <td id="T_3398e_row10_col12" class="data row10 col12" >116</td>
      <td id="T_3398e_row10_col13" class="data row10 col13" >179</td>
      <td id="T_3398e_row10_col14" class="data row10 col14" >166</td>
      <td id="T_3398e_row10_col15" class="data row10 col15" >96</td>
      <td id="T_3398e_row10_col16" class="data row10 col16" >40</td>
      <td id="T_3398e_row10_col17" class="data row10 col17" >31</td>
      <td id="T_3398e_row10_col18" class="data row10 col18" >35</td>
      <td id="T_3398e_row10_col19" class="data row10 col19" >44</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_3398e_row11_col0" class="data row11 col0" >25</td>
      <td id="T_3398e_row11_col1" class="data row11 col1" >68</td>
      <td id="T_3398e_row11_col2" class="data row11 col2" >95</td>
      <td id="T_3398e_row11_col3" class="data row11 col3" >81</td>
      <td id="T_3398e_row11_col4" class="data row11 col4" >156</td>
      <td id="T_3398e_row11_col5" class="data row11 col5" >142</td>
      <td id="T_3398e_row11_col6" class="data row11 col6" >133</td>
      <td id="T_3398e_row11_col7" class="data row11 col7" >122</td>
      <td id="T_3398e_row11_col8" class="data row11 col8" >137</td>
      <td id="T_3398e_row11_col9" class="data row11 col9" >119</td>
      <td id="T_3398e_row11_col10" class="data row11 col10" >104</td>
      <td id="T_3398e_row11_col11" class="data row11 col11" >95</td>
      <td id="T_3398e_row11_col12" class="data row11 col12" >128</td>
      <td id="T_3398e_row11_col13" class="data row11 col13" >159</td>
      <td id="T_3398e_row11_col14" class="data row11 col14" >155</td>
      <td id="T_3398e_row11_col15" class="data row11 col15" >140</td>
      <td id="T_3398e_row11_col16" class="data row11 col16" >83</td>
      <td id="T_3398e_row11_col17" class="data row11 col17" >69</td>
      <td id="T_3398e_row11_col18" class="data row11 col18" >78</td>
      <td id="T_3398e_row11_col19" class="data row11 col19" >85</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_3398e_row12_col0" class="data row12 col0" >43</td>
      <td id="T_3398e_row12_col1" class="data row12 col1" >115</td>
      <td id="T_3398e_row12_col2" class="data row12 col2" >52</td>
      <td id="T_3398e_row12_col3" class="data row12 col3" >107</td>
      <td id="T_3398e_row12_col4" class="data row12 col4" >148</td>
      <td id="T_3398e_row12_col5" class="data row12 col5" >136</td>
      <td id="T_3398e_row12_col6" class="data row12 col6" >123</td>
      <td id="T_3398e_row12_col7" class="data row12 col7" >108</td>
      <td id="T_3398e_row12_col8" class="data row12 col8" >141</td>
      <td id="T_3398e_row12_col9" class="data row12 col9" >120</td>
      <td id="T_3398e_row12_col10" class="data row12 col10" >105</td>
      <td id="T_3398e_row12_col11" class="data row12 col11" >82</td>
      <td id="T_3398e_row12_col12" class="data row12 col12" >110</td>
      <td id="T_3398e_row12_col13" class="data row12 col13" >133</td>
      <td id="T_3398e_row12_col14" class="data row12 col14" >124</td>
      <td id="T_3398e_row12_col15" class="data row12 col15" >126</td>
      <td id="T_3398e_row12_col16" class="data row12 col16" >162</td>
      <td id="T_3398e_row12_col17" class="data row12 col17" >126</td>
      <td id="T_3398e_row12_col18" class="data row12 col18" >112</td>
      <td id="T_3398e_row12_col19" class="data row12 col19" >123</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_3398e_row13_col0" class="data row13 col0" >104</td>
      <td id="T_3398e_row13_col1" class="data row13 col1" >75</td>
      <td id="T_3398e_row13_col2" class="data row13 col2" >67</td>
      <td id="T_3398e_row13_col3" class="data row13 col3" >145</td>
      <td id="T_3398e_row13_col4" class="data row13 col4" >140</td>
      <td id="T_3398e_row13_col5" class="data row13 col5" >136</td>
      <td id="T_3398e_row13_col6" class="data row13 col6" >117</td>
      <td id="T_3398e_row13_col7" class="data row13 col7" >105</td>
      <td id="T_3398e_row13_col8" class="data row13 col8" >134</td>
      <td id="T_3398e_row13_col9" class="data row13 col9" >115</td>
      <td id="T_3398e_row13_col10" class="data row13 col10" >100</td>
      <td id="T_3398e_row13_col11" class="data row13 col11" >92</td>
      <td id="T_3398e_row13_col12" class="data row13 col12" >115</td>
      <td id="T_3398e_row13_col13" class="data row13 col13" >123</td>
      <td id="T_3398e_row13_col14" class="data row13 col14" >119</td>
      <td id="T_3398e_row13_col15" class="data row13 col15" >108</td>
      <td id="T_3398e_row13_col16" class="data row13 col16" >140</td>
      <td id="T_3398e_row13_col17" class="data row13 col17" >135</td>
      <td id="T_3398e_row13_col18" class="data row13 col18" >90</td>
      <td id="T_3398e_row13_col19" class="data row13 col19" >95</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_3398e_row14_col0" class="data row14 col0" >111</td>
      <td id="T_3398e_row14_col1" class="data row14 col1" >42</td>
      <td id="T_3398e_row14_col2" class="data row14 col2" >70</td>
      <td id="T_3398e_row14_col3" class="data row14 col3" >145</td>
      <td id="T_3398e_row14_col4" class="data row14 col4" >144</td>
      <td id="T_3398e_row14_col5" class="data row14 col5" >134</td>
      <td id="T_3398e_row14_col6" class="data row14 col6" >124</td>
      <td id="T_3398e_row14_col7" class="data row14 col7" >127</td>
      <td id="T_3398e_row14_col8" class="data row14 col8" >132</td>
      <td id="T_3398e_row14_col9" class="data row14 col9" >109</td>
      <td id="T_3398e_row14_col10" class="data row14 col10" >96</td>
      <td id="T_3398e_row14_col11" class="data row14 col11" >113</td>
      <td id="T_3398e_row14_col12" class="data row14 col12" >111</td>
      <td id="T_3398e_row14_col13" class="data row14 col13" >116</td>
      <td id="T_3398e_row14_col14" class="data row14 col14" >112</td>
      <td id="T_3398e_row14_col15" class="data row14 col15" >105</td>
      <td id="T_3398e_row14_col16" class="data row14 col16" >132</td>
      <td id="T_3398e_row14_col17" class="data row14 col17" >125</td>
      <td id="T_3398e_row14_col18" class="data row14 col18" >68</td>
      <td id="T_3398e_row14_col19" class="data row14 col19" >68</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_3398e_row15_col0" class="data row15 col0" >59</td>
      <td id="T_3398e_row15_col1" class="data row15 col1" >43</td>
      <td id="T_3398e_row15_col2" class="data row15 col2" >64</td>
      <td id="T_3398e_row15_col3" class="data row15 col3" >135</td>
      <td id="T_3398e_row15_col4" class="data row15 col4" >140</td>
      <td id="T_3398e_row15_col5" class="data row15 col5" >132</td>
      <td id="T_3398e_row15_col6" class="data row15 col6" >131</td>
      <td id="T_3398e_row15_col7" class="data row15 col7" >137</td>
      <td id="T_3398e_row15_col8" class="data row15 col8" >132</td>
      <td id="T_3398e_row15_col9" class="data row15 col9" >109</td>
      <td id="T_3398e_row15_col10" class="data row15 col10" >97</td>
      <td id="T_3398e_row15_col11" class="data row15 col11" >110</td>
      <td id="T_3398e_row15_col12" class="data row15 col12" >118</td>
      <td id="T_3398e_row15_col13" class="data row15 col13" >120</td>
      <td id="T_3398e_row15_col14" class="data row15 col14" >112</td>
      <td id="T_3398e_row15_col15" class="data row15 col15" >113</td>
      <td id="T_3398e_row15_col16" class="data row15 col16" >130</td>
      <td id="T_3398e_row15_col17" class="data row15 col17" >116</td>
      <td id="T_3398e_row15_col18" class="data row15 col18" >58</td>
      <td id="T_3398e_row15_col19" class="data row15 col19" >52</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_3398e_row16_col0" class="data row16 col0" >36</td>
      <td id="T_3398e_row16_col1" class="data row16 col1" >37</td>
      <td id="T_3398e_row16_col2" class="data row16 col2" >58</td>
      <td id="T_3398e_row16_col3" class="data row16 col3" >118</td>
      <td id="T_3398e_row16_col4" class="data row16 col4" >121</td>
      <td id="T_3398e_row16_col5" class="data row16 col5" >115</td>
      <td id="T_3398e_row16_col6" class="data row16 col6" >117</td>
      <td id="T_3398e_row16_col7" class="data row16 col7" >132</td>
      <td id="T_3398e_row16_col8" class="data row16 col8" >131</td>
      <td id="T_3398e_row16_col9" class="data row16 col9" >113</td>
      <td id="T_3398e_row16_col10" class="data row16 col10" >103</td>
      <td id="T_3398e_row16_col11" class="data row16 col11" >111</td>
      <td id="T_3398e_row16_col12" class="data row16 col12" >128</td>
      <td id="T_3398e_row16_col13" class="data row16 col13" >121</td>
      <td id="T_3398e_row16_col14" class="data row16 col14" >114</td>
      <td id="T_3398e_row16_col15" class="data row16 col15" >126</td>
      <td id="T_3398e_row16_col16" class="data row16 col16" >125</td>
      <td id="T_3398e_row16_col17" class="data row16 col17" >112</td>
      <td id="T_3398e_row16_col18" class="data row16 col18" >74</td>
      <td id="T_3398e_row16_col19" class="data row16 col19" >52</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_3398e_row17_col0" class="data row17 col0" >40</td>
      <td id="T_3398e_row17_col1" class="data row17 col1" >35</td>
      <td id="T_3398e_row17_col2" class="data row17 col2" >47</td>
      <td id="T_3398e_row17_col3" class="data row17 col3" >96</td>
      <td id="T_3398e_row17_col4" class="data row17 col4" >87</td>
      <td id="T_3398e_row17_col5" class="data row17 col5" >86</td>
      <td id="T_3398e_row17_col6" class="data row17 col6" >98</td>
      <td id="T_3398e_row17_col7" class="data row17 col7" >121</td>
      <td id="T_3398e_row17_col8" class="data row17 col8" >117</td>
      <td id="T_3398e_row17_col9" class="data row17 col9" >105</td>
      <td id="T_3398e_row17_col10" class="data row17 col10" >101</td>
      <td id="T_3398e_row17_col11" class="data row17 col11" >105</td>
      <td id="T_3398e_row17_col12" class="data row17 col12" >125</td>
      <td id="T_3398e_row17_col13" class="data row17 col13" >116</td>
      <td id="T_3398e_row17_col14" class="data row17 col14" >106</td>
      <td id="T_3398e_row17_col15" class="data row17 col15" >115</td>
      <td id="T_3398e_row17_col16" class="data row17 col16" >117</td>
      <td id="T_3398e_row17_col17" class="data row17 col17" >101</td>
      <td id="T_3398e_row17_col18" class="data row17 col18" >62</td>
      <td id="T_3398e_row17_col19" class="data row17 col19" >42</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_3398e_row18_col0" class="data row18 col0" >37</td>
      <td id="T_3398e_row18_col1" class="data row18 col1" >34</td>
      <td id="T_3398e_row18_col2" class="data row18 col2" >37</td>
      <td id="T_3398e_row18_col3" class="data row18 col3" >85</td>
      <td id="T_3398e_row18_col4" class="data row18 col4" >82</td>
      <td id="T_3398e_row18_col5" class="data row18 col5" >79</td>
      <td id="T_3398e_row18_col6" class="data row18 col6" >82</td>
      <td id="T_3398e_row18_col7" class="data row18 col7" >88</td>
      <td id="T_3398e_row18_col8" class="data row18 col8" >78</td>
      <td id="T_3398e_row18_col9" class="data row18 col9" >75</td>
      <td id="T_3398e_row18_col10" class="data row18 col10" >73</td>
      <td id="T_3398e_row18_col11" class="data row18 col11" >75</td>
      <td id="T_3398e_row18_col12" class="data row18 col12" >83</td>
      <td id="T_3398e_row18_col13" class="data row18 col13" >79</td>
      <td id="T_3398e_row18_col14" class="data row18 col14" >72</td>
      <td id="T_3398e_row18_col15" class="data row18 col15" >76</td>
      <td id="T_3398e_row18_col16" class="data row18 col16" >82</td>
      <td id="T_3398e_row18_col17" class="data row18 col17" >81</td>
      <td id="T_3398e_row18_col18" class="data row18 col18" >43</td>
      <td id="T_3398e_row18_col19" class="data row18 col19" >31</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_3398e_row19_col0" class="data row19 col0" >30</td>
      <td id="T_3398e_row19_col1" class="data row19 col1" >27</td>
      <td id="T_3398e_row19_col2" class="data row19 col2" >33</td>
      <td id="T_3398e_row19_col3" class="data row19 col3" >70</td>
      <td id="T_3398e_row19_col4" class="data row19 col4" >88</td>
      <td id="T_3398e_row19_col5" class="data row19 col5" >79</td>
      <td id="T_3398e_row19_col6" class="data row19 col6" >71</td>
      <td id="T_3398e_row19_col7" class="data row19 col7" >72</td>
      <td id="T_3398e_row19_col8" class="data row19 col8" >64</td>
      <td id="T_3398e_row19_col9" class="data row19 col9" >61</td>
      <td id="T_3398e_row19_col10" class="data row19 col10" >63</td>
      <td id="T_3398e_row19_col11" class="data row19 col11" >57</td>
      <td id="T_3398e_row19_col12" class="data row19 col12" >59</td>
      <td id="T_3398e_row19_col13" class="data row19 col13" >59</td>
      <td id="T_3398e_row19_col14" class="data row19 col14" >56</td>
      <td id="T_3398e_row19_col15" class="data row19 col15" >57</td>
      <td id="T_3398e_row19_col16" class="data row19 col16" >61</td>
      <td id="T_3398e_row19_col17" class="data row19 col17" >74</td>
      <td id="T_3398e_row19_col18" class="data row19 col18" >47</td>
      <td id="T_3398e_row19_col19" class="data row19 col19" >29</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_3398e_row20_col0" class="data row20 col0" >26</td>
      <td id="T_3398e_row20_col1" class="data row20 col1" >19</td>
      <td id="T_3398e_row20_col2" class="data row20 col2" >40</td>
      <td id="T_3398e_row20_col3" class="data row20 col3" >73</td>
      <td id="T_3398e_row20_col4" class="data row20 col4" >93</td>
      <td id="T_3398e_row20_col5" class="data row20 col5" >81</td>
      <td id="T_3398e_row20_col6" class="data row20 col6" >76</td>
      <td id="T_3398e_row20_col7" class="data row20 col7" >72</td>
      <td id="T_3398e_row20_col8" class="data row20 col8" >63</td>
      <td id="T_3398e_row20_col9" class="data row20 col9" >61</td>
      <td id="T_3398e_row20_col10" class="data row20 col10" >66</td>
      <td id="T_3398e_row20_col11" class="data row20 col11" >59</td>
      <td id="T_3398e_row20_col12" class="data row20 col12" >59</td>
      <td id="T_3398e_row20_col13" class="data row20 col13" >61</td>
      <td id="T_3398e_row20_col14" class="data row20 col14" >64</td>
      <td id="T_3398e_row20_col15" class="data row20 col15" >67</td>
      <td id="T_3398e_row20_col16" class="data row20 col16" >68</td>
      <td id="T_3398e_row20_col17" class="data row20 col17" >62</td>
      <td id="T_3398e_row20_col18" class="data row20 col18" >43</td>
      <td id="T_3398e_row20_col19" class="data row20 col19" >38</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_3398e_row21_col0" class="data row21 col0" >25</td>
      <td id="T_3398e_row21_col1" class="data row21 col1" >35</td>
      <td id="T_3398e_row21_col2" class="data row21 col2" >61</td>
      <td id="T_3398e_row21_col3" class="data row21 col3" >41</td>
      <td id="T_3398e_row21_col4" class="data row21 col4" >101</td>
      <td id="T_3398e_row21_col5" class="data row21 col5" >94</td>
      <td id="T_3398e_row21_col6" class="data row21 col6" >82</td>
      <td id="T_3398e_row21_col7" class="data row21 col7" >74</td>
      <td id="T_3398e_row21_col8" class="data row21 col8" >66</td>
      <td id="T_3398e_row21_col9" class="data row21 col9" >68</td>
      <td id="T_3398e_row21_col10" class="data row21 col10" >66</td>
      <td id="T_3398e_row21_col11" class="data row21 col11" >65</td>
      <td id="T_3398e_row21_col12" class="data row21 col12" >64</td>
      <td id="T_3398e_row21_col13" class="data row21 col13" >65</td>
      <td id="T_3398e_row21_col14" class="data row21 col14" >70</td>
      <td id="T_3398e_row21_col15" class="data row21 col15" >68</td>
      <td id="T_3398e_row21_col16" class="data row21 col16" >64</td>
      <td id="T_3398e_row21_col17" class="data row21 col17" >39</td>
      <td id="T_3398e_row21_col18" class="data row21 col18" >34</td>
      <td id="T_3398e_row21_col19" class="data row21 col19" >41</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_3398e_row22_col0" class="data row22 col0" >33</td>
      <td id="T_3398e_row22_col1" class="data row22 col1" >66</td>
      <td id="T_3398e_row22_col2" class="data row22 col2" >33</td>
      <td id="T_3398e_row22_col3" class="data row22 col3" >52</td>
      <td id="T_3398e_row22_col4" class="data row22 col4" >111</td>
      <td id="T_3398e_row22_col5" class="data row22 col5" >98</td>
      <td id="T_3398e_row22_col6" class="data row22 col6" >86</td>
      <td id="T_3398e_row22_col7" class="data row22 col7" >78</td>
      <td id="T_3398e_row22_col8" class="data row22 col8" >71</td>
      <td id="T_3398e_row22_col9" class="data row22 col9" >69</td>
      <td id="T_3398e_row22_col10" class="data row22 col10" >66</td>
      <td id="T_3398e_row22_col11" class="data row22 col11" >67</td>
      <td id="T_3398e_row22_col12" class="data row22 col12" >67</td>
      <td id="T_3398e_row22_col13" class="data row22 col13" >68</td>
      <td id="T_3398e_row22_col14" class="data row22 col14" >68</td>
      <td id="T_3398e_row22_col15" class="data row22 col15" >63</td>
      <td id="T_3398e_row22_col16" class="data row22 col16" >40</td>
      <td id="T_3398e_row22_col17" class="data row22 col17" >28</td>
      <td id="T_3398e_row22_col18" class="data row22 col18" >37</td>
      <td id="T_3398e_row22_col19" class="data row22 col19" >34</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_3398e_row23_col0" class="data row23 col0" >108</td>
      <td id="T_3398e_row23_col1" class="data row23 col1" >108</td>
      <td id="T_3398e_row23_col2" class="data row23 col2" >28</td>
      <td id="T_3398e_row23_col3" class="data row23 col3" >51</td>
      <td id="T_3398e_row23_col4" class="data row23 col4" >99</td>
      <td id="T_3398e_row23_col5" class="data row23 col5" >103</td>
      <td id="T_3398e_row23_col6" class="data row23 col6" >88</td>
      <td id="T_3398e_row23_col7" class="data row23 col7" >76</td>
      <td id="T_3398e_row23_col8" class="data row23 col8" >70</td>
      <td id="T_3398e_row23_col9" class="data row23 col9" >67</td>
      <td id="T_3398e_row23_col10" class="data row23 col10" >68</td>
      <td id="T_3398e_row23_col11" class="data row23 col11" >72</td>
      <td id="T_3398e_row23_col12" class="data row23 col12" >70</td>
      <td id="T_3398e_row23_col13" class="data row23 col13" >64</td>
      <td id="T_3398e_row23_col14" class="data row23 col14" >59</td>
      <td id="T_3398e_row23_col15" class="data row23 col15" >54</td>
      <td id="T_3398e_row23_col16" class="data row23 col16" >25</td>
      <td id="T_3398e_row23_col17" class="data row23 col17" >17</td>
      <td id="T_3398e_row23_col18" class="data row23 col18" >33</td>
      <td id="T_3398e_row23_col19" class="data row23 col19" >36</td>
    </tr>
    <tr>
      <th id="T_3398e_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_3398e_row24_col0" class="data row24 col0" >103</td>
      <td id="T_3398e_row24_col1" class="data row24 col1" >97</td>
      <td id="T_3398e_row24_col2" class="data row24 col2" >60</td>
      <td id="T_3398e_row24_col3" class="data row24 col3" >61</td>
      <td id="T_3398e_row24_col4" class="data row24 col4" >84</td>
      <td id="T_3398e_row24_col5" class="data row24 col5" >89</td>
      <td id="T_3398e_row24_col6" class="data row24 col6" >85</td>
      <td id="T_3398e_row24_col7" class="data row24 col7" >82</td>
      <td id="T_3398e_row24_col8" class="data row24 col8" >79</td>
      <td id="T_3398e_row24_col9" class="data row24 col9" >80</td>
      <td id="T_3398e_row24_col10" class="data row24 col10" >83</td>
      <td id="T_3398e_row24_col11" class="data row24 col11" >88</td>
      <td id="T_3398e_row24_col12" class="data row24 col12" >99</td>
      <td id="T_3398e_row24_col13" class="data row24 col13" >101</td>
      <td id="T_3398e_row24_col14" class="data row24 col14" >94</td>
      <td id="T_3398e_row24_col15" class="data row24 col15" >98</td>
      <td id="T_3398e_row24_col16" class="data row24 col16" >87</td>
      <td id="T_3398e_row24_col17" class="data row24 col17" >75</td>
      <td id="T_3398e_row24_col18" class="data row24 col18" >85</td>
      <td id="T_3398e_row24_col19" class="data row24 col19" >90</td>
    </tr>
  </tbody>
</table>





```python
df_green = pd.DataFrame(im_tens[87:112,99:119,1])
df_green.style.set_properties(**{'font-size':'6pt'}).background_gradient('Greens')
```




<style type="text/css">
#T_54717_row0_col0, #T_54717_row0_col4, #T_54717_row0_col5, #T_54717_row0_col6, #T_54717_row0_col7, #T_54717_row0_col8, #T_54717_row1_col2, #T_54717_row1_col3, #T_54717_row2_col1, #T_54717_row2_col10, #T_54717_row4_col11, #T_54717_row4_col12, #T_54717_row8_col9, #T_54717_row10_col13, #T_54717_row10_col14, #T_54717_row11_col15, #T_54717_row12_col16, #T_54717_row12_col19, #T_54717_row13_col17, #T_54717_row24_col18 {
  font-size: 6pt;
  background-color: #00441b;
  color: #f1f1f1;
}
#T_54717_row0_col1, #T_54717_row1_col5, #T_54717_row24_col19 {
  font-size: 6pt;
  background-color: #005924;
  color: #f1f1f1;
}
#T_54717_row0_col2, #T_54717_row2_col4 {
  font-size: 6pt;
  background-color: #006b2b;
  color: #f1f1f1;
}
#T_54717_row0_col3, #T_54717_row1_col4 {
  font-size: 6pt;
  background-color: #00481d;
  color: #f1f1f1;
}
#T_54717_row0_col9 {
  font-size: 6pt;
  background-color: #2f974e;
  color: #f1f1f1;
}
#T_54717_row0_col10, #T_54717_row8_col7, #T_54717_row16_col19, #T_54717_row17_col10, #T_54717_row23_col1, #T_54717_row24_col10 {
  font-size: 6pt;
  background-color: #b6e2af;
  color: #000000;
}
#T_54717_row0_col11, #T_54717_row3_col6, #T_54717_row20_col3 {
  font-size: 6pt;
  background-color: #e5f5e1;
  color: #000000;
}
#T_54717_row0_col12, #T_54717_row4_col8, #T_54717_row5_col19, #T_54717_row6_col12, #T_54717_row7_col12, #T_54717_row12_col1 {
  font-size: 6pt;
  background-color: #97d492;
  color: #000000;
}
#T_54717_row0_col13 {
  font-size: 6pt;
  background-color: #16803c;
  color: #f1f1f1;
}
#T_54717_row0_col14, #T_54717_row14_col17 {
  font-size: 6pt;
  background-color: #00692a;
  color: #f1f1f1;
}
#T_54717_row0_col15, #T_54717_row6_col10 {
  font-size: 6pt;
  background-color: #39a257;
  color: #f1f1f1;
}
#T_54717_row0_col16, #T_54717_row23_col0, #T_54717_row24_col0 {
  font-size: 6pt;
  background-color: #c2e7bb;
  color: #000000;
}
#T_54717_row0_col17, #T_54717_row9_col5 {
  font-size: 6pt;
  background-color: #88ce87;
  color: #000000;
}
#T_54717_row0_col18, #T_54717_row5_col1, #T_54717_row5_col12 {
  font-size: 6pt;
  background-color: #50b264;
  color: #f1f1f1;
}
#T_54717_row0_col19, #T_54717_row1_col7, #T_54717_row17_col16, #T_54717_row24_col16 {
  font-size: 6pt;
  background-color: #66bd6f;
  color: #f1f1f1;
}
#T_54717_row1_col0, #T_54717_row1_col6 {
  font-size: 6pt;
  background-color: #05712f;
  color: #f1f1f1;
}
#T_54717_row1_col1 {
  font-size: 6pt;
  background-color: #005a24;
  color: #f1f1f1;
}
#T_54717_row1_col8, #T_54717_row9_col3, #T_54717_row14_col10, #T_54717_row18_col16, #T_54717_row24_col11 {
  font-size: 6pt;
  background-color: #bae3b3;
  color: #000000;
}
#T_54717_row1_col9, #T_54717_row6_col13, #T_54717_row21_col15, #T_54717_row22_col1 {
  font-size: 6pt;
  background-color: #e2f4dd;
  color: #000000;
}
#T_54717_row1_col10, #T_54717_row1_col15, #T_54717_row1_col19, #T_54717_row4_col10, #T_54717_row24_col9 {
  font-size: 6pt;
  background-color: #c4e8bd;
  color: #000000;
}
#T_54717_row1_col11, #T_54717_row13_col16 {
  font-size: 6pt;
  background-color: #248c46;
  color: #f1f1f1;
}
#T_54717_row1_col12 {
  font-size: 6pt;
  background-color: #157f3b;
  color: #f1f1f1;
}
#T_54717_row1_col13, #T_54717_row11_col4, #T_54717_row13_col3, #T_54717_row14_col9, #T_54717_row15_col11, #T_54717_row16_col11 {
  font-size: 6pt;
  background-color: #a5db9f;
  color: #000000;
}
#T_54717_row1_col14, #T_54717_row3_col16, #T_54717_row4_col5, #T_54717_row5_col3, #T_54717_row8_col6, #T_54717_row22_col5, #T_54717_row24_col2, #T_54717_row24_col5 {
  font-size: 6pt;
  background-color: #e5f5e0;
  color: #000000;
}
#T_54717_row1_col16, #T_54717_row7_col16, #T_54717_row8_col16 {
  font-size: 6pt;
  background-color: #b1e0ab;
  color: #000000;
}
#T_54717_row1_col17 {
  font-size: 6pt;
  background-color: #84cc83;
  color: #000000;
}
#T_54717_row1_col18, #T_54717_row8_col12, #T_54717_row17_col14, #T_54717_row24_col12 {
  font-size: 6pt;
  background-color: #a7dba0;
  color: #000000;
}
#T_54717_row2_col0 {
  font-size: 6pt;
  background-color: #006729;
  color: #f1f1f1;
}
#T_54717_row2_col2 {
  font-size: 6pt;
  background-color: #005221;
  color: #f1f1f1;
}
#T_54717_row2_col3 {
  font-size: 6pt;
  background-color: #005321;
  color: #f1f1f1;
}
#T_54717_row2_col5, #T_54717_row9_col10 {
  font-size: 6pt;
  background-color: #1d8640;
  color: #f1f1f1;
}
#T_54717_row2_col6, #T_54717_row14_col8, #T_54717_row17_col18 {
  font-size: 6pt;
  background-color: #9bd696;
  color: #000000;
}
#T_54717_row2_col7, #T_54717_row7_col8, #T_54717_row12_col7, #T_54717_row18_col14 {
  font-size: 6pt;
  background-color: #e3f4de;
  color: #000000;
}
#T_54717_row2_col8, #T_54717_row3_col14, #T_54717_row4_col19, #T_54717_row18_col1, #T_54717_row18_col4, #T_54717_row18_col7, #T_54717_row22_col9, #T_54717_row23_col14 {
  font-size: 6pt;
  background-color: #ecf8e8;
  color: #000000;
}
#T_54717_row2_col9, #T_54717_row6_col17, #T_54717_row9_col8, #T_54717_row10_col12 {
  font-size: 6pt;
  background-color: #8dd08a;
  color: #000000;
}
#T_54717_row2_col11, #T_54717_row12_col14 {
  font-size: 6pt;
  background-color: #87cd86;
  color: #000000;
}
#T_54717_row2_col12, #T_54717_row5_col14, #T_54717_row20_col2, #T_54717_row21_col1, #T_54717_row21_col12, #T_54717_row22_col11 {
  font-size: 6pt;
  background-color: #f3faf0;
  color: #000000;
}
#T_54717_row2_col13, #T_54717_row6_col15, #T_54717_row7_col13, #T_54717_row8_col2, #T_54717_row16_col0, #T_54717_row17_col2, #T_54717_row18_col0, #T_54717_row19_col15, #T_54717_row21_col14, #T_54717_row22_col6, #T_54717_row23_col6 {
  font-size: 6pt;
  background-color: #ebf7e7;
  color: #000000;
}
#T_54717_row2_col14, #T_54717_row9_col18, #T_54717_row14_col6, #T_54717_row21_col16 {
  font-size: 6pt;
  background-color: #d0edca;
  color: #000000;
}
#T_54717_row2_col15, #T_54717_row7_col15, #T_54717_row8_col15, #T_54717_row13_col5, #T_54717_row14_col5 {
  font-size: 6pt;
  background-color: #c8e9c1;
  color: #000000;
}
#T_54717_row2_col16, #T_54717_row24_col8 {
  font-size: 6pt;
  background-color: #daf0d4;
  color: #000000;
}
#T_54717_row2_col17, #T_54717_row10_col2, #T_54717_row13_col4, #T_54717_row14_col4, #T_54717_row15_col3 {
  font-size: 6pt;
  background-color: #bee5b8;
  color: #000000;
}
#T_54717_row2_col18, #T_54717_row12_col5, #T_54717_row13_col1, #T_54717_row20_col18 {
  font-size: 6pt;
  background-color: #c7e9c0;
  color: #000000;
}
#T_54717_row2_col19, #T_54717_row5_col6 {
  font-size: 6pt;
  background-color: #d1edcb;
  color: #000000;
}
#T_54717_row3_col0 {
  font-size: 6pt;
  background-color: #17813d;
  color: #f1f1f1;
}
#T_54717_row3_col1, #T_54717_row8_col19 {
  font-size: 6pt;
  background-color: #00682a;
  color: #f1f1f1;
}
#T_54717_row3_col2 {
  font-size: 6pt;
  background-color: #005b25;
  color: #f1f1f1;
}
#T_54717_row3_col3 {
  font-size: 6pt;
  background-color: #006428;
  color: #f1f1f1;
}
#T_54717_row3_col4, #T_54717_row24_col17 {
  font-size: 6pt;
  background-color: #2d954d;
  color: #f1f1f1;
}
#T_54717_row3_col5 {
  font-size: 6pt;
  background-color: #b5e1ae;
  color: #000000;
}
#T_54717_row3_col7, #T_54717_row3_col18, #T_54717_row19_col14, #T_54717_row21_col7, #T_54717_row22_col2, #T_54717_row22_col3, #T_54717_row23_col3 {
  font-size: 6pt;
  background-color: #f5fbf2;
  color: #000000;
}
#T_54717_row3_col8, #T_54717_row15_col18, #T_54717_row16_col8 {
  font-size: 6pt;
  background-color: #a0d99b;
  color: #000000;
}
#T_54717_row3_col9 {
  font-size: 6pt;
  background-color: #208843;
  color: #f1f1f1;
}
#T_54717_row3_col10, #T_54717_row5_col7, #T_54717_row11_col11, #T_54717_row20_col16 {
  font-size: 6pt;
  background-color: #ccebc6;
  color: #000000;
}
#T_54717_row3_col11 {
  font-size: 6pt;
  background-color: #bce4b5;
  color: #000000;
}
#T_54717_row3_col12 {
  font-size: 6pt;
  background-color: #3ba458;
  color: #f1f1f1;
}
#T_54717_row3_col13, #T_54717_row14_col11, #T_54717_row14_col14, #T_54717_row16_col13 {
  font-size: 6pt;
  background-color: #9cd797;
  color: #000000;
}
#T_54717_row3_col15, #T_54717_row7_col3, #T_54717_row19_col0, #T_54717_row20_col14, #T_54717_row21_col9, #T_54717_row23_col11 {
  font-size: 6pt;
  background-color: #edf8ea;
  color: #000000;
}
#T_54717_row3_col17, #T_54717_row6_col9, #T_54717_row6_col16, #T_54717_row11_col6 {
  font-size: 6pt;
  background-color: #ceecc8;
  color: #000000;
}
#T_54717_row3_col19, #T_54717_row4_col6, #T_54717_row4_col15, #T_54717_row5_col5, #T_54717_row6_col4, #T_54717_row6_col7, #T_54717_row6_col14, #T_54717_row9_col13, #T_54717_row19_col2, #T_54717_row19_col8, #T_54717_row19_col9, #T_54717_row19_col10, #T_54717_row19_col11, #T_54717_row19_col12, #T_54717_row20_col1, #T_54717_row21_col0, #T_54717_row21_col3, #T_54717_row23_col16, #T_54717_row23_col17, #T_54717_row23_col18 {
  font-size: 6pt;
  background-color: #f7fcf5;
  color: #000000;
}
#T_54717_row4_col0, #T_54717_row4_col3 {
  font-size: 6pt;
  background-color: #76c578;
  color: #000000;
}
#T_54717_row4_col1 {
  font-size: 6pt;
  background-color: #68be70;
  color: #000000;
}
#T_54717_row4_col2, #T_54717_row5_col0 {
  font-size: 6pt;
  background-color: #63bc6e;
  color: #f1f1f1;
}
#T_54717_row4_col4, #T_54717_row10_col5, #T_54717_row14_col3, #T_54717_row16_col10 {
  font-size: 6pt;
  background-color: #afdfa8;
  color: #000000;
}
#T_54717_row4_col7, #T_54717_row16_col7 {
  font-size: 6pt;
  background-color: #c3e7bc;
  color: #000000;
}
#T_54717_row4_col9, #T_54717_row8_col3, #T_54717_row12_col0, #T_54717_row17_col0, #T_54717_row17_col6, #T_54717_row18_col9, #T_54717_row22_col17 {
  font-size: 6pt;
  background-color: #eaf7e6;
  color: #000000;
}
#T_54717_row4_col13 {
  font-size: 6pt;
  background-color: #3aa357;
  color: #f1f1f1;
}
#T_54717_row4_col14, #T_54717_row5_col4, #T_54717_row8_col14, #T_54717_row16_col1, #T_54717_row21_col5, #T_54717_row23_col4, #T_54717_row24_col4, #T_54717_row24_col7 {
  font-size: 6pt;
  background-color: #e8f6e3;
  color: #000000;
}
#T_54717_row4_col16, #T_54717_row10_col19, #T_54717_row14_col1, #T_54717_row15_col1, #T_54717_row18_col12, #T_54717_row18_col13, #T_54717_row23_col15 {
  font-size: 6pt;
  background-color: #e7f6e3;
  color: #000000;
}
#T_54717_row4_col17, #T_54717_row7_col1, #T_54717_row12_col3, #T_54717_row16_col3, #T_54717_row18_col15 {
  font-size: 6pt;
  background-color: #cbebc5;
  color: #000000;
}
#T_54717_row4_col18 {
  font-size: 6pt;
  background-color: #e1f3dc;
  color: #000000;
}
#T_54717_row5_col2 {
  font-size: 6pt;
  background-color: #a2d99c;
  color: #000000;
}
#T_54717_row5_col8, #T_54717_row5_col15, #T_54717_row8_col13, #T_54717_row10_col1, #T_54717_row10_col16, #T_54717_row18_col2, #T_54717_row18_col6, #T_54717_row19_col5, #T_54717_row20_col0, #T_54717_row21_col13, #T_54717_row23_col13 {
  font-size: 6pt;
  background-color: #f1faee;
  color: #000000;
}
#T_54717_row5_col9, #T_54717_row6_col3, #T_54717_row9_col15, #T_54717_row11_col3, #T_54717_row20_col15, #T_54717_row21_col4 {
  font-size: 6pt;
  background-color: #dff3da;
  color: #000000;
}
#T_54717_row5_col10 {
  font-size: 6pt;
  background-color: #3ca559;
  color: #f1f1f1;
}
#T_54717_row5_col11 {
  font-size: 6pt;
  background-color: #1f8742;
  color: #f1f1f1;
}
#T_54717_row5_col13, #T_54717_row11_col17, #T_54717_row17_col13 {
  font-size: 6pt;
  background-color: #b0dfaa;
  color: #000000;
}
#T_54717_row5_col16, #T_54717_row19_col16 {
  font-size: 6pt;
  background-color: #d5efcf;
  color: #000000;
}
#T_54717_row5_col17, #T_54717_row20_col17 {
  font-size: 6pt;
  background-color: #95d391;
  color: #000000;
}
#T_54717_row5_col18, #T_54717_row14_col18 {
  font-size: 6pt;
  background-color: #7ac77b;
  color: #000000;
}
#T_54717_row6_col0, #T_54717_row6_col19, #T_54717_row13_col15 {
  font-size: 6pt;
  background-color: #60ba6c;
  color: #f1f1f1;
}
#T_54717_row6_col1, #T_54717_row11_col12, #T_54717_row16_col12 {
  font-size: 6pt;
  background-color: #83cb82;
  color: #000000;
}
#T_54717_row6_col2, #T_54717_row12_col11, #T_54717_row16_col6, #T_54717_row22_col4 {
  font-size: 6pt;
  background-color: #dbf1d5;
  color: #000000;
}
#T_54717_row6_col5, #T_54717_row20_col19 {
  font-size: 6pt;
  background-color: #e0f3db;
  color: #000000;
}
#T_54717_row6_col6, #T_54717_row7_col5, #T_54717_row18_col18, #T_54717_row21_col17 {
  font-size: 6pt;
  background-color: #d4eece;
  color: #000000;
}
#T_54717_row6_col8, #T_54717_row7_col4, #T_54717_row7_col14, #T_54717_row8_col5, #T_54717_row17_col1, #T_54717_row18_col11, #T_54717_row18_col19, #T_54717_row22_col14, #T_54717_row24_col3 {
  font-size: 6pt;
  background-color: #e9f7e5;
  color: #000000;
}
#T_54717_row6_col11 {
  font-size: 6pt;
  background-color: #72c375;
  color: #000000;
}
#T_54717_row6_col18, #T_54717_row7_col11 {
  font-size: 6pt;
  background-color: #56b567;
  color: #f1f1f1;
}
#T_54717_row7_col0, #T_54717_row10_col10 {
  font-size: 6pt;
  background-color: #70c274;
  color: #000000;
}
#T_54717_row7_col2 {
  font-size: 6pt;
  background-color: #d8f0d2;
  color: #000000;
}
#T_54717_row7_col6, #T_54717_row19_col6, #T_54717_row19_col7, #T_54717_row20_col7, #T_54717_row20_col8, #T_54717_row20_col10, #T_54717_row20_col11, #T_54717_row20_col12, #T_54717_row21_col8, #T_54717_row21_col10, #T_54717_row21_col11, #T_54717_row22_col10, #T_54717_row23_col2 {
  font-size: 6pt;
  background-color: #f6fcf4;
  color: #000000;
}
#T_54717_row7_col7, #T_54717_row19_col13, #T_54717_row20_col6, #T_54717_row22_col0, #T_54717_row22_col7 {
  font-size: 6pt;
  background-color: #f4fbf1;
  color: #000000;
}
#T_54717_row7_col9, #T_54717_row11_col8 {
  font-size: 6pt;
  background-color: #91d28e;
  color: #000000;
}
#T_54717_row7_col10 {
  font-size: 6pt;
  background-color: #005622;
  color: #f1f1f1;
}
#T_54717_row7_col17, #T_54717_row8_col17, #T_54717_row11_col19 {
  font-size: 6pt;
  background-color: #53b466;
  color: #f1f1f1;
}
#T_54717_row7_col18, #T_54717_row11_col14, #T_54717_row16_col17, #T_54717_row24_col15 {
  font-size: 6pt;
  background-color: #1e8741;
  color: #f1f1f1;
}
#T_54717_row7_col19 {
  font-size: 6pt;
  background-color: #349d53;
  color: #f1f1f1;
}
#T_54717_row8_col0, #T_54717_row11_col5, #T_54717_row15_col7, #T_54717_row17_col9 {
  font-size: 6pt;
  background-color: #bbe4b4;
  color: #000000;
}
#T_54717_row8_col1, #T_54717_row12_col6 {
  font-size: 6pt;
  background-color: #d7efd1;
  color: #000000;
}
#T_54717_row8_col4, #T_54717_row15_col5 {
  font-size: 6pt;
  background-color: #caeac3;
  color: #000000;
}
#T_54717_row8_col8 {
  font-size: 6pt;
  background-color: #4db163;
  color: #f1f1f1;
}
#T_54717_row8_col10 {
  font-size: 6pt;
  background-color: #00471c;
  color: #f1f1f1;
}
#T_54717_row8_col11, #T_54717_row17_col17 {
  font-size: 6pt;
  background-color: #329b51;
  color: #f1f1f1;
}
#T_54717_row8_col18, #T_54717_row12_col18 {
  font-size: 6pt;
  background-color: #006c2c;
  color: #f1f1f1;
}
#T_54717_row9_col0, #T_54717_row9_col12, #T_54717_row9_col16, #T_54717_row11_col7, #T_54717_row18_col3, #T_54717_row22_col15 {
  font-size: 6pt;
  background-color: #dbf1d6;
  color: #000000;
}
#T_54717_row9_col1, #T_54717_row10_col0, #T_54717_row12_col2, #T_54717_row19_col3, #T_54717_row19_col4, #T_54717_row21_col2, #T_54717_row23_col5, #T_54717_row24_col6 {
  font-size: 6pt;
  background-color: #e7f6e2;
  color: #000000;
}
#T_54717_row9_col2, #T_54717_row22_col13 {
  font-size: 6pt;
  background-color: #eff9eb;
  color: #000000;
}
#T_54717_row9_col4, #T_54717_row11_col10, #T_54717_row13_col10, #T_54717_row14_col13 {
  font-size: 6pt;
  background-color: #b2e0ac;
  color: #000000;
}
#T_54717_row9_col6, #T_54717_row10_col8, #T_54717_row11_col18 {
  font-size: 6pt;
  background-color: #81ca81;
  color: #000000;
}
#T_54717_row9_col7, #T_54717_row11_col9, #T_54717_row12_col13, #T_54717_row24_col13 {
  font-size: 6pt;
  background-color: #8ace88;
  color: #000000;
}
#T_54717_row9_col9 {
  font-size: 6pt;
  background-color: #016e2d;
  color: #f1f1f1;
}
#T_54717_row9_col11, #T_54717_row24_col14 {
  font-size: 6pt;
  background-color: #6abf71;
  color: #000000;
}
#T_54717_row9_col14, #T_54717_row18_col5, #T_54717_row19_col19, #T_54717_row23_col9, #T_54717_row23_col19 {
  font-size: 6pt;
  background-color: #f0f9ed;
  color: #000000;
}
#T_54717_row9_col17, #T_54717_row10_col6 {
  font-size: 6pt;
  background-color: #c9eac2;
  color: #000000;
}
#T_54717_row9_col19, #T_54717_row12_col12 {
  font-size: 6pt;
  background-color: #acdea6;
  color: #000000;
}
#T_54717_row10_col3, #T_54717_row15_col6 {
  font-size: 6pt;
  background-color: #cbeac4;
  color: #000000;
}
#T_54717_row10_col4, #T_54717_row13_col12, #T_54717_row15_col9, #T_54717_row15_col12 {
  font-size: 6pt;
  background-color: #a9dca3;
  color: #000000;
}
#T_54717_row10_col7, #T_54717_row17_col7 {
  font-size: 6pt;
  background-color: #d2edcc;
  color: #000000;
}
#T_54717_row10_col9 {
  font-size: 6pt;
  background-color: #46ae60;
  color: #f1f1f1;
}
#T_54717_row10_col11 {
  font-size: 6pt;
  background-color: #aedea7;
  color: #000000;
}
#T_54717_row10_col15 {
  font-size: 6pt;
  background-color: #5bb86a;
  color: #f1f1f1;
}
#T_54717_row10_col17, #T_54717_row10_col18, #T_54717_row13_col6, #T_54717_row17_col19 {
  font-size: 6pt;
  background-color: #dcf2d7;
  color: #000000;
}
#T_54717_row11_col0, #T_54717_row20_col9, #T_54717_row23_col7 {
  font-size: 6pt;
  background-color: #f4fbf2;
  color: #000000;
}
#T_54717_row11_col1 {
  font-size: 6pt;
  background-color: #cfecc9;
  color: #000000;
}
#T_54717_row11_col2, #T_54717_row17_col11 {
  font-size: 6pt;
  background-color: #b4e1ad;
  color: #000000;
}
#T_54717_row11_col13 {
  font-size: 6pt;
  background-color: #319a50;
  color: #f1f1f1;
}
#T_54717_row11_col16, #T_54717_row12_col4, #T_54717_row19_col18 {
  font-size: 6pt;
  background-color: #b7e2b1;
  color: #000000;
}
#T_54717_row12_col8, #T_54717_row17_col12 {
  font-size: 6pt;
  background-color: #94d390;
  color: #000000;
}
#T_54717_row12_col9 {
  font-size: 6pt;
  background-color: #7fc97f;
  color: #000000;
}
#T_54717_row12_col10 {
  font-size: 6pt;
  background-color: #a4da9e;
  color: #000000;
}
#T_54717_row12_col15, #T_54717_row16_col15 {
  font-size: 6pt;
  background-color: #1a843f;
  color: #f1f1f1;
}
#T_54717_row12_col17 {
  font-size: 6pt;
  background-color: #006529;
  color: #f1f1f1;
}
#T_54717_row13_col0, #T_54717_row13_col13 {
  font-size: 6pt;
  background-color: #aadda4;
  color: #000000;
}
#T_54717_row13_col2 {
  font-size: 6pt;
  background-color: #def2d9;
  color: #000000;
}
#T_54717_row13_col7, #T_54717_row15_col2, #T_54717_row20_col4 {
  font-size: 6pt;
  background-color: #e4f5df;
  color: #000000;
}
#T_54717_row13_col8 {
  font-size: 6pt;
  background-color: #98d594;
  color: #000000;
}
#T_54717_row13_col9 {
  font-size: 6pt;
  background-color: #9fd899;
  color: #000000;
}
#T_54717_row13_col11, #T_54717_row14_col7 {
  font-size: 6pt;
  background-color: #c6e8bf;
  color: #000000;
}
#T_54717_row13_col14 {
  font-size: 6pt;
  background-color: #8ed08b;
  color: #000000;
}
#T_54717_row13_col18 {
  font-size: 6pt;
  background-color: #48ae60;
  color: #f1f1f1;
}
#T_54717_row13_col19, #T_54717_row14_col16 {
  font-size: 6pt;
  background-color: #38a156;
  color: #f1f1f1;
}
#T_54717_row14_col0, #T_54717_row14_col19, #T_54717_row15_col13, #T_54717_row16_col9 {
  font-size: 6pt;
  background-color: #a3da9d;
  color: #000000;
}
#T_54717_row14_col2, #T_54717_row15_col0, #T_54717_row16_col5 {
  font-size: 6pt;
  background-color: #ddf2d8;
  color: #000000;
}
#T_54717_row14_col12, #T_54717_row15_col10 {
  font-size: 6pt;
  background-color: #bde5b6;
  color: #000000;
}
#T_54717_row14_col15 {
  font-size: 6pt;
  background-color: #6bc072;
  color: #000000;
}
#T_54717_row15_col4, #T_54717_row17_col8 {
  font-size: 6pt;
  background-color: #c1e6ba;
  color: #000000;
}
#T_54717_row15_col8 {
  font-size: 6pt;
  background-color: #9ed798;
  color: #000000;
}
#T_54717_row15_col14 {
  font-size: 6pt;
  background-color: #99d595;
  color: #000000;
}
#T_54717_row15_col15 {
  font-size: 6pt;
  background-color: #4aaf61;
  color: #f1f1f1;
}
#T_54717_row15_col16 {
  font-size: 6pt;
  background-color: #3da65a;
  color: #f1f1f1;
}
#T_54717_row15_col17 {
  font-size: 6pt;
  background-color: #117b38;
  color: #f1f1f1;
}
#T_54717_row15_col19 {
  font-size: 6pt;
  background-color: #cdecc7;
  color: #000000;
}
#T_54717_row16_col2 {
  font-size: 6pt;
  background-color: #e6f5e1;
  color: #000000;
}
#T_54717_row16_col4, #T_54717_row17_col3 {
  font-size: 6pt;
  background-color: #d6efd0;
  color: #000000;
}
#T_54717_row16_col14 {
  font-size: 6pt;
  background-color: #92d28f;
  color: #000000;
}
#T_54717_row16_col16 {
  font-size: 6pt;
  background-color: #55b567;
  color: #f1f1f1;
}
#T_54717_row16_col18 {
  font-size: 6pt;
  background-color: #40aa5d;
  color: #f1f1f1;
}
#T_54717_row17_col4, #T_54717_row21_col18, #T_54717_row22_col16 {
  font-size: 6pt;
  background-color: #edf8e9;
  color: #000000;
}
#T_54717_row17_col5, #T_54717_row18_col8, #T_54717_row21_col6, #T_54717_row23_col12 {
  font-size: 6pt;
  background-color: #eff9ec;
  color: #000000;
}
#T_54717_row17_col15 {
  font-size: 6pt;
  background-color: #3fa95c;
  color: #f1f1f1;
}
#T_54717_row18_col10 {
  font-size: 6pt;
  background-color: #e8f6e4;
  color: #000000;
}
#T_54717_row18_col17 {
  font-size: 6pt;
  background-color: #62bb6d;
  color: #f1f1f1;
}
#T_54717_row19_col1, #T_54717_row22_col12 {
  font-size: 6pt;
  background-color: #f2faef;
  color: #000000;
}
#T_54717_row19_col17 {
  font-size: 6pt;
  background-color: #80ca80;
  color: #000000;
}
#T_54717_row20_col5, #T_54717_row22_col19 {
  font-size: 6pt;
  background-color: #eef8ea;
  color: #000000;
}
#T_54717_row20_col13, #T_54717_row22_col8, #T_54717_row23_col8, #T_54717_row23_col10 {
  font-size: 6pt;
  background-color: #f2faf0;
  color: #000000;
}
#T_54717_row21_col19 {
  font-size: 6pt;
  background-color: #d9f0d3;
  color: #000000;
}
#T_54717_row22_col18 {
  font-size: 6pt;
  background-color: #f0f9ec;
  color: #000000;
}
#T_54717_row24_col1 {
  font-size: 6pt;
  background-color: #c0e6b9;
  color: #000000;
}
</style>
<table id="T_54717">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_54717_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_54717_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_54717_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_54717_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_54717_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_54717_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_54717_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_54717_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_54717_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_54717_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_54717_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_54717_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_54717_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_54717_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_54717_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_54717_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_54717_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_54717_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_54717_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_54717_level0_col19" class="col_heading level0 col19" >19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_54717_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_54717_row0_col0" class="data row0 col0" >255</td>
      <td id="T_54717_row0_col1" class="data row0 col1" >230</td>
      <td id="T_54717_row0_col2" class="data row0 col2" >221</td>
      <td id="T_54717_row0_col3" class="data row0 col3" >241</td>
      <td id="T_54717_row0_col4" class="data row0 col4" >242</td>
      <td id="T_54717_row0_col5" class="data row0 col5" >240</td>
      <td id="T_54717_row0_col6" class="data row0 col6" >232</td>
      <td id="T_54717_row0_col7" class="data row0 col7" >219</td>
      <td id="T_54717_row0_col8" class="data row0 col8" >171</td>
      <td id="T_54717_row0_col9" class="data row0 col9" >97</td>
      <td id="T_54717_row0_col10" class="data row0 col10" >55</td>
      <td id="T_54717_row0_col11" class="data row0 col11" >41</td>
      <td id="T_54717_row0_col12" class="data row0 col12" >72</td>
      <td id="T_54717_row0_col13" class="data row0 col13" >117</td>
      <td id="T_54717_row0_col14" class="data row0 col14" >114</td>
      <td id="T_54717_row0_col15" class="data row0 col15" >70</td>
      <td id="T_54717_row0_col16" class="data row0 col16" >40</td>
      <td id="T_54717_row0_col17" class="data row0 col17" >39</td>
      <td id="T_54717_row0_col18" class="data row0 col18" >44</td>
      <td id="T_54717_row0_col19" class="data row0 col19" >43</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_54717_row1_col0" class="data row1 col0" >220</td>
      <td id="T_54717_row1_col1" class="data row1 col1" >229</td>
      <td id="T_54717_row1_col2" class="data row1 col2" >249</td>
      <td id="T_54717_row1_col3" class="data row1 col3" >244</td>
      <td id="T_54717_row1_col4" class="data row1 col4" >239</td>
      <td id="T_54717_row1_col5" class="data row1 col5" >227</td>
      <td id="T_54717_row1_col6" class="data row1 col6" >204</td>
      <td id="T_54717_row1_col7" class="data row1 col7" >133</td>
      <td id="T_54717_row1_col8" class="data row1 col8" >70</td>
      <td id="T_54717_row1_col9" class="data row1 col9" >41</td>
      <td id="T_54717_row1_col10" class="data row1 col10" >51</td>
      <td id="T_54717_row1_col11" class="data row1 col11" >106</td>
      <td id="T_54717_row1_col12" class="data row1 col12" >115</td>
      <td id="T_54717_row1_col13" class="data row1 col13" >67</td>
      <td id="T_54717_row1_col14" class="data row1 col14" >36</td>
      <td id="T_54717_row1_col15" class="data row1 col15" >41</td>
      <td id="T_54717_row1_col16" class="data row1 col16" >46</td>
      <td id="T_54717_row1_col17" class="data row1 col17" >40</td>
      <td id="T_54717_row1_col18" class="data row1 col18" >31</td>
      <td id="T_54717_row1_col19" class="data row1 col19" >26</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_54717_row2_col0" class="data row2 col0" >229</td>
      <td id="T_54717_row2_col1" class="data row2 col1" >245</td>
      <td id="T_54717_row2_col2" class="data row2 col2" >238</td>
      <td id="T_54717_row2_col3" class="data row2 col3" >233</td>
      <td id="T_54717_row2_col4" class="data row2 col4" >217</td>
      <td id="T_54717_row2_col5" class="data row2 col5" >193</td>
      <td id="T_54717_row2_col6" class="data row2 col6" >112</td>
      <td id="T_54717_row2_col7" class="data row2 col7" >59</td>
      <td id="T_54717_row2_col8" class="data row2 col8" >39</td>
      <td id="T_54717_row2_col9" class="data row2 col9" >70</td>
      <td id="T_54717_row2_col10" class="data row2 col10" >116</td>
      <td id="T_54717_row2_col11" class="data row2 col11" >75</td>
      <td id="T_54717_row2_col12" class="data row2 col12" >31</td>
      <td id="T_54717_row2_col13" class="data row2 col13" >35</td>
      <td id="T_54717_row2_col14" class="data row2 col14" >45</td>
      <td id="T_54717_row2_col15" class="data row2 col15" >40</td>
      <td id="T_54717_row2_col16" class="data row2 col16" >30</td>
      <td id="T_54717_row2_col17" class="data row2 col17" >25</td>
      <td id="T_54717_row2_col18" class="data row2 col18" >25</td>
      <td id="T_54717_row2_col19" class="data row2 col19" >23</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_54717_row3_col0" class="data row3 col0" >204</td>
      <td id="T_54717_row3_col1" class="data row3 col1" >219</td>
      <td id="T_54717_row3_col2" class="data row3 col2" >232</td>
      <td id="T_54717_row3_col3" class="data row3 col3" >222</td>
      <td id="T_54717_row3_col4" class="data row3 col4" >181</td>
      <td id="T_54717_row3_col5" class="data row3 col5" >97</td>
      <td id="T_54717_row3_col6" class="data row3 col6" >58</td>
      <td id="T_54717_row3_col7" class="data row3 col7" >37</td>
      <td id="T_54717_row3_col8" class="data row3 col8" >82</td>
      <td id="T_54717_row3_col9" class="data row3 col9" >103</td>
      <td id="T_54717_row3_col10" class="data row3 col10" >48</td>
      <td id="T_54717_row3_col11" class="data row3 col11" >58</td>
      <td id="T_54717_row3_col12" class="data row3 col12" >99</td>
      <td id="T_54717_row3_col13" class="data row3 col13" >70</td>
      <td id="T_54717_row3_col14" class="data row3 col14" >31</td>
      <td id="T_54717_row3_col15" class="data row3 col15" >27</td>
      <td id="T_54717_row3_col16" class="data row3 col16" >25</td>
      <td id="T_54717_row3_col17" class="data row3 col17" >20</td>
      <td id="T_54717_row3_col18" class="data row3 col18" >12</td>
      <td id="T_54717_row3_col19" class="data row3 col19" >10</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_54717_row4_col0" class="data row4 col0" >132</td>
      <td id="T_54717_row4_col1" class="data row4 col1" >134</td>
      <td id="T_54717_row4_col2" class="data row4 col2" >141</td>
      <td id="T_54717_row4_col3" class="data row4 col3" >130</td>
      <td id="T_54717_row4_col4" class="data row4 col4" >101</td>
      <td id="T_54717_row4_col5" class="data row4 col5" >59</td>
      <td id="T_54717_row4_col6" class="data row4 col6" >34</td>
      <td id="T_54717_row4_col7" class="data row4 col7" >83</td>
      <td id="T_54717_row4_col8" class="data row4 col8" >86</td>
      <td id="T_54717_row4_col9" class="data row4 col9" >36</td>
      <td id="T_54717_row4_col10" class="data row4 col10" >51</td>
      <td id="T_54717_row4_col11" class="data row4 col11" >133</td>
      <td id="T_54717_row4_col12" class="data row4 col12" >137</td>
      <td id="T_54717_row4_col13" class="data row4 col13" >101</td>
      <td id="T_54717_row4_col14" class="data row4 col14" >34</td>
      <td id="T_54717_row4_col15" class="data row4 col15" >22</td>
      <td id="T_54717_row4_col16" class="data row4 col16" >23</td>
      <td id="T_54717_row4_col17" class="data row4 col17" >21</td>
      <td id="T_54717_row4_col18" class="data row4 col18" >19</td>
      <td id="T_54717_row4_col19" class="data row4 col19" >15</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_54717_row5_col0" class="data row5 col0" >143</td>
      <td id="T_54717_row5_col1" class="data row5 col1" >148</td>
      <td id="T_54717_row5_col2" class="data row5 col2" >101</td>
      <td id="T_54717_row5_col3" class="data row5 col3" >48</td>
      <td id="T_54717_row5_col4" class="data row5 col4" >54</td>
      <td id="T_54717_row5_col5" class="data row5 col5" >33</td>
      <td id="T_54717_row5_col6" class="data row5 col6" >75</td>
      <td id="T_54717_row5_col7" class="data row5 col7" >76</td>
      <td id="T_54717_row5_col8" class="data row5 col8" >34</td>
      <td id="T_54717_row5_col9" class="data row5 col9" >42</td>
      <td id="T_54717_row5_col10" class="data row5 col10" >85</td>
      <td id="T_54717_row5_col11" class="data row5 col11" >108</td>
      <td id="T_54717_row5_col12" class="data row5 col12" >92</td>
      <td id="T_54717_row5_col13" class="data row5 col13" >63</td>
      <td id="T_54717_row5_col14" class="data row5 col14" >26</td>
      <td id="T_54717_row5_col15" class="data row5 col15" >25</td>
      <td id="T_54717_row5_col16" class="data row5 col16" >32</td>
      <td id="T_54717_row5_col17" class="data row5 col17" >36</td>
      <td id="T_54717_row5_col18" class="data row5 col18" >38</td>
      <td id="T_54717_row5_col19" class="data row5 col19" >35</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_54717_row6_col0" class="data row6 col0" >145</td>
      <td id="T_54717_row6_col1" class="data row6 col1" >118</td>
      <td id="T_54717_row6_col2" class="data row6 col2" >53</td>
      <td id="T_54717_row6_col3" class="data row6 col3" >53</td>
      <td id="T_54717_row6_col4" class="data row6 col4" >31</td>
      <td id="T_54717_row6_col5" class="data row6 col5" >63</td>
      <td id="T_54717_row6_col6" class="data row6 col6" >73</td>
      <td id="T_54717_row6_col7" class="data row6 col7" >34</td>
      <td id="T_54717_row6_col8" class="data row6 col8" >42</td>
      <td id="T_54717_row6_col9" class="data row6 col9" >49</td>
      <td id="T_54717_row6_col10" class="data row6 col10" >86</td>
      <td id="T_54717_row6_col11" class="data row6 col11" >81</td>
      <td id="T_54717_row6_col12" class="data row6 col12" >72</td>
      <td id="T_54717_row6_col13" class="data row6 col13" >41</td>
      <td id="T_54717_row6_col14" class="data row6 col14" >23</td>
      <td id="T_54717_row6_col15" class="data row6 col15" >28</td>
      <td id="T_54717_row6_col16" class="data row6 col16" >35</td>
      <td id="T_54717_row6_col17" class="data row6 col17" >38</td>
      <td id="T_54717_row6_col18" class="data row6 col18" >43</td>
      <td id="T_54717_row6_col19" class="data row6 col19" >44</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_54717_row7_col0" class="data row7 col0" >136</td>
      <td id="T_54717_row7_col1" class="data row7 col1" >65</td>
      <td id="T_54717_row7_col2" class="data row7 col2" >56</td>
      <td id="T_54717_row7_col3" class="data row7 col3" >34</td>
      <td id="T_54717_row7_col4" class="data row7 col4" >51</td>
      <td id="T_54717_row7_col5" class="data row7 col5" >74</td>
      <td id="T_54717_row7_col6" class="data row7 col6" >35</td>
      <td id="T_54717_row7_col7" class="data row7 col7" >39</td>
      <td id="T_54717_row7_col8" class="data row7 col8" >47</td>
      <td id="T_54717_row7_col9" class="data row7 col9" >69</td>
      <td id="T_54717_row7_col10" class="data row7 col10" >111</td>
      <td id="T_54717_row7_col11" class="data row7 col11" >88</td>
      <td id="T_54717_row7_col12" class="data row7 col12" >72</td>
      <td id="T_54717_row7_col13" class="data row7 col13" >35</td>
      <td id="T_54717_row7_col14" class="data row7 col14" >33</td>
      <td id="T_54717_row7_col15" class="data row7 col15" >40</td>
      <td id="T_54717_row7_col16" class="data row7 col16" >46</td>
      <td id="T_54717_row7_col17" class="data row7 col17" >51</td>
      <td id="T_54717_row7_col18" class="data row7 col18" >54</td>
      <td id="T_54717_row7_col19" class="data row7 col19" >52</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_54717_row8_col0" class="data row8 col0" >83</td>
      <td id="T_54717_row8_col1" class="data row8 col1" >54</td>
      <td id="T_54717_row8_col2" class="data row8 col2" >33</td>
      <td id="T_54717_row8_col3" class="data row8 col3" >40</td>
      <td id="T_54717_row8_col4" class="data row8 col4" >82</td>
      <td id="T_54717_row8_col5" class="data row8 col5" >54</td>
      <td id="T_54717_row8_col6" class="data row8 col6" >59</td>
      <td id="T_54717_row8_col7" class="data row8 col7" >91</td>
      <td id="T_54717_row8_col8" class="data row8 col8" >113</td>
      <td id="T_54717_row8_col9" class="data row8 col9" >127</td>
      <td id="T_54717_row8_col10" class="data row8 col10" >115</td>
      <td id="T_54717_row8_col11" class="data row8 col11" >100</td>
      <td id="T_54717_row8_col12" class="data row8 col12" >67</td>
      <td id="T_54717_row8_col13" class="data row8 col13" >30</td>
      <td id="T_54717_row8_col14" class="data row8 col14" >34</td>
      <td id="T_54717_row8_col15" class="data row8 col15" >40</td>
      <td id="T_54717_row8_col16" class="data row8 col16" >46</td>
      <td id="T_54717_row8_col17" class="data row8 col17" >51</td>
      <td id="T_54717_row8_col18" class="data row8 col18" >60</td>
      <td id="T_54717_row8_col19" class="data row8 col19" >65</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_54717_row9_col0" class="data row9 col0" >52</td>
      <td id="T_54717_row9_col1" class="data row9 col1" >37</td>
      <td id="T_54717_row9_col2" class="data row9 col2" >27</td>
      <td id="T_54717_row9_col3" class="data row9 col3" >85</td>
      <td id="T_54717_row9_col4" class="data row9 col4" >98</td>
      <td id="T_54717_row9_col5" class="data row9 col5" >125</td>
      <td id="T_54717_row9_col6" class="data row9 col6" >126</td>
      <td id="T_54717_row9_col7" class="data row9 col7" >115</td>
      <td id="T_54717_row9_col8" class="data row9 col8" >90</td>
      <td id="T_54717_row9_col9" class="data row9 col9" >114</td>
      <td id="T_54717_row9_col10" class="data row9 col10" >96</td>
      <td id="T_54717_row9_col11" class="data row9 col11" >83</td>
      <td id="T_54717_row9_col12" class="data row9 col12" >46</td>
      <td id="T_54717_row9_col13" class="data row9 col13" >25</td>
      <td id="T_54717_row9_col14" class="data row9 col14" >28</td>
      <td id="T_54717_row9_col15" class="data row9 col15" >33</td>
      <td id="T_54717_row9_col16" class="data row9 col16" >29</td>
      <td id="T_54717_row9_col17" class="data row9 col17" >22</td>
      <td id="T_54717_row9_col18" class="data row9 col18" >23</td>
      <td id="T_54717_row9_col19" class="data row9 col19" >31</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_54717_row10_col0" class="data row10 col0" >40</td>
      <td id="T_54717_row10_col1" class="data row10 col1" >21</td>
      <td id="T_54717_row10_col2" class="data row10 col2" >79</td>
      <td id="T_54717_row10_col3" class="data row10 col3" >72</td>
      <td id="T_54717_row10_col4" class="data row10 col4" >105</td>
      <td id="T_54717_row10_col5" class="data row10 col5" >101</td>
      <td id="T_54717_row10_col6" class="data row10 col6" >82</td>
      <td id="T_54717_row10_col7" class="data row10 col7" >72</td>
      <td id="T_54717_row10_col8" class="data row10 col8" >94</td>
      <td id="T_54717_row10_col9" class="data row10 col9" >88</td>
      <td id="T_54717_row10_col10" class="data row10 col10" >73</td>
      <td id="T_54717_row10_col11" class="data row10 col11" >63</td>
      <td id="T_54717_row10_col12" class="data row10 col12" >75</td>
      <td id="T_54717_row10_col13" class="data row10 col13" >141</td>
      <td id="T_54717_row10_col14" class="data row10 col14" >126</td>
      <td id="T_54717_row10_col15" class="data row10 col15" >63</td>
      <td id="T_54717_row10_col16" class="data row10 col16" >16</td>
      <td id="T_54717_row10_col17" class="data row10 col17" >15</td>
      <td id="T_54717_row10_col18" class="data row10 col18" >20</td>
      <td id="T_54717_row10_col19" class="data row10 col19" >17</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_54717_row11_col0" class="data row11 col0" >17</td>
      <td id="T_54717_row11_col1" class="data row11 col1" >61</td>
      <td id="T_54717_row11_col2" class="data row11 col2" >87</td>
      <td id="T_54717_row11_col3" class="data row11 col3" >53</td>
      <td id="T_54717_row11_col4" class="data row11 col4" >107</td>
      <td id="T_54717_row11_col5" class="data row11 col5" >93</td>
      <td id="T_54717_row11_col6" class="data row11 col6" >78</td>
      <td id="T_54717_row11_col7" class="data row11 col7" >65</td>
      <td id="T_54717_row11_col8" class="data row11 col8" >88</td>
      <td id="T_54717_row11_col9" class="data row11 col9" >71</td>
      <td id="T_54717_row11_col10" class="data row11 col10" >56</td>
      <td id="T_54717_row11_col11" class="data row11 col11" >52</td>
      <td id="T_54717_row11_col12" class="data row11 col12" >78</td>
      <td id="T_54717_row11_col13" class="data row11 col13" >105</td>
      <td id="T_54717_row11_col14" class="data row11 col14" >102</td>
      <td id="T_54717_row11_col15" class="data row11 col15" >95</td>
      <td id="T_54717_row11_col16" class="data row11 col16" >44</td>
      <td id="T_54717_row11_col17" class="data row11 col17" >29</td>
      <td id="T_54717_row11_col18" class="data row11 col18" >37</td>
      <td id="T_54717_row11_col19" class="data row11 col19" >46</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_54717_row12_col0" class="data row12 col0" >34</td>
      <td id="T_54717_row12_col1" class="data row12 col1" >105</td>
      <td id="T_54717_row12_col2" class="data row12 col2" >40</td>
      <td id="T_54717_row12_col3" class="data row12 col3" >71</td>
      <td id="T_54717_row12_col4" class="data row12 col4" >95</td>
      <td id="T_54717_row12_col5" class="data row12 col5" >85</td>
      <td id="T_54717_row12_col6" class="data row12 col6" >71</td>
      <td id="T_54717_row12_col7" class="data row12 col7" >59</td>
      <td id="T_54717_row12_col8" class="data row12 col8" >87</td>
      <td id="T_54717_row12_col9" class="data row12 col9" >74</td>
      <td id="T_54717_row12_col10" class="data row12 col10" >60</td>
      <td id="T_54717_row12_col11" class="data row12 col11" >46</td>
      <td id="T_54717_row12_col12" class="data row12 col12" >65</td>
      <td id="T_54717_row12_col13" class="data row12 col13" >76</td>
      <td id="T_54717_row12_col14" class="data row12 col14" >69</td>
      <td id="T_54717_row12_col15" class="data row12 col15" >79</td>
      <td id="T_54717_row12_col16" class="data row12 col16" >120</td>
      <td id="T_54717_row12_col17" class="data row12 col17" >78</td>
      <td id="T_54717_row12_col18" class="data row12 col18" >60</td>
      <td id="T_54717_row12_col19" class="data row12 col19" >72</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_54717_row13_col0" class="data row13 col0" >96</td>
      <td id="T_54717_row13_col1" class="data row13 col1" >69</td>
      <td id="T_54717_row13_col2" class="data row13 col2" >49</td>
      <td id="T_54717_row13_col3" class="data row13 col3" >100</td>
      <td id="T_54717_row13_col4" class="data row13 col4" >90</td>
      <td id="T_54717_row13_col5" class="data row13 col5" >84</td>
      <td id="T_54717_row13_col6" class="data row13 col6" >66</td>
      <td id="T_54717_row13_col7" class="data row13 col7" >58</td>
      <td id="T_54717_row13_col8" class="data row13 col8" >85</td>
      <td id="T_54717_row13_col9" class="data row13 col9" >65</td>
      <td id="T_54717_row13_col10" class="data row13 col10" >56</td>
      <td id="T_54717_row13_col11" class="data row13 col11" >55</td>
      <td id="T_54717_row13_col12" class="data row13 col12" >66</td>
      <td id="T_54717_row13_col13" class="data row13 col13" >65</td>
      <td id="T_54717_row13_col14" class="data row13 col14" >67</td>
      <td id="T_54717_row13_col15" class="data row13 col15" >62</td>
      <td id="T_54717_row13_col16" class="data row13 col16" >92</td>
      <td id="T_54717_row13_col17" class="data row13 col17" >87</td>
      <td id="T_54717_row13_col18" class="data row13 col18" >45</td>
      <td id="T_54717_row13_col19" class="data row13 col19" >51</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_54717_row14_col0" class="data row14 col0" >102</td>
      <td id="T_54717_row14_col1" class="data row14 col1" >36</td>
      <td id="T_54717_row14_col2" class="data row14 col2" >50</td>
      <td id="T_54717_row14_col3" class="data row14 col3" >93</td>
      <td id="T_54717_row14_col4" class="data row14 col4" >90</td>
      <td id="T_54717_row14_col5" class="data row14 col5" >84</td>
      <td id="T_54717_row14_col6" class="data row14 col6" >76</td>
      <td id="T_54717_row14_col7" class="data row14 col7" >81</td>
      <td id="T_54717_row14_col8" class="data row14 col8" >84</td>
      <td id="T_54717_row14_col9" class="data row14 col9" >63</td>
      <td id="T_54717_row14_col10" class="data row14 col10" >54</td>
      <td id="T_54717_row14_col11" class="data row14 col11" >69</td>
      <td id="T_54717_row14_col12" class="data row14 col12" >59</td>
      <td id="T_54717_row14_col13" class="data row14 col13" >62</td>
      <td id="T_54717_row14_col14" class="data row14 col14" >63</td>
      <td id="T_54717_row14_col15" class="data row14 col15" >60</td>
      <td id="T_54717_row14_col16" class="data row14 col16" >83</td>
      <td id="T_54717_row14_col17" class="data row14 col17" >77</td>
      <td id="T_54717_row14_col18" class="data row14 col18" >38</td>
      <td id="T_54717_row14_col19" class="data row14 col19" >33</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_54717_row15_col0" class="data row15 col0" >50</td>
      <td id="T_54717_row15_col1" class="data row15 col1" >36</td>
      <td id="T_54717_row15_col2" class="data row15 col2" >44</td>
      <td id="T_54717_row15_col3" class="data row15 col3" >82</td>
      <td id="T_54717_row15_col4" class="data row15 col4" >88</td>
      <td id="T_54717_row15_col5" class="data row15 col5" >83</td>
      <td id="T_54717_row15_col6" class="data row15 col6" >81</td>
      <td id="T_54717_row15_col7" class="data row15 col7" >88</td>
      <td id="T_54717_row15_col8" class="data row15 col8" >83</td>
      <td id="T_54717_row15_col9" class="data row15 col9" >62</td>
      <td id="T_54717_row15_col10" class="data row15 col10" >53</td>
      <td id="T_54717_row15_col11" class="data row15 col11" >66</td>
      <td id="T_54717_row15_col12" class="data row15 col12" >66</td>
      <td id="T_54717_row15_col13" class="data row15 col13" >68</td>
      <td id="T_54717_row15_col14" class="data row15 col14" >64</td>
      <td id="T_54717_row15_col15" class="data row15 col15" >66</td>
      <td id="T_54717_row15_col16" class="data row15 col16" >81</td>
      <td id="T_54717_row15_col17" class="data row15 col17" >71</td>
      <td id="T_54717_row15_col18" class="data row15 col18" >32</td>
      <td id="T_54717_row15_col19" class="data row15 col19" >24</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_54717_row16_col0" class="data row16 col0" >33</td>
      <td id="T_54717_row16_col1" class="data row16 col1" >35</td>
      <td id="T_54717_row16_col2" class="data row16 col2" >41</td>
      <td id="T_54717_row16_col3" class="data row16 col3" >71</td>
      <td id="T_54717_row16_col4" class="data row16 col4" >71</td>
      <td id="T_54717_row16_col5" class="data row16 col5" >66</td>
      <td id="T_54717_row16_col6" class="data row16 col6" >68</td>
      <td id="T_54717_row16_col7" class="data row16 col7" >83</td>
      <td id="T_54717_row16_col8" class="data row16 col8" >82</td>
      <td id="T_54717_row16_col9" class="data row16 col9" >64</td>
      <td id="T_54717_row16_col10" class="data row16 col10" >57</td>
      <td id="T_54717_row16_col11" class="data row16 col11" >66</td>
      <td id="T_54717_row16_col12" class="data row16 col12" >78</td>
      <td id="T_54717_row16_col13" class="data row16 col13" >70</td>
      <td id="T_54717_row16_col14" class="data row16 col14" >66</td>
      <td id="T_54717_row16_col15" class="data row16 col15" >79</td>
      <td id="T_54717_row16_col16" class="data row16 col16" >74</td>
      <td id="T_54717_row16_col17" class="data row16 col17" >67</td>
      <td id="T_54717_row16_col18" class="data row16 col18" >46</td>
      <td id="T_54717_row16_col19" class="data row16 col19" >29</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_54717_row17_col0" class="data row17 col0" >34</td>
      <td id="T_54717_row17_col1" class="data row17 col1" >33</td>
      <td id="T_54717_row17_col2" class="data row17 col2" >34</td>
      <td id="T_54717_row17_col3" class="data row17 col3" >62</td>
      <td id="T_54717_row17_col4" class="data row17 col4" >46</td>
      <td id="T_54717_row17_col5" class="data row17 col5" >45</td>
      <td id="T_54717_row17_col6" class="data row17 col6" >52</td>
      <td id="T_54717_row17_col7" class="data row17 col7" >72</td>
      <td id="T_54717_row17_col8" class="data row17 col8" >67</td>
      <td id="T_54717_row17_col9" class="data row17 col9" >56</td>
      <td id="T_54717_row17_col10" class="data row17 col10" >55</td>
      <td id="T_54717_row17_col11" class="data row17 col11" >61</td>
      <td id="T_54717_row17_col12" class="data row17 col12" >73</td>
      <td id="T_54717_row17_col13" class="data row17 col13" >63</td>
      <td id="T_54717_row17_col14" class="data row17 col14" >60</td>
      <td id="T_54717_row17_col15" class="data row17 col15" >68</td>
      <td id="T_54717_row17_col16" class="data row17 col16" >69</td>
      <td id="T_54717_row17_col17" class="data row17 col17" >60</td>
      <td id="T_54717_row17_col18" class="data row17 col18" >33</td>
      <td id="T_54717_row17_col19" class="data row17 col19" >20</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_54717_row18_col0" class="data row18 col0" >32</td>
      <td id="T_54717_row18_col1" class="data row18 col1" >28</td>
      <td id="T_54717_row18_col2" class="data row18 col2" >23</td>
      <td id="T_54717_row18_col3" class="data row18 col3" >56</td>
      <td id="T_54717_row18_col4" class="data row18 col4" >48</td>
      <td id="T_54717_row18_col5" class="data row18 col5" >43</td>
      <td id="T_54717_row18_col6" class="data row18 col6" >43</td>
      <td id="T_54717_row18_col7" class="data row18 col7" >48</td>
      <td id="T_54717_row18_col8" class="data row18 col8" >36</td>
      <td id="T_54717_row18_col9" class="data row18 col9" >36</td>
      <td id="T_54717_row18_col10" class="data row18 col10" >37</td>
      <td id="T_54717_row18_col11" class="data row18 col11" >38</td>
      <td id="T_54717_row18_col12" class="data row18 col12" >40</td>
      <td id="T_54717_row18_col13" class="data row18 col13" >38</td>
      <td id="T_54717_row18_col14" class="data row18 col14" >37</td>
      <td id="T_54717_row18_col15" class="data row18 col15" >39</td>
      <td id="T_54717_row18_col16" class="data row18 col16" >43</td>
      <td id="T_54717_row18_col17" class="data row18 col17" >48</td>
      <td id="T_54717_row18_col18" class="data row18 col18" >22</td>
      <td id="T_54717_row18_col19" class="data row18 col19" >16</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_54717_row19_col0" class="data row19 col0" >29</td>
      <td id="T_54717_row19_col1" class="data row19 col1" >19</td>
      <td id="T_54717_row19_col2" class="data row19 col2" >13</td>
      <td id="T_54717_row19_col3" class="data row19 col3" >45</td>
      <td id="T_54717_row19_col4" class="data row19 col4" >55</td>
      <td id="T_54717_row19_col5" class="data row19 col5" >42</td>
      <td id="T_54717_row19_col6" class="data row19 col6" >36</td>
      <td id="T_54717_row19_col7" class="data row19 col7" >36</td>
      <td id="T_54717_row19_col8" class="data row19 col8" >28</td>
      <td id="T_54717_row19_col9" class="data row19 col9" >27</td>
      <td id="T_54717_row19_col10" class="data row19 col10" >28</td>
      <td id="T_54717_row19_col11" class="data row19 col11" >28</td>
      <td id="T_54717_row19_col12" class="data row19 col12" >28</td>
      <td id="T_54717_row19_col13" class="data row19 col13" >28</td>
      <td id="T_54717_row19_col14" class="data row19 col14" >25</td>
      <td id="T_54717_row19_col15" class="data row19 col15" >28</td>
      <td id="T_54717_row19_col16" class="data row19 col16" >32</td>
      <td id="T_54717_row19_col17" class="data row19 col17" >41</td>
      <td id="T_54717_row19_col18" class="data row19 col18" >28</td>
      <td id="T_54717_row19_col19" class="data row19 col19" >13</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_54717_row20_col0" class="data row20 col0" >22</td>
      <td id="T_54717_row20_col1" class="data row20 col1" >10</td>
      <td id="T_54717_row20_col2" class="data row20 col2" >20</td>
      <td id="T_54717_row20_col3" class="data row20 col3" >47</td>
      <td id="T_54717_row20_col4" class="data row20 col4" >59</td>
      <td id="T_54717_row20_col5" class="data row20 col5" >46</td>
      <td id="T_54717_row20_col6" class="data row20 col6" >39</td>
      <td id="T_54717_row20_col7" class="data row20 col7" >35</td>
      <td id="T_54717_row20_col8" class="data row20 col8" >29</td>
      <td id="T_54717_row20_col9" class="data row20 col9" >29</td>
      <td id="T_54717_row20_col10" class="data row20 col10" >29</td>
      <td id="T_54717_row20_col11" class="data row20 col11" >29</td>
      <td id="T_54717_row20_col12" class="data row20 col12" >29</td>
      <td id="T_54717_row20_col13" class="data row20 col13" >29</td>
      <td id="T_54717_row20_col14" class="data row20 col14" >30</td>
      <td id="T_54717_row20_col15" class="data row20 col15" >33</td>
      <td id="T_54717_row20_col16" class="data row20 col16" >36</td>
      <td id="T_54717_row20_col17" class="data row20 col17" >36</td>
      <td id="T_54717_row20_col18" class="data row20 col18" >25</td>
      <td id="T_54717_row20_col19" class="data row20 col19" >19</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_54717_row21_col0" class="data row21 col0" >12</td>
      <td id="T_54717_row21_col1" class="data row21 col1" >17</td>
      <td id="T_54717_row21_col2" class="data row21 col2" >40</td>
      <td id="T_54717_row21_col3" class="data row21 col3" >19</td>
      <td id="T_54717_row21_col4" class="data row21 col4" >63</td>
      <td id="T_54717_row21_col5" class="data row21 col5" >55</td>
      <td id="T_54717_row21_col6" class="data row21 col6" >45</td>
      <td id="T_54717_row21_col7" class="data row21 col7" >37</td>
      <td id="T_54717_row21_col8" class="data row21 col8" >29</td>
      <td id="T_54717_row21_col9" class="data row21 col9" >34</td>
      <td id="T_54717_row21_col10" class="data row21 col10" >29</td>
      <td id="T_54717_row21_col11" class="data row21 col11" >29</td>
      <td id="T_54717_row21_col12" class="data row21 col12" >31</td>
      <td id="T_54717_row21_col13" class="data row21 col13" >30</td>
      <td id="T_54717_row21_col14" class="data row21 col14" >32</td>
      <td id="T_54717_row21_col15" class="data row21 col15" >32</td>
      <td id="T_54717_row21_col16" class="data row21 col16" >34</td>
      <td id="T_54717_row21_col17" class="data row21 col17" >18</td>
      <td id="T_54717_row21_col18" class="data row21 col18" >15</td>
      <td id="T_54717_row21_col19" class="data row21 col19" >21</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_54717_row22_col0" class="data row22 col0" >18</td>
      <td id="T_54717_row22_col1" class="data row22 col1" >43</td>
      <td id="T_54717_row22_col2" class="data row22 col2" >17</td>
      <td id="T_54717_row22_col3" class="data row22 col3" >23</td>
      <td id="T_54717_row22_col4" class="data row22 col4" >67</td>
      <td id="T_54717_row22_col5" class="data row22 col5" >59</td>
      <td id="T_54717_row22_col6" class="data row22 col6" >51</td>
      <td id="T_54717_row22_col7" class="data row22 col7" >39</td>
      <td id="T_54717_row22_col8" class="data row22 col8" >33</td>
      <td id="T_54717_row22_col9" class="data row22 col9" >35</td>
      <td id="T_54717_row22_col10" class="data row22 col10" >29</td>
      <td id="T_54717_row22_col11" class="data row22 col11" >31</td>
      <td id="T_54717_row22_col12" class="data row22 col12" >32</td>
      <td id="T_54717_row22_col13" class="data row22 col13" >32</td>
      <td id="T_54717_row22_col14" class="data row22 col14" >33</td>
      <td id="T_54717_row22_col15" class="data row22 col15" >34</td>
      <td id="T_54717_row22_col16" class="data row22 col16" >19</td>
      <td id="T_54717_row22_col17" class="data row22 col17" >9</td>
      <td id="T_54717_row22_col18" class="data row22 col18" >14</td>
      <td id="T_54717_row22_col19" class="data row22 col19" >14</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_54717_row23_col0" class="data row23 col0" >77</td>
      <td id="T_54717_row23_col1" class="data row23 col1" >82</td>
      <td id="T_54717_row23_col2" class="data row23 col2" >14</td>
      <td id="T_54717_row23_col3" class="data row23 col3" >23</td>
      <td id="T_54717_row23_col4" class="data row23 col4" >54</td>
      <td id="T_54717_row23_col5" class="data row23 col5" >57</td>
      <td id="T_54717_row23_col6" class="data row23 col6" >51</td>
      <td id="T_54717_row23_col7" class="data row23 col7" >38</td>
      <td id="T_54717_row23_col8" class="data row23 col8" >33</td>
      <td id="T_54717_row23_col9" class="data row23 col9" >32</td>
      <td id="T_54717_row23_col10" class="data row23 col10" >31</td>
      <td id="T_54717_row23_col11" class="data row23 col11" >35</td>
      <td id="T_54717_row23_col12" class="data row23 col12" >34</td>
      <td id="T_54717_row23_col13" class="data row23 col13" >30</td>
      <td id="T_54717_row23_col14" class="data row23 col14" >31</td>
      <td id="T_54717_row23_col15" class="data row23 col15" >30</td>
      <td id="T_54717_row23_col16" class="data row23 col16" >11</td>
      <td id="T_54717_row23_col17" class="data row23 col17" >1</td>
      <td id="T_54717_row23_col18" class="data row23 col18" >11</td>
      <td id="T_54717_row23_col19" class="data row23 col19" >13</td>
    </tr>
    <tr>
      <th id="T_54717_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_54717_row24_col0" class="data row24 col0" >77</td>
      <td id="T_54717_row24_col1" class="data row24 col1" >75</td>
      <td id="T_54717_row24_col2" class="data row24 col2" >43</td>
      <td id="T_54717_row24_col3" class="data row24 col3" >41</td>
      <td id="T_54717_row24_col4" class="data row24 col4" >54</td>
      <td id="T_54717_row24_col5" class="data row24 col5" >59</td>
      <td id="T_54717_row24_col6" class="data row24 col6" >57</td>
      <td id="T_54717_row24_col7" class="data row24 col7" >54</td>
      <td id="T_54717_row24_col8" class="data row24 col8" >53</td>
      <td id="T_54717_row24_col9" class="data row24 col9" >53</td>
      <td id="T_54717_row24_col10" class="data row24 col10" >55</td>
      <td id="T_54717_row24_col11" class="data row24 col11" >59</td>
      <td id="T_54717_row24_col12" class="data row24 col12" >67</td>
      <td id="T_54717_row24_col13" class="data row24 col13" >76</td>
      <td id="T_54717_row24_col14" class="data row24 col14" >77</td>
      <td id="T_54717_row24_col15" class="data row24 col15" >78</td>
      <td id="T_54717_row24_col16" class="data row24 col16" >69</td>
      <td id="T_54717_row24_col17" class="data row24 col17" >62</td>
      <td id="T_54717_row24_col18" class="data row24 col18" >67</td>
      <td id="T_54717_row24_col19" class="data row24 col19" >68</td>
    </tr>
  </tbody>
</table>





```python
df_blue = pd.DataFrame(im_tens[87:112,99:119,2])
df_red.style.set_properties(**{'font-size':'6pt'}).background_gradient('Blues')
```




<style type="text/css">
#T_f47c4_row0_col0, #T_f47c4_row0_col4, #T_f47c4_row0_col5, #T_f47c4_row0_col6, #T_f47c4_row0_col7, #T_f47c4_row0_col8, #T_f47c4_row1_col2, #T_f47c4_row1_col3, #T_f47c4_row2_col1, #T_f47c4_row4_col11, #T_f47c4_row4_col12, #T_f47c4_row7_col10, #T_f47c4_row8_col9, #T_f47c4_row10_col13, #T_f47c4_row10_col14, #T_f47c4_row11_col15, #T_f47c4_row12_col16, #T_f47c4_row12_col18, #T_f47c4_row12_col19, #T_f47c4_row13_col17 {
  font-size: 6pt;
  background-color: #08306b;
  color: #f1f1f1;
}
#T_f47c4_row0_col1 {
  font-size: 6pt;
  background-color: #08458a;
  color: #f1f1f1;
}
#T_f47c4_row0_col2, #T_f47c4_row3_col1 {
  font-size: 6pt;
  background-color: #084c95;
  color: #f1f1f1;
}
#T_f47c4_row0_col3, #T_f47c4_row2_col2 {
  font-size: 6pt;
  background-color: #083471;
  color: #f1f1f1;
}
#T_f47c4_row0_col9, #T_f47c4_row4_col0, #T_f47c4_row14_col19, #T_f47c4_row19_col17 {
  font-size: 6pt;
  background-color: #72b2d8;
  color: #f1f1f1;
}
#T_f47c4_row0_col10 {
  font-size: 6pt;
  background-color: #f2f7fd;
  color: #000000;
}
#T_f47c4_row0_col11, #T_f47c4_row2_col8, #T_f47c4_row2_col12, #T_f47c4_row2_col13, #T_f47c4_row3_col10, #T_f47c4_row3_col14, #T_f47c4_row3_col15, #T_f47c4_row3_col18, #T_f47c4_row3_col19, #T_f47c4_row4_col6, #T_f47c4_row4_col9, #T_f47c4_row5_col5, #T_f47c4_row5_col8, #T_f47c4_row6_col4, #T_f47c4_row6_col7, #T_f47c4_row7_col3, #T_f47c4_row11_col0, #T_f47c4_row20_col1, #T_f47c4_row21_col0, #T_f47c4_row21_col3, #T_f47c4_row23_col2, #T_f47c4_row23_col16, #T_f47c4_row23_col17 {
  font-size: 6pt;
  background-color: #f7fbff;
  color: #000000;
}
#T_f47c4_row0_col12, #T_f47c4_row1_col8, #T_f47c4_row1_col13, #T_f47c4_row4_col7, #T_f47c4_row9_col12, #T_f47c4_row20_col4, #T_f47c4_row21_col14, #T_f47c4_row24_col8 {
  font-size: 6pt;
  background-color: #c6dbef;
  color: #000000;
}
#T_f47c4_row0_col13, #T_f47c4_row6_col19, #T_f47c4_row16_col8 {
  font-size: 6pt;
  background-color: #4090c5;
  color: #f1f1f1;
}
#T_f47c4_row0_col14, #T_f47c4_row12_col13 {
  font-size: 6pt;
  background-color: #3787c0;
  color: #f1f1f1;
}
#T_f47c4_row0_col15, #T_f47c4_row11_col7 {
  font-size: 6pt;
  background-color: #8cc0dd;
  color: #000000;
}
#T_f47c4_row0_col16, #T_f47c4_row13_col2, #T_f47c4_row21_col13 {
  font-size: 6pt;
  background-color: #d5e5f4;
  color: #000000;
}
#T_f47c4_row0_col17, #T_f47c4_row5_col16, #T_f47c4_row10_col2, #T_f47c4_row10_col3, #T_f47c4_row10_col19, #T_f47c4_row12_col11, #T_f47c4_row17_col3 {
  font-size: 6pt;
  background-color: #c4daee;
  color: #000000;
}
#T_f47c4_row0_col18, #T_f47c4_row20_col15 {
  font-size: 6pt;
  background-color: #b0d2e7;
  color: #000000;
}
#T_f47c4_row0_col19, #T_f47c4_row23_col5 {
  font-size: 6pt;
  background-color: #b5d4e9;
  color: #000000;
}
#T_f47c4_row1_col0 {
  font-size: 6pt;
  background-color: #08519c;
  color: #f1f1f1;
}
#T_f47c4_row1_col1, #T_f47c4_row8_col8 {
  font-size: 6pt;
  background-color: #08478d;
  color: #f1f1f1;
}
#T_f47c4_row1_col4 {
  font-size: 6pt;
  background-color: #083370;
  color: #f1f1f1;
}
#T_f47c4_row1_col5, #T_f47c4_row2_col0 {
  font-size: 6pt;
  background-color: #084184;
  color: #f1f1f1;
}
#T_f47c4_row1_col6 {
  font-size: 6pt;
  background-color: #0d57a1;
  color: #f1f1f1;
}
#T_f47c4_row1_col7, #T_f47c4_row6_col0, #T_f47c4_row10_col5, #T_f47c4_row13_col9, #T_f47c4_row17_col14 {
  font-size: 6pt;
  background-color: #60a7d2;
  color: #f1f1f1;
}
#T_f47c4_row1_col9, #T_f47c4_row8_col13, #T_f47c4_row12_col0 {
  font-size: 6pt;
  background-color: #e7f1fa;
  color: #000000;
}
#T_f47c4_row1_col10 {
  font-size: 6pt;
  background-color: #f4f9fe;
  color: #000000;
}
#T_f47c4_row1_col11, #T_f47c4_row8_col7 {
  font-size: 6pt;
  background-color: #79b5d9;
  color: #000000;
}
#T_f47c4_row1_col12, #T_f47c4_row10_col4, #T_f47c4_row11_col4, #T_f47c4_row14_col13, #T_f47c4_row17_col12, #T_f47c4_row17_col13 {
  font-size: 6pt;
  background-color: #5ca4d0;
  color: #f1f1f1;
}
#T_f47c4_row1_col14, #T_f47c4_row8_col3, #T_f47c4_row19_col1, #T_f47c4_row22_col0 {
  font-size: 6pt;
  background-color: #f1f7fd;
  color: #000000;
}
#T_f47c4_row1_col15, #T_f47c4_row2_col18, #T_f47c4_row2_col19, #T_f47c4_row4_col14, #T_f47c4_row6_col14, #T_f47c4_row9_col1, #T_f47c4_row15_col1, #T_f47c4_row23_col18 {
  font-size: 6pt;
  background-color: #e3eef8;
  color: #000000;
}
#T_f47c4_row1_col16, #T_f47c4_row4_col17, #T_f47c4_row18_col3, #T_f47c4_row20_col14 {
  font-size: 6pt;
  background-color: #cfe1f2;
  color: #000000;
}
#T_f47c4_row1_col17, #T_f47c4_row2_col9, #T_f47c4_row3_col11 {
  font-size: 6pt;
  background-color: #bfd8ed;
  color: #000000;
}
#T_f47c4_row1_col18, #T_f47c4_row7_col2, #T_f47c4_row15_col2, #T_f47c4_row21_col7 {
  font-size: 6pt;
  background-color: #d8e7f5;
  color: #000000;
}
#T_f47c4_row1_col19, #T_f47c4_row19_col12, #T_f47c4_row19_col14, #T_f47c4_row20_col12, #T_f47c4_row24_col2 {
  font-size: 6pt;
  background-color: #dbe9f6;
  color: #000000;
}
#T_f47c4_row2_col3 {
  font-size: 6pt;
  background-color: #083b7c;
  color: #f1f1f1;
}
#T_f47c4_row2_col4, #T_f47c4_row9_col9 {
  font-size: 6pt;
  background-color: #084e98;
  color: #f1f1f1;
}
#T_f47c4_row2_col5 {
  font-size: 6pt;
  background-color: #1f6eb3;
  color: #f1f1f1;
}
#T_f47c4_row2_col6, #T_f47c4_row3_col13, #T_f47c4_row8_col16 {
  font-size: 6pt;
  background-color: #97c6df;
  color: #000000;
}
#T_f47c4_row2_col7, #T_f47c4_row16_col2, #T_f47c4_row18_col19, #T_f47c4_row19_col13, #T_f47c4_row20_col8 {
  font-size: 6pt;
  background-color: #ddeaf7;
  color: #000000;
}
#T_f47c4_row2_col10, #T_f47c4_row5_col12 {
  font-size: 6pt;
  background-color: #3888c1;
  color: #f1f1f1;
}
#T_f47c4_row2_col11, #T_f47c4_row8_col0 {
  font-size: 6pt;
  background-color: #b9d6ea;
  color: #000000;
}
#T_f47c4_row2_col14, #T_f47c4_row2_col17, #T_f47c4_row18_col10, #T_f47c4_row19_col3 {
  font-size: 6pt;
  background-color: #dceaf6;
  color: #000000;
}
#T_f47c4_row2_col15, #T_f47c4_row9_col0, #T_f47c4_row20_col3 {
  font-size: 6pt;
  background-color: #d9e8f5;
  color: #000000;
}
#T_f47c4_row2_col16, #T_f47c4_row14_col1, #T_f47c4_row20_col11 {
  font-size: 6pt;
  background-color: #e4eff9;
  color: #000000;
}
#T_f47c4_row3_col0 {
  font-size: 6pt;
  background-color: #135fa7;
  color: #f1f1f1;
}
#T_f47c4_row3_col2 {
  font-size: 6pt;
  background-color: #083c7d;
  color: #f1f1f1;
}
#T_f47c4_row3_col3 {
  font-size: 6pt;
  background-color: #084a91;
  color: #f1f1f1;
}
#T_f47c4_row3_col4 {
  font-size: 6pt;
  background-color: #2a7ab9;
  color: #f1f1f1;
}
#T_f47c4_row3_col5, #T_f47c4_row12_col3 {
  font-size: 6pt;
  background-color: #b4d3e9;
  color: #000000;
}
#T_f47c4_row3_col6, #T_f47c4_row21_col18 {
  font-size: 6pt;
  background-color: #e0ecf8;
  color: #000000;
}
#T_f47c4_row3_col7, #T_f47c4_row7_col6, #T_f47c4_row20_col0 {
  font-size: 6pt;
  background-color: #f6faff;
  color: #000000;
}
#T_f47c4_row3_col8, #T_f47c4_row11_col11 {
  font-size: 6pt;
  background-color: #a8cee4;
  color: #000000;
}
#T_f47c4_row3_col9 {
  font-size: 6pt;
  background-color: #68acd5;
  color: #f1f1f1;
}
#T_f47c4_row3_col12, #T_f47c4_row7_col11 {
  font-size: 6pt;
  background-color: #4493c7;
  color: #f1f1f1;
}
#T_f47c4_row3_col16 {
  font-size: 6pt;
  background-color: #ecf4fb;
  color: #000000;
}
#T_f47c4_row3_col17, #T_f47c4_row4_col5, #T_f47c4_row4_col19, #T_f47c4_row7_col8, #T_f47c4_row10_col16, #T_f47c4_row19_col19, #T_f47c4_row22_col16 {
  font-size: 6pt;
  background-color: #e1edf8;
  color: #000000;
}
#T_f47c4_row4_col1, #T_f47c4_row6_col17 {
  font-size: 6pt;
  background-color: #63a8d3;
  color: #f1f1f1;
}
#T_f47c4_row4_col2, #T_f47c4_row18_col17 {
  font-size: 6pt;
  background-color: #5da5d1;
  color: #f1f1f1;
}
#T_f47c4_row4_col3, #T_f47c4_row7_col9, #T_f47c4_row12_col5, #T_f47c4_row13_col5, #T_f47c4_row14_col12, #T_f47c4_row16_col11, #T_f47c4_row17_col9 {
  font-size: 6pt;
  background-color: #7cb7da;
  color: #000000;
}
#T_f47c4_row4_col4 {
  font-size: 6pt;
  background-color: #b3d3e8;
  color: #000000;
}
#T_f47c4_row4_col8, #T_f47c4_row16_col4, #T_f47c4_row20_col17 {
  font-size: 6pt;
  background-color: #9cc9e1;
  color: #000000;
}
#T_f47c4_row4_col10, #T_f47c4_row11_col3, #T_f47c4_row14_col2, #T_f47c4_row21_col17, #T_f47c4_row23_col8 {
  font-size: 6pt;
  background-color: #d3e3f3;
  color: #000000;
}
#T_f47c4_row4_col13, #T_f47c4_row9_col8 {
  font-size: 6pt;
  background-color: #1663aa;
  color: #f1f1f1;
}
#T_f47c4_row4_col15, #T_f47c4_row8_col2, #T_f47c4_row9_col13, #T_f47c4_row16_col0, #T_f47c4_row23_col3 {
  font-size: 6pt;
  background-color: #eef5fc;
  color: #000000;
}
#T_f47c4_row4_col16, #T_f47c4_row5_col15, #T_f47c4_row6_col2, #T_f47c4_row8_col5, #T_f47c4_row19_col6, #T_f47c4_row19_col8, #T_f47c4_row19_col9, #T_f47c4_row20_col9, #T_f47c4_row21_col11 {
  font-size: 6pt;
  background-color: #dce9f6;
  color: #000000;
}
#T_f47c4_row4_col18, #T_f47c4_row10_col18 {
  font-size: 6pt;
  background-color: #deebf7;
  color: #000000;
}
#T_f47c4_row5_col0 {
  font-size: 6pt;
  background-color: #64a9d3;
  color: #f1f1f1;
}
#T_f47c4_row5_col1, #T_f47c4_row11_col12, #T_f47c4_row16_col12 {
  font-size: 6pt;
  background-color: #56a0ce;
  color: #f1f1f1;
}
#T_f47c4_row5_col2, #T_f47c4_row9_col19 {
  font-size: 6pt;
  background-color: #a6cee4;
  color: #000000;
}
#T_f47c4_row5_col3, #T_f47c4_row6_col3, #T_f47c4_row17_col2, #T_f47c4_row19_col11 {
  font-size: 6pt;
  background-color: #e7f0fa;
  color: #000000;
}
#T_f47c4_row5_col4 {
  font-size: 6pt;
  background-color: #e3eef9;
  color: #000000;
}
#T_f47c4_row5_col6, #T_f47c4_row6_col6, #T_f47c4_row7_col5, #T_f47c4_row11_col1, #T_f47c4_row18_col11, #T_f47c4_row23_col12, #T_f47c4_row24_col4, #T_f47c4_row24_col6 {
  font-size: 6pt;
  background-color: #cee0f2;
  color: #000000;
}
#T_f47c4_row5_col7, #T_f47c4_row22_col6 {
  font-size: 6pt;
  background-color: #cde0f1;
  color: #000000;
}
#T_f47c4_row5_col9, #T_f47c4_row17_col0, #T_f47c4_row18_col1 {
  font-size: 6pt;
  background-color: #eaf3fb;
  color: #000000;
}
#T_f47c4_row5_col10, #T_f47c4_row7_col17, #T_f47c4_row17_col16 {
  font-size: 6pt;
  background-color: #3686c0;
  color: #f1f1f1;
}
#T_f47c4_row5_col11 {
  font-size: 6pt;
  background-color: #1966ad;
  color: #f1f1f1;
}
#T_f47c4_row5_col13, #T_f47c4_row9_col4, #T_f47c4_row13_col4, #T_f47c4_row14_col11, #T_f47c4_row15_col4 {
  font-size: 6pt;
  background-color: #77b5d9;
  color: #000000;
}
#T_f47c4_row5_col14 {
  font-size: 6pt;
  background-color: #e9f2fa;
  color: #000000;
}
#T_f47c4_row5_col17, #T_f47c4_row5_col19, #T_f47c4_row17_col11, #T_f47c4_row18_col16 {
  font-size: 6pt;
  background-color: #8dc1dd;
  color: #000000;
}
#T_f47c4_row5_col18, #T_f47c4_row14_col18, #T_f47c4_row15_col7, #T_f47c4_row15_col12 {
  font-size: 6pt;
  background-color: #6aaed6;
  color: #f1f1f1;
}
#T_f47c4_row6_col1, #T_f47c4_row11_col10, #T_f47c4_row24_col13 {
  font-size: 6pt;
  background-color: #85bcdc;
  color: #000000;
}
#T_f47c4_row6_col5, #T_f47c4_row8_col1, #T_f47c4_row21_col12, #T_f47c4_row23_col7 {
  font-size: 6pt;
  background-color: #d6e5f4;
  color: #000000;
}
#T_f47c4_row6_col8, #T_f47c4_row10_col1, #T_f47c4_row19_col10 {
  font-size: 6pt;
  background-color: #f0f6fd;
  color: #000000;
}
#T_f47c4_row6_col9, #T_f47c4_row7_col13, #T_f47c4_row9_col14, #T_f47c4_row20_col6, #T_f47c4_row23_col13, #T_f47c4_row23_col14 {
  font-size: 6pt;
  background-color: #d6e6f4;
  color: #000000;
}
#T_f47c4_row6_col10, #T_f47c4_row13_col19 {
  font-size: 6pt;
  background-color: #2474b7;
  color: #f1f1f1;
}
#T_f47c4_row6_col11 {
  font-size: 6pt;
  background-color: #4f9bcb;
  color: #f1f1f1;
}
#T_f47c4_row6_col12, #T_f47c4_row7_col12, #T_f47c4_row11_col5, #T_f47c4_row13_col3, #T_f47c4_row13_col12, #T_f47c4_row14_col3 {
  font-size: 6pt;
  background-color: #71b1d7;
  color: #f1f1f1;
}
#T_f47c4_row6_col13, #T_f47c4_row9_col16, #T_f47c4_row17_col4, #T_f47c4_row17_col5, #T_f47c4_row19_col15, #T_f47c4_row21_col19, #T_f47c4_row23_col6 {
  font-size: 6pt;
  background-color: #cbdef1;
  color: #000000;
}
#T_f47c4_row6_col15, #T_f47c4_row18_col7, #T_f47c4_row24_col10 {
  font-size: 6pt;
  background-color: #caddf0;
  color: #000000;
}
#T_f47c4_row6_col16, #T_f47c4_row20_col16 {
  font-size: 6pt;
  background-color: #b2d2e8;
  color: #000000;
}
#T_f47c4_row6_col18 {
  font-size: 6pt;
  background-color: #3989c1;
  color: #f1f1f1;
}
#T_f47c4_row7_col0 {
  font-size: 6pt;
  background-color: #6dafd7;
  color: #f1f1f1;
}
#T_f47c4_row7_col1, #T_f47c4_row19_col4 {
  font-size: 6pt;
  background-color: #cadef0;
  color: #000000;
}
#T_f47c4_row7_col4, #T_f47c4_row12_col2 {
  font-size: 6pt;
  background-color: #e2edf8;
  color: #000000;
}
#T_f47c4_row7_col7, #T_f47c4_row9_col2 {
  font-size: 6pt;
  background-color: #f2f8fd;
  color: #000000;
}
#T_f47c4_row7_col14, #T_f47c4_row8_col6, #T_f47c4_row20_col19, #T_f47c4_row22_col1, #T_f47c4_row22_col9, #T_f47c4_row22_col13, #T_f47c4_row23_col15 {
  font-size: 6pt;
  background-color: #d0e2f2;
  color: #000000;
}
#T_f47c4_row7_col15 {
  font-size: 6pt;
  background-color: #a1cbe2;
  color: #000000;
}
#T_f47c4_row7_col16, #T_f47c4_row11_col16 {
  font-size: 6pt;
  background-color: #8abfdd;
  color: #000000;
}
#T_f47c4_row7_col18 {
  font-size: 6pt;
  background-color: #0e59a2;
  color: #f1f1f1;
}
#T_f47c4_row7_col19 {
  font-size: 6pt;
  background-color: #2777b8;
  color: #f1f1f1;
}
#T_f47c4_row8_col4, #T_f47c4_row19_col18, #T_f47c4_row21_col5 {
  font-size: 6pt;
  background-color: #c3daee;
  color: #000000;
}
#T_f47c4_row8_col10 {
  font-size: 6pt;
  background-color: #083776;
  color: #f1f1f1;
}
#T_f47c4_row8_col11, #T_f47c4_row24_col19 {
  font-size: 6pt;
  background-color: #3181bd;
  color: #f1f1f1;
}
#T_f47c4_row8_col12, #T_f47c4_row17_col7 {
  font-size: 6pt;
  background-color: #8fc2de;
  color: #000000;
}
#T_f47c4_row8_col14 {
  font-size: 6pt;
  background-color: #ccdff1;
  color: #000000;
}
#T_f47c4_row8_col15, #T_f47c4_row14_col0 {
  font-size: 6pt;
  background-color: #9fcae1;
  color: #000000;
}
#T_f47c4_row8_col17 {
  font-size: 6pt;
  background-color: #4997c9;
  color: #f1f1f1;
}
#T_f47c4_row8_col18 {
  font-size: 6pt;
  background-color: #115ca5;
  color: #f1f1f1;
}
#T_f47c4_row8_col19 {
  font-size: 6pt;
  background-color: #1561a9;
  color: #f1f1f1;
}
#T_f47c4_row9_col3, #T_f47c4_row17_col6, #T_f47c4_row18_col13 {
  font-size: 6pt;
  background-color: #bed8ec;
  color: #000000;
}
#T_f47c4_row9_col5 {
  font-size: 6pt;
  background-color: #4191c6;
  color: #f1f1f1;
}
#T_f47c4_row9_col6, #T_f47c4_row11_col8 {
  font-size: 6pt;
  background-color: #3484bf;
  color: #f1f1f1;
}
#T_f47c4_row9_col7 {
  font-size: 6pt;
  background-color: #2575b7;
  color: #f1f1f1;
}
#T_f47c4_row9_col10 {
  font-size: 6pt;
  background-color: #1460a8;
  color: #f1f1f1;
}
#T_f47c4_row9_col11 {
  font-size: 6pt;
  background-color: #529dcc;
  color: #f1f1f1;
}
#T_f47c4_row9_col15, #T_f47c4_row22_col5, #T_f47c4_row22_col15, #T_f47c4_row23_col4, #T_f47c4_row24_col9 {
  font-size: 6pt;
  background-color: #bdd7ec;
  color: #000000;
}
#T_f47c4_row9_col17, #T_f47c4_row18_col8 {
  font-size: 6pt;
  background-color: #c7dcef;
  color: #000000;
}
#T_f47c4_row9_col18, #T_f47c4_row18_col9, #T_f47c4_row24_col5 {
  font-size: 6pt;
  background-color: #c8dcf0;
  color: #000000;
}
#T_f47c4_row10_col0, #T_f47c4_row22_col17, #T_f47c4_row24_col3 {
  font-size: 6pt;
  background-color: #e5eff9;
  color: #000000;
}
#T_f47c4_row10_col6 {
  font-size: 6pt;
  background-color: #74b3d8;
  color: #000000;
}
#T_f47c4_row10_col7, #T_f47c4_row10_col12, #T_f47c4_row14_col4, #T_f47c4_row14_col9, #T_f47c4_row15_col9, #T_f47c4_row24_col17 {
  font-size: 6pt;
  background-color: #6fb0d7;
  color: #f1f1f1;
}
#T_f47c4_row10_col8 {
  font-size: 6pt;
  background-color: #2272b6;
  color: #f1f1f1;
}
#T_f47c4_row10_col9, #T_f47c4_row13_col15 {
  font-size: 6pt;
  background-color: #2e7ebc;
  color: #f1f1f1;
}
#T_f47c4_row10_col10 {
  font-size: 6pt;
  background-color: #4d99ca;
  color: #f1f1f1;
}
#T_f47c4_row10_col11, #T_f47c4_row17_col18 {
  font-size: 6pt;
  background-color: #87bddc;
  color: #000000;
}
#T_f47c4_row10_col15 {
  font-size: 6pt;
  background-color: #4e9acb;
  color: #f1f1f1;
}
#T_f47c4_row10_col17 {
  font-size: 6pt;
  background-color: #dfecf7;
  color: #000000;
}
#T_f47c4_row11_col2 {
  font-size: 6pt;
  background-color: #b8d5ea;
  color: #000000;
}
#T_f47c4_row11_col6, #T_f47c4_row12_col10, #T_f47c4_row14_col5, #T_f47c4_row14_col7 {
  font-size: 6pt;
  background-color: #81badb;
  color: #000000;
}
#T_f47c4_row11_col9 {
  font-size: 6pt;
  background-color: #57a0ce;
  color: #f1f1f1;
}
#T_f47c4_row11_col13 {
  font-size: 6pt;
  background-color: #0c56a0;
  color: #f1f1f1;
}
#T_f47c4_row11_col14, #T_f47c4_row14_col17 {
  font-size: 6pt;
  background-color: #08468b;
  color: #f1f1f1;
}
#T_f47c4_row11_col17, #T_f47c4_row15_col3, #T_f47c4_row15_col5, #T_f47c4_row15_col6, #T_f47c4_row24_col14 {
  font-size: 6pt;
  background-color: #84bcdb;
  color: #000000;
}
#T_f47c4_row11_col18 {
  font-size: 6pt;
  background-color: #4594c7;
  color: #f1f1f1;
}
#T_f47c4_row11_col19 {
  font-size: 6pt;
  background-color: #3d8dc4;
  color: #f1f1f1;
}
#T_f47c4_row12_col1, #T_f47c4_row17_col10 {
  font-size: 6pt;
  background-color: #91c3de;
  color: #000000;
}
#T_f47c4_row12_col4 {
  font-size: 6pt;
  background-color: #69add5;
  color: #f1f1f1;
}
#T_f47c4_row12_col6, #T_f47c4_row13_col10 {
  font-size: 6pt;
  background-color: #95c5df;
  color: #000000;
}
#T_f47c4_row12_col7, #T_f47c4_row13_col0 {
  font-size: 6pt;
  background-color: #a9cfe5;
  color: #000000;
}
#T_f47c4_row12_col8 {
  font-size: 6pt;
  background-color: #2d7dbb;
  color: #f1f1f1;
}
#T_f47c4_row12_col9, #T_f47c4_row16_col18 {
  font-size: 6pt;
  background-color: #549fcd;
  color: #f1f1f1;
}
#T_f47c4_row12_col12 {
  font-size: 6pt;
  background-color: #7db8da;
  color: #000000;
}
#T_f47c4_row12_col14, #T_f47c4_row14_col15 {
  font-size: 6pt;
  background-color: #3585bf;
  color: #f1f1f1;
}
#T_f47c4_row12_col15, #T_f47c4_row16_col15 {
  font-size: 6pt;
  background-color: #09529d;
  color: #f1f1f1;
}
#T_f47c4_row12_col17 {
  font-size: 6pt;
  background-color: #084488;
  color: #f1f1f1;
}
#T_f47c4_row13_col1, #T_f47c4_row17_col19, #T_f47c4_row22_col14 {
  font-size: 6pt;
  background-color: #c9ddf0;
  color: #000000;
}
#T_f47c4_row13_col6, #T_f47c4_row15_col10, #T_f47c4_row16_col6 {
  font-size: 6pt;
  background-color: #a0cbe2;
  color: #000000;
}
#T_f47c4_row13_col7, #T_f47c4_row13_col11, #T_f47c4_row21_col15 {
  font-size: 6pt;
  background-color: #aed1e7;
  color: #000000;
}
#T_f47c4_row13_col8 {
  font-size: 6pt;
  background-color: #3a8ac2;
  color: #f1f1f1;
}
#T_f47c4_row13_col13, #T_f47c4_row16_col14 {
  font-size: 6pt;
  background-color: #4b98ca;
  color: #f1f1f1;
}
#T_f47c4_row13_col14 {
  font-size: 6pt;
  background-color: #3f8fc5;
  color: #f1f1f1;
}
#T_f47c4_row13_col16, #T_f47c4_row15_col17 {
  font-size: 6pt;
  background-color: #0f5aa3;
  color: #f1f1f1;
}
#T_f47c4_row13_col18 {
  font-size: 6pt;
  background-color: #2070b4;
  color: #f1f1f1;
}
#T_f47c4_row14_col6 {
  font-size: 6pt;
  background-color: #92c4de;
  color: #000000;
}
#T_f47c4_row14_col8, #T_f47c4_row15_col8 {
  font-size: 6pt;
  background-color: #3e8ec4;
  color: #f1f1f1;
}
#T_f47c4_row14_col10 {
  font-size: 6pt;
  background-color: #a4cce3;
  color: #000000;
}
#T_f47c4_row14_col14, #T_f47c4_row15_col14, #T_f47c4_row16_col13 {
  font-size: 6pt;
  background-color: #519ccc;
  color: #f1f1f1;
}
#T_f47c4_row14_col16 {
  font-size: 6pt;
  background-color: #1b69af;
  color: #f1f1f1;
}
#T_f47c4_row15_col0, #T_f47c4_row19_col7, #T_f47c4_row20_col7, #T_f47c4_row20_col13, #T_f47c4_row21_col2, #T_f47c4_row22_col18 {
  font-size: 6pt;
  background-color: #dae8f6;
  color: #000000;
}
#T_f47c4_row15_col11, #T_f47c4_row24_col16 {
  font-size: 6pt;
  background-color: #7fb9da;
  color: #000000;
}
#T_f47c4_row15_col13 {
  font-size: 6pt;
  background-color: #539ecd;
  color: #f1f1f1;
}
#T_f47c4_row15_col15 {
  font-size: 6pt;
  background-color: #2171b5;
  color: #f1f1f1;
}
#T_f47c4_row15_col16, #T_f47c4_row17_col15 {
  font-size: 6pt;
  background-color: #1d6cb1;
  color: #f1f1f1;
}
#T_f47c4_row15_col18, #T_f47c4_row24_col12 {
  font-size: 6pt;
  background-color: #9ac8e0;
  color: #000000;
}
#T_f47c4_row15_col19, #T_f47c4_row16_col19, #T_f47c4_row24_col1 {
  font-size: 6pt;
  background-color: #add0e6;
  color: #000000;
}
#T_f47c4_row16_col1 {
  font-size: 6pt;
  background-color: #e8f1fa;
  color: #000000;
}
#T_f47c4_row16_col3, #T_f47c4_row16_col5, #T_f47c4_row23_col0 {
  font-size: 6pt;
  background-color: #a3cce3;
  color: #000000;
}
#T_f47c4_row16_col7 {
  font-size: 6pt;
  background-color: #75b4d8;
  color: #000000;
}
#T_f47c4_row16_col9 {
  font-size: 6pt;
  background-color: #65aad4;
  color: #f1f1f1;
}
#T_f47c4_row16_col10 {
  font-size: 6pt;
  background-color: #89bedc;
  color: #000000;
}
#T_f47c4_row16_col16 {
  font-size: 6pt;
  background-color: #2676b8;
  color: #f1f1f1;
}
#T_f47c4_row16_col17 {
  font-size: 6pt;
  background-color: #1562a9;
  color: #f1f1f1;
}
#T_f47c4_row17_col1, #T_f47c4_row20_col10, #T_f47c4_row21_col1, #T_f47c4_row21_col10, #T_f47c4_row22_col10 {
  font-size: 6pt;
  background-color: #eaf2fb;
  color: #000000;
}
#T_f47c4_row17_col8 {
  font-size: 6pt;
  background-color: #61a7d2;
  color: #f1f1f1;
}
#T_f47c4_row17_col17 {
  font-size: 6pt;
  background-color: #2b7bba;
  color: #f1f1f1;
}
#T_f47c4_row18_col0, #T_f47c4_row20_col2, #T_f47c4_row22_col3 {
  font-size: 6pt;
  background-color: #edf4fc;
  color: #000000;
}
#T_f47c4_row18_col2 {
  font-size: 6pt;
  background-color: #eff6fc;
  color: #000000;
}
#T_f47c4_row18_col4, #T_f47c4_row20_col5, #T_f47c4_row24_col7 {
  font-size: 6pt;
  background-color: #d0e1f2;
  color: #000000;
}
#T_f47c4_row18_col5, #T_f47c4_row18_col6, #T_f47c4_row19_col5, #T_f47c4_row21_col6 {
  font-size: 6pt;
  background-color: #d1e2f3;
  color: #000000;
}
#T_f47c4_row18_col12, #T_f47c4_row21_col4 {
  font-size: 6pt;
  background-color: #bad6eb;
  color: #000000;
}
#T_f47c4_row18_col14 {
  font-size: 6pt;
  background-color: #c1d9ed;
  color: #000000;
}
#T_f47c4_row18_col15 {
  font-size: 6pt;
  background-color: #94c4df;
  color: #000000;
}
#T_f47c4_row18_col18, #T_f47c4_row20_col18 {
  font-size: 6pt;
  background-color: #cddff1;
  color: #000000;
}
#T_f47c4_row19_col0, #T_f47c4_row19_col2, #T_f47c4_row22_col2 {
  font-size: 6pt;
  background-color: #f3f8fe;
  color: #000000;
}
#T_f47c4_row19_col16 {
  font-size: 6pt;
  background-color: #c2d9ee;
  color: #000000;
}
#T_f47c4_row21_col8, #T_f47c4_row22_col11 {
  font-size: 6pt;
  background-color: #d9e7f5;
  color: #000000;
}
#T_f47c4_row21_col9, #T_f47c4_row22_col8, #T_f47c4_row22_col12, #T_f47c4_row23_col11 {
  font-size: 6pt;
  background-color: #d2e3f3;
  color: #000000;
}
#T_f47c4_row21_col16 {
  font-size: 6pt;
  background-color: #bcd7eb;
  color: #000000;
}
#T_f47c4_row22_col4 {
  font-size: 6pt;
  background-color: #abd0e6;
  color: #000000;
}
#T_f47c4_row22_col7, #T_f47c4_row23_col19 {
  font-size: 6pt;
  background-color: #d4e4f4;
  color: #000000;
}
#T_f47c4_row22_col19 {
  font-size: 6pt;
  background-color: #d7e6f5;
  color: #000000;
}
#T_f47c4_row23_col1 {
  font-size: 6pt;
  background-color: #9dcae1;
  color: #000000;
}
#T_f47c4_row23_col9 {
  font-size: 6pt;
  background-color: #d3e4f3;
  color: #000000;
}
#T_f47c4_row23_col10 {
  font-size: 6pt;
  background-color: #e6f0f9;
  color: #000000;
}
#T_f47c4_row24_col0 {
  font-size: 6pt;
  background-color: #aacfe5;
  color: #000000;
}
#T_f47c4_row24_col11 {
  font-size: 6pt;
  background-color: #b7d4ea;
  color: #000000;
}
#T_f47c4_row24_col15 {
  font-size: 6pt;
  background-color: #4896c8;
  color: #f1f1f1;
}
#T_f47c4_row24_col18 {
  font-size: 6pt;
  background-color: #3080bd;
  color: #f1f1f1;
}
</style>
<table id="T_f47c4">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_f47c4_level0_col0" class="col_heading level0 col0" >0</th>
      <th id="T_f47c4_level0_col1" class="col_heading level0 col1" >1</th>
      <th id="T_f47c4_level0_col2" class="col_heading level0 col2" >2</th>
      <th id="T_f47c4_level0_col3" class="col_heading level0 col3" >3</th>
      <th id="T_f47c4_level0_col4" class="col_heading level0 col4" >4</th>
      <th id="T_f47c4_level0_col5" class="col_heading level0 col5" >5</th>
      <th id="T_f47c4_level0_col6" class="col_heading level0 col6" >6</th>
      <th id="T_f47c4_level0_col7" class="col_heading level0 col7" >7</th>
      <th id="T_f47c4_level0_col8" class="col_heading level0 col8" >8</th>
      <th id="T_f47c4_level0_col9" class="col_heading level0 col9" >9</th>
      <th id="T_f47c4_level0_col10" class="col_heading level0 col10" >10</th>
      <th id="T_f47c4_level0_col11" class="col_heading level0 col11" >11</th>
      <th id="T_f47c4_level0_col12" class="col_heading level0 col12" >12</th>
      <th id="T_f47c4_level0_col13" class="col_heading level0 col13" >13</th>
      <th id="T_f47c4_level0_col14" class="col_heading level0 col14" >14</th>
      <th id="T_f47c4_level0_col15" class="col_heading level0 col15" >15</th>
      <th id="T_f47c4_level0_col16" class="col_heading level0 col16" >16</th>
      <th id="T_f47c4_level0_col17" class="col_heading level0 col17" >17</th>
      <th id="T_f47c4_level0_col18" class="col_heading level0 col18" >18</th>
      <th id="T_f47c4_level0_col19" class="col_heading level0 col19" >19</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_f47c4_level0_row0" class="row_heading level0 row0" >0</th>
      <td id="T_f47c4_row0_col0" class="data row0 col0" >255</td>
      <td id="T_f47c4_row0_col1" class="data row0 col1" >236</td>
      <td id="T_f47c4_row0_col2" class="data row0 col2" >230</td>
      <td id="T_f47c4_row0_col3" class="data row0 col3" >251</td>
      <td id="T_f47c4_row0_col4" class="data row0 col4" >254</td>
      <td id="T_f47c4_row0_col5" class="data row0 col5" >252</td>
      <td id="T_f47c4_row0_col6" class="data row0 col6" >244</td>
      <td id="T_f47c4_row0_col7" class="data row0 col7" >229</td>
      <td id="T_f47c4_row0_col8" class="data row0 col8" >181</td>
      <td id="T_f47c4_row0_col9" class="data row0 col9" >108</td>
      <td id="T_f47c4_row0_col10" class="data row0 col10" >62</td>
      <td id="T_f47c4_row0_col11" class="data row0 col11" >45</td>
      <td id="T_f47c4_row0_col12" class="data row0 col12" >77</td>
      <td id="T_f47c4_row0_col13" class="data row0 col13" >128</td>
      <td id="T_f47c4_row0_col14" class="data row0 col14" >123</td>
      <td id="T_f47c4_row0_col15" class="data row0 col15" >78</td>
      <td id="T_f47c4_row0_col16" class="data row0 col16" >49</td>
      <td id="T_f47c4_row0_col17" class="data row0 col17" >47</td>
      <td id="T_f47c4_row0_col18" class="data row0 col18" >52</td>
      <td id="T_f47c4_row0_col19" class="data row0 col19" >49</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row1" class="row_heading level0 row1" >1</th>
      <td id="T_f47c4_row1_col0" class="data row1 col0" >226</td>
      <td id="T_f47c4_row1_col1" class="data row1 col1" >234</td>
      <td id="T_f47c4_row1_col2" class="data row1 col2" >254</td>
      <td id="T_f47c4_row1_col3" class="data row1 col3" >255</td>
      <td id="T_f47c4_row1_col4" class="data row1 col4" >251</td>
      <td id="T_f47c4_row1_col5" class="data row1 col5" >238</td>
      <td id="T_f47c4_row1_col6" class="data row1 col6" >214</td>
      <td id="T_f47c4_row1_col7" class="data row1 col7" >143</td>
      <td id="T_f47c4_row1_col8" class="data row1 col8" >79</td>
      <td id="T_f47c4_row1_col9" class="data row1 col9" >53</td>
      <td id="T_f47c4_row1_col10" class="data row1 col10" >61</td>
      <td id="T_f47c4_row1_col11" class="data row1 col11" >112</td>
      <td id="T_f47c4_row1_col12" class="data row1 col12" >125</td>
      <td id="T_f47c4_row1_col13" class="data row1 col13" >76</td>
      <td id="T_f47c4_row1_col14" class="data row1 col14" >42</td>
      <td id="T_f47c4_row1_col15" class="data row1 col15" >44</td>
      <td id="T_f47c4_row1_col16" class="data row1 col16" >53</td>
      <td id="T_f47c4_row1_col17" class="data row1 col17" >49</td>
      <td id="T_f47c4_row1_col18" class="data row1 col18" >38</td>
      <td id="T_f47c4_row1_col19" class="data row1 col19" >32</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row2" class="row_heading level0 row2" >2</th>
      <td id="T_f47c4_row2_col0" class="data row2 col0" >240</td>
      <td id="T_f47c4_row2_col1" class="data row2 col1" >255</td>
      <td id="T_f47c4_row2_col2" class="data row2 col2" >250</td>
      <td id="T_f47c4_row2_col3" class="data row2 col3" >245</td>
      <td id="T_f47c4_row2_col4" class="data row2 col4" >229</td>
      <td id="T_f47c4_row2_col5" class="data row2 col5" >201</td>
      <td id="T_f47c4_row2_col6" class="data row2 col6" >122</td>
      <td id="T_f47c4_row2_col7" class="data row2 col7" >69</td>
      <td id="T_f47c4_row2_col8" class="data row2 col8" >45</td>
      <td id="T_f47c4_row2_col9" class="data row2 col9" >79</td>
      <td id="T_f47c4_row2_col10" class="data row2 col10" >127</td>
      <td id="T_f47c4_row2_col11" class="data row2 col11" >87</td>
      <td id="T_f47c4_row2_col12" class="data row2 col12" >36</td>
      <td id="T_f47c4_row2_col13" class="data row2 col13" >41</td>
      <td id="T_f47c4_row2_col14" class="data row2 col14" >55</td>
      <td id="T_f47c4_row2_col15" class="data row2 col15" >49</td>
      <td id="T_f47c4_row2_col16" class="data row2 col16" >38</td>
      <td id="T_f47c4_row2_col17" class="data row2 col17" >33</td>
      <td id="T_f47c4_row2_col18" class="data row2 col18" >33</td>
      <td id="T_f47c4_row2_col19" class="data row2 col19" >28</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row3" class="row_heading level0 row3" >3</th>
      <td id="T_f47c4_row3_col0" class="data row3 col0" >213</td>
      <td id="T_f47c4_row3_col1" class="data row3 col1" >230</td>
      <td id="T_f47c4_row3_col2" class="data row3 col2" >243</td>
      <td id="T_f47c4_row3_col3" class="data row3 col3" >234</td>
      <td id="T_f47c4_row3_col4" class="data row3 col4" >193</td>
      <td id="T_f47c4_row3_col5" class="data row3 col5" >104</td>
      <td id="T_f47c4_row3_col6" class="data row3 col6" >66</td>
      <td id="T_f47c4_row3_col7" class="data row3 col7" >46</td>
      <td id="T_f47c4_row3_col8" class="data row3 col8" >92</td>
      <td id="T_f47c4_row3_col9" class="data row3 col9" >112</td>
      <td id="T_f47c4_row3_col10" class="data row3 col10" >59</td>
      <td id="T_f47c4_row3_col11" class="data row3 col11" >84</td>
      <td id="T_f47c4_row3_col12" class="data row3 col12" >137</td>
      <td id="T_f47c4_row3_col13" class="data row3 col13" >95</td>
      <td id="T_f47c4_row3_col14" class="data row3 col14" >38</td>
      <td id="T_f47c4_row3_col15" class="data row3 col15" >33</td>
      <td id="T_f47c4_row3_col16" class="data row3 col16" >33</td>
      <td id="T_f47c4_row3_col17" class="data row3 col17" >30</td>
      <td id="T_f47c4_row3_col18" class="data row3 col18" >24</td>
      <td id="T_f47c4_row3_col19" class="data row3 col19" >17</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row4" class="row_heading level0 row4" >4</th>
      <td id="T_f47c4_row4_col0" class="data row4 col0" >136</td>
      <td id="T_f47c4_row4_col1" class="data row4 col1" >143</td>
      <td id="T_f47c4_row4_col2" class="data row4 col2" >150</td>
      <td id="T_f47c4_row4_col3" class="data row4 col3" >139</td>
      <td id="T_f47c4_row4_col4" class="data row4 col4" >106</td>
      <td id="T_f47c4_row4_col5" class="data row4 col5" >62</td>
      <td id="T_f47c4_row4_col6" class="data row4 col6" >43</td>
      <td id="T_f47c4_row4_col7" class="data row4 col7" >91</td>
      <td id="T_f47c4_row4_col8" class="data row4 col8" >97</td>
      <td id="T_f47c4_row4_col9" class="data row4 col9" >42</td>
      <td id="T_f47c4_row4_col10" class="data row4 col10" >78</td>
      <td id="T_f47c4_row4_col11" class="data row4 col11" >189</td>
      <td id="T_f47c4_row4_col12" class="data row4 col12" >199</td>
      <td id="T_f47c4_row4_col13" class="data row4 col13" >152</td>
      <td id="T_f47c4_row4_col14" class="data row4 col14" >51</td>
      <td id="T_f47c4_row4_col15" class="data row4 col15" >38</td>
      <td id="T_f47c4_row4_col16" class="data row4 col16" >44</td>
      <td id="T_f47c4_row4_col17" class="data row4 col17" >41</td>
      <td id="T_f47c4_row4_col18" class="data row4 col18" >35</td>
      <td id="T_f47c4_row4_col19" class="data row4 col19" >29</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row5" class="row_heading level0 row5" >5</th>
      <td id="T_f47c4_row5_col0" class="data row5 col0" >145</td>
      <td id="T_f47c4_row5_col1" class="data row5 col1" >152</td>
      <td id="T_f47c4_row5_col2" class="data row5 col2" >107</td>
      <td id="T_f47c4_row5_col3" class="data row5 col3" >59</td>
      <td id="T_f47c4_row5_col4" class="data row5 col4" >60</td>
      <td id="T_f47c4_row5_col5" class="data row5 col5" >38</td>
      <td id="T_f47c4_row5_col6" class="data row5 col6" >85</td>
      <td id="T_f47c4_row5_col7" class="data row5 col7" >84</td>
      <td id="T_f47c4_row5_col8" class="data row5 col8" >45</td>
      <td id="T_f47c4_row5_col9" class="data row5 col9" >51</td>
      <td id="T_f47c4_row5_col10" class="data row5 col10" >128</td>
      <td id="T_f47c4_row5_col11" class="data row5 col11" >159</td>
      <td id="T_f47c4_row5_col12" class="data row5 col12" >144</td>
      <td id="T_f47c4_row5_col13" class="data row5 col13" >106</td>
      <td id="T_f47c4_row5_col14" class="data row5 col14" >47</td>
      <td id="T_f47c4_row5_col15" class="data row5 col15" >48</td>
      <td id="T_f47c4_row5_col16" class="data row5 col16" >60</td>
      <td id="T_f47c4_row5_col17" class="data row5 col17" >66</td>
      <td id="T_f47c4_row5_col18" class="data row5 col18" >68</td>
      <td id="T_f47c4_row5_col19" class="data row5 col19" >61</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row6" class="row_heading level0 row6" >6</th>
      <td id="T_f47c4_row6_col0" class="data row6 col0" >148</td>
      <td id="T_f47c4_row6_col1" class="data row6 col1" >122</td>
      <td id="T_f47c4_row6_col2" class="data row6 col2" >59</td>
      <td id="T_f47c4_row6_col3" class="data row6 col3" >59</td>
      <td id="T_f47c4_row6_col4" class="data row6 col4" >39</td>
      <td id="T_f47c4_row6_col5" class="data row6 col5" >74</td>
      <td id="T_f47c4_row6_col6" class="data row6 col6" >85</td>
      <td id="T_f47c4_row6_col7" class="data row6 col7" >45</td>
      <td id="T_f47c4_row6_col8" class="data row6 col8" >50</td>
      <td id="T_f47c4_row6_col9" class="data row6 col9" >65</td>
      <td id="T_f47c4_row6_col10" class="data row6 col10" >135</td>
      <td id="T_f47c4_row6_col11" class="data row6 col11" >129</td>
      <td id="T_f47c4_row6_col12" class="data row6 col12" >115</td>
      <td id="T_f47c4_row6_col13" class="data row6 col13" >72</td>
      <td id="T_f47c4_row6_col14" class="data row6 col14" >51</td>
      <td id="T_f47c4_row6_col15" class="data row6 col15" >58</td>
      <td id="T_f47c4_row6_col16" class="data row6 col16" >68</td>
      <td id="T_f47c4_row6_col17" class="data row6 col17" >79</td>
      <td id="T_f47c4_row6_col18" class="data row6 col18" >82</td>
      <td id="T_f47c4_row6_col19" class="data row6 col19" >84</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row7" class="row_heading level0 row7" >7</th>
      <td id="T_f47c4_row7_col0" class="data row7 col0" >139</td>
      <td id="T_f47c4_row7_col1" class="data row7 col1" >73</td>
      <td id="T_f47c4_row7_col2" class="data row7 col2" >64</td>
      <td id="T_f47c4_row7_col3" class="data row7 col3" >41</td>
      <td id="T_f47c4_row7_col4" class="data row7 col4" >62</td>
      <td id="T_f47c4_row7_col5" class="data row7 col5" >83</td>
      <td id="T_f47c4_row7_col6" class="data row7 col6" >44</td>
      <td id="T_f47c4_row7_col7" class="data row7 col7" >50</td>
      <td id="T_f47c4_row7_col8" class="data row7 col8" >60</td>
      <td id="T_f47c4_row7_col9" class="data row7 col9" >105</td>
      <td id="T_f47c4_row7_col10" class="data row7 col10" >162</td>
      <td id="T_f47c4_row7_col11" class="data row7 col11" >134</td>
      <td id="T_f47c4_row7_col12" class="data row7 col12" >115</td>
      <td id="T_f47c4_row7_col13" class="data row7 col13" >64</td>
      <td id="T_f47c4_row7_col14" class="data row7 col14" >63</td>
      <td id="T_f47c4_row7_col15" class="data row7 col15" >72</td>
      <td id="T_f47c4_row7_col16" class="data row7 col16" >83</td>
      <td id="T_f47c4_row7_col17" class="data row7 col17" >96</td>
      <td id="T_f47c4_row7_col18" class="data row7 col18" >98</td>
      <td id="T_f47c4_row7_col19" class="data row7 col19" >94</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row8" class="row_heading level0 row8" >8</th>
      <td id="T_f47c4_row8_col0" class="data row8 col0" >92</td>
      <td id="T_f47c4_row8_col1" class="data row8 col1" >59</td>
      <td id="T_f47c4_row8_col2" class="data row8 col2" >39</td>
      <td id="T_f47c4_row8_col3" class="data row8 col3" >48</td>
      <td id="T_f47c4_row8_col4" class="data row8 col4" >95</td>
      <td id="T_f47c4_row8_col5" class="data row8 col5" >68</td>
      <td id="T_f47c4_row8_col6" class="data row8 col6" >83</td>
      <td id="T_f47c4_row8_col7" class="data row8 col7" >131</td>
      <td id="T_f47c4_row8_col8" class="data row8 col8" >169</td>
      <td id="T_f47c4_row8_col9" class="data row8 col9" >179</td>
      <td id="T_f47c4_row8_col10" class="data row8 col10" >159</td>
      <td id="T_f47c4_row8_col11" class="data row8 col11" >144</td>
      <td id="T_f47c4_row8_col12" class="data row8 col12" >103</td>
      <td id="T_f47c4_row8_col13" class="data row8 col13" >52</td>
      <td id="T_f47c4_row8_col14" class="data row8 col14" >66</td>
      <td id="T_f47c4_row8_col15" class="data row8 col15" >73</td>
      <td id="T_f47c4_row8_col16" class="data row8 col16" >79</td>
      <td id="T_f47c4_row8_col17" class="data row8 col17" >88</td>
      <td id="T_f47c4_row8_col18" class="data row8 col18" >97</td>
      <td id="T_f47c4_row8_col19" class="data row8 col19" >103</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row9" class="row_heading level0 row9" >9</th>
      <td id="T_f47c4_row9_col0" class="data row9 col0" >60</td>
      <td id="T_f47c4_row9_col1" class="data row9 col1" >43</td>
      <td id="T_f47c4_row9_col2" class="data row9 col2" >34</td>
      <td id="T_f47c4_row9_col3" class="data row9 col3" >100</td>
      <td id="T_f47c4_row9_col4" class="data row9 col4" >140</td>
      <td id="T_f47c4_row9_col5" class="data row9 col5" >172</td>
      <td id="T_f47c4_row9_col6" class="data row9 col6" >179</td>
      <td id="T_f47c4_row9_col7" class="data row9 col7" >180</td>
      <td id="T_f47c4_row9_col8" class="data row9 col8" >154</td>
      <td id="T_f47c4_row9_col9" class="data row9 col9" >163</td>
      <td id="T_f47c4_row9_col10" class="data row9 col10" >143</td>
      <td id="T_f47c4_row9_col11" class="data row9 col11" >128</td>
      <td id="T_f47c4_row9_col12" class="data row9 col12" >77</td>
      <td id="T_f47c4_row9_col13" class="data row9 col13" >48</td>
      <td id="T_f47c4_row9_col14" class="data row9 col14" >59</td>
      <td id="T_f47c4_row9_col15" class="data row9 col15" >63</td>
      <td id="T_f47c4_row9_col16" class="data row9 col16" >56</td>
      <td id="T_f47c4_row9_col17" class="data row9 col17" >46</td>
      <td id="T_f47c4_row9_col18" class="data row9 col18" >45</td>
      <td id="T_f47c4_row9_col19" class="data row9 col19" >54</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row10" class="row_heading level0 row10" >10</th>
      <td id="T_f47c4_row10_col0" class="data row10 col0" >46</td>
      <td id="T_f47c4_row10_col1" class="data row10 col1" >28</td>
      <td id="T_f47c4_row10_col2" class="data row10 col2" >86</td>
      <td id="T_f47c4_row10_col3" class="data row10 col3" >96</td>
      <td id="T_f47c4_row10_col4" class="data row10 col4" >156</td>
      <td id="T_f47c4_row10_col5" class="data row10 col5" >152</td>
      <td id="T_f47c4_row10_col6" class="data row10 col6" >139</td>
      <td id="T_f47c4_row10_col7" class="data row10 col7" >135</td>
      <td id="T_f47c4_row10_col8" class="data row10 col8" >146</td>
      <td id="T_f47c4_row10_col9" class="data row10 col9" >138</td>
      <td id="T_f47c4_row10_col10" class="data row10 col10" >120</td>
      <td id="T_f47c4_row10_col11" class="data row10 col11" >107</td>
      <td id="T_f47c4_row10_col12" class="data row10 col12" >116</td>
      <td id="T_f47c4_row10_col13" class="data row10 col13" >179</td>
      <td id="T_f47c4_row10_col14" class="data row10 col14" >166</td>
      <td id="T_f47c4_row10_col15" class="data row10 col15" >96</td>
      <td id="T_f47c4_row10_col16" class="data row10 col16" >40</td>
      <td id="T_f47c4_row10_col17" class="data row10 col17" >31</td>
      <td id="T_f47c4_row10_col18" class="data row10 col18" >35</td>
      <td id="T_f47c4_row10_col19" class="data row10 col19" >44</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row11" class="row_heading level0 row11" >11</th>
      <td id="T_f47c4_row11_col0" class="data row11 col0" >25</td>
      <td id="T_f47c4_row11_col1" class="data row11 col1" >68</td>
      <td id="T_f47c4_row11_col2" class="data row11 col2" >95</td>
      <td id="T_f47c4_row11_col3" class="data row11 col3" >81</td>
      <td id="T_f47c4_row11_col4" class="data row11 col4" >156</td>
      <td id="T_f47c4_row11_col5" class="data row11 col5" >142</td>
      <td id="T_f47c4_row11_col6" class="data row11 col6" >133</td>
      <td id="T_f47c4_row11_col7" class="data row11 col7" >122</td>
      <td id="T_f47c4_row11_col8" class="data row11 col8" >137</td>
      <td id="T_f47c4_row11_col9" class="data row11 col9" >119</td>
      <td id="T_f47c4_row11_col10" class="data row11 col10" >104</td>
      <td id="T_f47c4_row11_col11" class="data row11 col11" >95</td>
      <td id="T_f47c4_row11_col12" class="data row11 col12" >128</td>
      <td id="T_f47c4_row11_col13" class="data row11 col13" >159</td>
      <td id="T_f47c4_row11_col14" class="data row11 col14" >155</td>
      <td id="T_f47c4_row11_col15" class="data row11 col15" >140</td>
      <td id="T_f47c4_row11_col16" class="data row11 col16" >83</td>
      <td id="T_f47c4_row11_col17" class="data row11 col17" >69</td>
      <td id="T_f47c4_row11_col18" class="data row11 col18" >78</td>
      <td id="T_f47c4_row11_col19" class="data row11 col19" >85</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row12" class="row_heading level0 row12" >12</th>
      <td id="T_f47c4_row12_col0" class="data row12 col0" >43</td>
      <td id="T_f47c4_row12_col1" class="data row12 col1" >115</td>
      <td id="T_f47c4_row12_col2" class="data row12 col2" >52</td>
      <td id="T_f47c4_row12_col3" class="data row12 col3" >107</td>
      <td id="T_f47c4_row12_col4" class="data row12 col4" >148</td>
      <td id="T_f47c4_row12_col5" class="data row12 col5" >136</td>
      <td id="T_f47c4_row12_col6" class="data row12 col6" >123</td>
      <td id="T_f47c4_row12_col7" class="data row12 col7" >108</td>
      <td id="T_f47c4_row12_col8" class="data row12 col8" >141</td>
      <td id="T_f47c4_row12_col9" class="data row12 col9" >120</td>
      <td id="T_f47c4_row12_col10" class="data row12 col10" >105</td>
      <td id="T_f47c4_row12_col11" class="data row12 col11" >82</td>
      <td id="T_f47c4_row12_col12" class="data row12 col12" >110</td>
      <td id="T_f47c4_row12_col13" class="data row12 col13" >133</td>
      <td id="T_f47c4_row12_col14" class="data row12 col14" >124</td>
      <td id="T_f47c4_row12_col15" class="data row12 col15" >126</td>
      <td id="T_f47c4_row12_col16" class="data row12 col16" >162</td>
      <td id="T_f47c4_row12_col17" class="data row12 col17" >126</td>
      <td id="T_f47c4_row12_col18" class="data row12 col18" >112</td>
      <td id="T_f47c4_row12_col19" class="data row12 col19" >123</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row13" class="row_heading level0 row13" >13</th>
      <td id="T_f47c4_row13_col0" class="data row13 col0" >104</td>
      <td id="T_f47c4_row13_col1" class="data row13 col1" >75</td>
      <td id="T_f47c4_row13_col2" class="data row13 col2" >67</td>
      <td id="T_f47c4_row13_col3" class="data row13 col3" >145</td>
      <td id="T_f47c4_row13_col4" class="data row13 col4" >140</td>
      <td id="T_f47c4_row13_col5" class="data row13 col5" >136</td>
      <td id="T_f47c4_row13_col6" class="data row13 col6" >117</td>
      <td id="T_f47c4_row13_col7" class="data row13 col7" >105</td>
      <td id="T_f47c4_row13_col8" class="data row13 col8" >134</td>
      <td id="T_f47c4_row13_col9" class="data row13 col9" >115</td>
      <td id="T_f47c4_row13_col10" class="data row13 col10" >100</td>
      <td id="T_f47c4_row13_col11" class="data row13 col11" >92</td>
      <td id="T_f47c4_row13_col12" class="data row13 col12" >115</td>
      <td id="T_f47c4_row13_col13" class="data row13 col13" >123</td>
      <td id="T_f47c4_row13_col14" class="data row13 col14" >119</td>
      <td id="T_f47c4_row13_col15" class="data row13 col15" >108</td>
      <td id="T_f47c4_row13_col16" class="data row13 col16" >140</td>
      <td id="T_f47c4_row13_col17" class="data row13 col17" >135</td>
      <td id="T_f47c4_row13_col18" class="data row13 col18" >90</td>
      <td id="T_f47c4_row13_col19" class="data row13 col19" >95</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row14" class="row_heading level0 row14" >14</th>
      <td id="T_f47c4_row14_col0" class="data row14 col0" >111</td>
      <td id="T_f47c4_row14_col1" class="data row14 col1" >42</td>
      <td id="T_f47c4_row14_col2" class="data row14 col2" >70</td>
      <td id="T_f47c4_row14_col3" class="data row14 col3" >145</td>
      <td id="T_f47c4_row14_col4" class="data row14 col4" >144</td>
      <td id="T_f47c4_row14_col5" class="data row14 col5" >134</td>
      <td id="T_f47c4_row14_col6" class="data row14 col6" >124</td>
      <td id="T_f47c4_row14_col7" class="data row14 col7" >127</td>
      <td id="T_f47c4_row14_col8" class="data row14 col8" >132</td>
      <td id="T_f47c4_row14_col9" class="data row14 col9" >109</td>
      <td id="T_f47c4_row14_col10" class="data row14 col10" >96</td>
      <td id="T_f47c4_row14_col11" class="data row14 col11" >113</td>
      <td id="T_f47c4_row14_col12" class="data row14 col12" >111</td>
      <td id="T_f47c4_row14_col13" class="data row14 col13" >116</td>
      <td id="T_f47c4_row14_col14" class="data row14 col14" >112</td>
      <td id="T_f47c4_row14_col15" class="data row14 col15" >105</td>
      <td id="T_f47c4_row14_col16" class="data row14 col16" >132</td>
      <td id="T_f47c4_row14_col17" class="data row14 col17" >125</td>
      <td id="T_f47c4_row14_col18" class="data row14 col18" >68</td>
      <td id="T_f47c4_row14_col19" class="data row14 col19" >68</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row15" class="row_heading level0 row15" >15</th>
      <td id="T_f47c4_row15_col0" class="data row15 col0" >59</td>
      <td id="T_f47c4_row15_col1" class="data row15 col1" >43</td>
      <td id="T_f47c4_row15_col2" class="data row15 col2" >64</td>
      <td id="T_f47c4_row15_col3" class="data row15 col3" >135</td>
      <td id="T_f47c4_row15_col4" class="data row15 col4" >140</td>
      <td id="T_f47c4_row15_col5" class="data row15 col5" >132</td>
      <td id="T_f47c4_row15_col6" class="data row15 col6" >131</td>
      <td id="T_f47c4_row15_col7" class="data row15 col7" >137</td>
      <td id="T_f47c4_row15_col8" class="data row15 col8" >132</td>
      <td id="T_f47c4_row15_col9" class="data row15 col9" >109</td>
      <td id="T_f47c4_row15_col10" class="data row15 col10" >97</td>
      <td id="T_f47c4_row15_col11" class="data row15 col11" >110</td>
      <td id="T_f47c4_row15_col12" class="data row15 col12" >118</td>
      <td id="T_f47c4_row15_col13" class="data row15 col13" >120</td>
      <td id="T_f47c4_row15_col14" class="data row15 col14" >112</td>
      <td id="T_f47c4_row15_col15" class="data row15 col15" >113</td>
      <td id="T_f47c4_row15_col16" class="data row15 col16" >130</td>
      <td id="T_f47c4_row15_col17" class="data row15 col17" >116</td>
      <td id="T_f47c4_row15_col18" class="data row15 col18" >58</td>
      <td id="T_f47c4_row15_col19" class="data row15 col19" >52</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row16" class="row_heading level0 row16" >16</th>
      <td id="T_f47c4_row16_col0" class="data row16 col0" >36</td>
      <td id="T_f47c4_row16_col1" class="data row16 col1" >37</td>
      <td id="T_f47c4_row16_col2" class="data row16 col2" >58</td>
      <td id="T_f47c4_row16_col3" class="data row16 col3" >118</td>
      <td id="T_f47c4_row16_col4" class="data row16 col4" >121</td>
      <td id="T_f47c4_row16_col5" class="data row16 col5" >115</td>
      <td id="T_f47c4_row16_col6" class="data row16 col6" >117</td>
      <td id="T_f47c4_row16_col7" class="data row16 col7" >132</td>
      <td id="T_f47c4_row16_col8" class="data row16 col8" >131</td>
      <td id="T_f47c4_row16_col9" class="data row16 col9" >113</td>
      <td id="T_f47c4_row16_col10" class="data row16 col10" >103</td>
      <td id="T_f47c4_row16_col11" class="data row16 col11" >111</td>
      <td id="T_f47c4_row16_col12" class="data row16 col12" >128</td>
      <td id="T_f47c4_row16_col13" class="data row16 col13" >121</td>
      <td id="T_f47c4_row16_col14" class="data row16 col14" >114</td>
      <td id="T_f47c4_row16_col15" class="data row16 col15" >126</td>
      <td id="T_f47c4_row16_col16" class="data row16 col16" >125</td>
      <td id="T_f47c4_row16_col17" class="data row16 col17" >112</td>
      <td id="T_f47c4_row16_col18" class="data row16 col18" >74</td>
      <td id="T_f47c4_row16_col19" class="data row16 col19" >52</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row17" class="row_heading level0 row17" >17</th>
      <td id="T_f47c4_row17_col0" class="data row17 col0" >40</td>
      <td id="T_f47c4_row17_col1" class="data row17 col1" >35</td>
      <td id="T_f47c4_row17_col2" class="data row17 col2" >47</td>
      <td id="T_f47c4_row17_col3" class="data row17 col3" >96</td>
      <td id="T_f47c4_row17_col4" class="data row17 col4" >87</td>
      <td id="T_f47c4_row17_col5" class="data row17 col5" >86</td>
      <td id="T_f47c4_row17_col6" class="data row17 col6" >98</td>
      <td id="T_f47c4_row17_col7" class="data row17 col7" >121</td>
      <td id="T_f47c4_row17_col8" class="data row17 col8" >117</td>
      <td id="T_f47c4_row17_col9" class="data row17 col9" >105</td>
      <td id="T_f47c4_row17_col10" class="data row17 col10" >101</td>
      <td id="T_f47c4_row17_col11" class="data row17 col11" >105</td>
      <td id="T_f47c4_row17_col12" class="data row17 col12" >125</td>
      <td id="T_f47c4_row17_col13" class="data row17 col13" >116</td>
      <td id="T_f47c4_row17_col14" class="data row17 col14" >106</td>
      <td id="T_f47c4_row17_col15" class="data row17 col15" >115</td>
      <td id="T_f47c4_row17_col16" class="data row17 col16" >117</td>
      <td id="T_f47c4_row17_col17" class="data row17 col17" >101</td>
      <td id="T_f47c4_row17_col18" class="data row17 col18" >62</td>
      <td id="T_f47c4_row17_col19" class="data row17 col19" >42</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row18" class="row_heading level0 row18" >18</th>
      <td id="T_f47c4_row18_col0" class="data row18 col0" >37</td>
      <td id="T_f47c4_row18_col1" class="data row18 col1" >34</td>
      <td id="T_f47c4_row18_col2" class="data row18 col2" >37</td>
      <td id="T_f47c4_row18_col3" class="data row18 col3" >85</td>
      <td id="T_f47c4_row18_col4" class="data row18 col4" >82</td>
      <td id="T_f47c4_row18_col5" class="data row18 col5" >79</td>
      <td id="T_f47c4_row18_col6" class="data row18 col6" >82</td>
      <td id="T_f47c4_row18_col7" class="data row18 col7" >88</td>
      <td id="T_f47c4_row18_col8" class="data row18 col8" >78</td>
      <td id="T_f47c4_row18_col9" class="data row18 col9" >75</td>
      <td id="T_f47c4_row18_col10" class="data row18 col10" >73</td>
      <td id="T_f47c4_row18_col11" class="data row18 col11" >75</td>
      <td id="T_f47c4_row18_col12" class="data row18 col12" >83</td>
      <td id="T_f47c4_row18_col13" class="data row18 col13" >79</td>
      <td id="T_f47c4_row18_col14" class="data row18 col14" >72</td>
      <td id="T_f47c4_row18_col15" class="data row18 col15" >76</td>
      <td id="T_f47c4_row18_col16" class="data row18 col16" >82</td>
      <td id="T_f47c4_row18_col17" class="data row18 col17" >81</td>
      <td id="T_f47c4_row18_col18" class="data row18 col18" >43</td>
      <td id="T_f47c4_row18_col19" class="data row18 col19" >31</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row19" class="row_heading level0 row19" >19</th>
      <td id="T_f47c4_row19_col0" class="data row19 col0" >30</td>
      <td id="T_f47c4_row19_col1" class="data row19 col1" >27</td>
      <td id="T_f47c4_row19_col2" class="data row19 col2" >33</td>
      <td id="T_f47c4_row19_col3" class="data row19 col3" >70</td>
      <td id="T_f47c4_row19_col4" class="data row19 col4" >88</td>
      <td id="T_f47c4_row19_col5" class="data row19 col5" >79</td>
      <td id="T_f47c4_row19_col6" class="data row19 col6" >71</td>
      <td id="T_f47c4_row19_col7" class="data row19 col7" >72</td>
      <td id="T_f47c4_row19_col8" class="data row19 col8" >64</td>
      <td id="T_f47c4_row19_col9" class="data row19 col9" >61</td>
      <td id="T_f47c4_row19_col10" class="data row19 col10" >63</td>
      <td id="T_f47c4_row19_col11" class="data row19 col11" >57</td>
      <td id="T_f47c4_row19_col12" class="data row19 col12" >59</td>
      <td id="T_f47c4_row19_col13" class="data row19 col13" >59</td>
      <td id="T_f47c4_row19_col14" class="data row19 col14" >56</td>
      <td id="T_f47c4_row19_col15" class="data row19 col15" >57</td>
      <td id="T_f47c4_row19_col16" class="data row19 col16" >61</td>
      <td id="T_f47c4_row19_col17" class="data row19 col17" >74</td>
      <td id="T_f47c4_row19_col18" class="data row19 col18" >47</td>
      <td id="T_f47c4_row19_col19" class="data row19 col19" >29</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row20" class="row_heading level0 row20" >20</th>
      <td id="T_f47c4_row20_col0" class="data row20 col0" >26</td>
      <td id="T_f47c4_row20_col1" class="data row20 col1" >19</td>
      <td id="T_f47c4_row20_col2" class="data row20 col2" >40</td>
      <td id="T_f47c4_row20_col3" class="data row20 col3" >73</td>
      <td id="T_f47c4_row20_col4" class="data row20 col4" >93</td>
      <td id="T_f47c4_row20_col5" class="data row20 col5" >81</td>
      <td id="T_f47c4_row20_col6" class="data row20 col6" >76</td>
      <td id="T_f47c4_row20_col7" class="data row20 col7" >72</td>
      <td id="T_f47c4_row20_col8" class="data row20 col8" >63</td>
      <td id="T_f47c4_row20_col9" class="data row20 col9" >61</td>
      <td id="T_f47c4_row20_col10" class="data row20 col10" >66</td>
      <td id="T_f47c4_row20_col11" class="data row20 col11" >59</td>
      <td id="T_f47c4_row20_col12" class="data row20 col12" >59</td>
      <td id="T_f47c4_row20_col13" class="data row20 col13" >61</td>
      <td id="T_f47c4_row20_col14" class="data row20 col14" >64</td>
      <td id="T_f47c4_row20_col15" class="data row20 col15" >67</td>
      <td id="T_f47c4_row20_col16" class="data row20 col16" >68</td>
      <td id="T_f47c4_row20_col17" class="data row20 col17" >62</td>
      <td id="T_f47c4_row20_col18" class="data row20 col18" >43</td>
      <td id="T_f47c4_row20_col19" class="data row20 col19" >38</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row21" class="row_heading level0 row21" >21</th>
      <td id="T_f47c4_row21_col0" class="data row21 col0" >25</td>
      <td id="T_f47c4_row21_col1" class="data row21 col1" >35</td>
      <td id="T_f47c4_row21_col2" class="data row21 col2" >61</td>
      <td id="T_f47c4_row21_col3" class="data row21 col3" >41</td>
      <td id="T_f47c4_row21_col4" class="data row21 col4" >101</td>
      <td id="T_f47c4_row21_col5" class="data row21 col5" >94</td>
      <td id="T_f47c4_row21_col6" class="data row21 col6" >82</td>
      <td id="T_f47c4_row21_col7" class="data row21 col7" >74</td>
      <td id="T_f47c4_row21_col8" class="data row21 col8" >66</td>
      <td id="T_f47c4_row21_col9" class="data row21 col9" >68</td>
      <td id="T_f47c4_row21_col10" class="data row21 col10" >66</td>
      <td id="T_f47c4_row21_col11" class="data row21 col11" >65</td>
      <td id="T_f47c4_row21_col12" class="data row21 col12" >64</td>
      <td id="T_f47c4_row21_col13" class="data row21 col13" >65</td>
      <td id="T_f47c4_row21_col14" class="data row21 col14" >70</td>
      <td id="T_f47c4_row21_col15" class="data row21 col15" >68</td>
      <td id="T_f47c4_row21_col16" class="data row21 col16" >64</td>
      <td id="T_f47c4_row21_col17" class="data row21 col17" >39</td>
      <td id="T_f47c4_row21_col18" class="data row21 col18" >34</td>
      <td id="T_f47c4_row21_col19" class="data row21 col19" >41</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row22" class="row_heading level0 row22" >22</th>
      <td id="T_f47c4_row22_col0" class="data row22 col0" >33</td>
      <td id="T_f47c4_row22_col1" class="data row22 col1" >66</td>
      <td id="T_f47c4_row22_col2" class="data row22 col2" >33</td>
      <td id="T_f47c4_row22_col3" class="data row22 col3" >52</td>
      <td id="T_f47c4_row22_col4" class="data row22 col4" >111</td>
      <td id="T_f47c4_row22_col5" class="data row22 col5" >98</td>
      <td id="T_f47c4_row22_col6" class="data row22 col6" >86</td>
      <td id="T_f47c4_row22_col7" class="data row22 col7" >78</td>
      <td id="T_f47c4_row22_col8" class="data row22 col8" >71</td>
      <td id="T_f47c4_row22_col9" class="data row22 col9" >69</td>
      <td id="T_f47c4_row22_col10" class="data row22 col10" >66</td>
      <td id="T_f47c4_row22_col11" class="data row22 col11" >67</td>
      <td id="T_f47c4_row22_col12" class="data row22 col12" >67</td>
      <td id="T_f47c4_row22_col13" class="data row22 col13" >68</td>
      <td id="T_f47c4_row22_col14" class="data row22 col14" >68</td>
      <td id="T_f47c4_row22_col15" class="data row22 col15" >63</td>
      <td id="T_f47c4_row22_col16" class="data row22 col16" >40</td>
      <td id="T_f47c4_row22_col17" class="data row22 col17" >28</td>
      <td id="T_f47c4_row22_col18" class="data row22 col18" >37</td>
      <td id="T_f47c4_row22_col19" class="data row22 col19" >34</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row23" class="row_heading level0 row23" >23</th>
      <td id="T_f47c4_row23_col0" class="data row23 col0" >108</td>
      <td id="T_f47c4_row23_col1" class="data row23 col1" >108</td>
      <td id="T_f47c4_row23_col2" class="data row23 col2" >28</td>
      <td id="T_f47c4_row23_col3" class="data row23 col3" >51</td>
      <td id="T_f47c4_row23_col4" class="data row23 col4" >99</td>
      <td id="T_f47c4_row23_col5" class="data row23 col5" >103</td>
      <td id="T_f47c4_row23_col6" class="data row23 col6" >88</td>
      <td id="T_f47c4_row23_col7" class="data row23 col7" >76</td>
      <td id="T_f47c4_row23_col8" class="data row23 col8" >70</td>
      <td id="T_f47c4_row23_col9" class="data row23 col9" >67</td>
      <td id="T_f47c4_row23_col10" class="data row23 col10" >68</td>
      <td id="T_f47c4_row23_col11" class="data row23 col11" >72</td>
      <td id="T_f47c4_row23_col12" class="data row23 col12" >70</td>
      <td id="T_f47c4_row23_col13" class="data row23 col13" >64</td>
      <td id="T_f47c4_row23_col14" class="data row23 col14" >59</td>
      <td id="T_f47c4_row23_col15" class="data row23 col15" >54</td>
      <td id="T_f47c4_row23_col16" class="data row23 col16" >25</td>
      <td id="T_f47c4_row23_col17" class="data row23 col17" >17</td>
      <td id="T_f47c4_row23_col18" class="data row23 col18" >33</td>
      <td id="T_f47c4_row23_col19" class="data row23 col19" >36</td>
    </tr>
    <tr>
      <th id="T_f47c4_level0_row24" class="row_heading level0 row24" >24</th>
      <td id="T_f47c4_row24_col0" class="data row24 col0" >103</td>
      <td id="T_f47c4_row24_col1" class="data row24 col1" >97</td>
      <td id="T_f47c4_row24_col2" class="data row24 col2" >60</td>
      <td id="T_f47c4_row24_col3" class="data row24 col3" >61</td>
      <td id="T_f47c4_row24_col4" class="data row24 col4" >84</td>
      <td id="T_f47c4_row24_col5" class="data row24 col5" >89</td>
      <td id="T_f47c4_row24_col6" class="data row24 col6" >85</td>
      <td id="T_f47c4_row24_col7" class="data row24 col7" >82</td>
      <td id="T_f47c4_row24_col8" class="data row24 col8" >79</td>
      <td id="T_f47c4_row24_col9" class="data row24 col9" >80</td>
      <td id="T_f47c4_row24_col10" class="data row24 col10" >83</td>
      <td id="T_f47c4_row24_col11" class="data row24 col11" >88</td>
      <td id="T_f47c4_row24_col12" class="data row24 col12" >99</td>
      <td id="T_f47c4_row24_col13" class="data row24 col13" >101</td>
      <td id="T_f47c4_row24_col14" class="data row24 col14" >94</td>
      <td id="T_f47c4_row24_col15" class="data row24 col15" >98</td>
      <td id="T_f47c4_row24_col16" class="data row24 col16" >87</td>
      <td id="T_f47c4_row24_col17" class="data row24 col17" >75</td>
      <td id="T_f47c4_row24_col18" class="data row24 col18" >85</td>
      <td id="T_f47c4_row24_col19" class="data row24 col19" >90</td>
    </tr>
  </tbody>
</table>




Here there are 2 options:

- take the average across the three colours for each image
- use data for every single pixel from all three colours for each image

Since I am simply looking for the same occurence of the same sub image within each image, I opted for taking the average across all three colours. This would save on training time and computing power needed.

Some operations in PyTorch, like taking the mean, require us to _cast_ our interger types to float types. We can do that here too.

>Casting in PyTorch is as simple as typing the name of the type you wish to cast to, and treating it as a method (in this case float)


```python
def resizeImageAndGetMeanAcrossAllColours(img):
    resized = tensor(Image.open(img).resize((128,128)))
    return resized.float().mean(2)

negative_tensors = [resizeImageAndGetMeanAcrossAllColours(o) for o in training_negatives]
positive_tensors = [resizeImageAndGetMeanAcrossAllColours(o) for o in training_positives]

negative_tensors[0].shape,positive_tensors[0].shape,len(negative_tensors),len(positive_tensors)
```




    (torch.Size([128, 128]), torch.Size([128, 128]), 10, 10)



`negative_tensors` and `positive_tensors` are currently just lists of tensors, made by using a [list comprehension](https://docs.python.org/3/tutorial/datastructures.html#list-comprehensions). We will now create a single rank-3 tensor out of each list of tensors by 'stacking' each item within each list. 

Generally, when images are floats, the pixel values are expected to be between 0 and 1, so I divide by 255 here (the highest value that any individual pixel can have)


```python
stacked_negatives = torch.stack(negative_tensors)/255
stacked_positives = torch.stack(positive_tensors)/255

stacked_positives.shape,stacked_negatives.shape
```




    (torch.Size([10, 128, 128]), torch.Size([10, 128, 128]))



We can now begin to get our data ready to load.

We will concatenate thew negative and positive tensors. then we use `view` to change the shape of the tensor without changing its' contents. We want a list of vectors (a rank-2 tensor) instead of a list of matrices (a rank-3 tensor). The `-1`, passed to `view`, tells it to "make that axis as big as neccessary" in order to fit all the data.


```python
train_x = torch.cat([stacked_negatives, stacked_positives]).view(-1,128*128)

train_x.shape
```




    torch.Size([20, 16384])



We will also need a label for each image, will use 0 for negatives and 1 for positives.

Note that we use `unsqueeze` to insert a dimension of size 1 at the _specified position_. example from the docs:

    x = torch.tensor([1, 2, 3, 4])

    torch.unsqueeze(x, 0)
    tensor([[ 1,  2,  3,  4]])

    torch.unsqueeze(x, 1)
    tensor([[ 1],
            [ 2],
            [ 3],
            [ 4]])


This is so that both `train_x` and `train_y` have a shape that corresponds to each other.


```python
train_y = tensor([0]*len(stacked_negatives) + [1]*len(stacked_positives)).unsqueeze(1)

train_x.shape, train_y.shape, tensor([0]*len(stacked_negatives) + [1]*len(stacked_positives)).shape
```




    (torch.Size([20, 16384]), torch.Size([20, 1]), torch.Size([20]))



For fastai, a Dataset needs to return a tuple of independent and dependent variable (x,y), when indexed.

Python's `zip` combined with `list` provides, a simple way to get this functionality


```python
training_dset = list(zip(train_x,train_y))

x,y = training_dset[0]
x.shape,y
```




    (torch.Size([16384]), tensor([0]))



So now we have a training data set, lets create a validation data set as well.


```python
validation_negatives = get_image_files(path/'validation/negative')
validation_positives = get_image_files(path/'validation/positive')

valid_negative_tensors = [resizeImageAndGetMeanAcrossAllColours(o) for o in validation_negatives]
valid_positive_tensors = [resizeImageAndGetMeanAcrossAllColours(o) for o in validation_positives]

stacked_valid_negatives = torch.stack(valid_negative_tensors)/255
stacked_valid_positives = torch.stack(valid_positive_tensors)/255

valid_x = torch.cat([stacked_valid_negatives, stacked_valid_positives]).view(-1,128*128)
valid_y = tensor([0]*len(stacked_valid_negatives) + [1]*len(stacked_valid_positives)).unsqueeze(1)

valid_dset = list(zip(valid_x,valid_y))

x,y = valid_dset[-1]
x.shape,y
```




    (torch.Size([16384]), tensor([1]))



Datasets are fed in to DataLoader in order to create a collection of mini batches.


```python
training_dl = DataLoader(training_dset, batch_size=2, shuffle=True)
valid_dl = DataLoader(valid_dset, batch_size=2, shuffle=True)

list(training_dl)[0]
```




    (tensor([[0.4706, 0.5137, 0.6092,  ..., 0.5804, 0.5804, 0.5856],
             [0.5830, 0.5895, 0.5974,  ..., 0.2902, 0.3007, 0.5137]]),
     tensor([[1],
             [0]]))



We can now use `DataLoaders` as a wrapper for our training and validation loaders. We do this as this is the format we need them in in order to pass then to fastai's Learner (see further below).


```python
dls = DataLoaders(training_dl,valid_dl)
```

### Sidebar

Lets first see how we would do predictions if we were using a simple linear model.

Note that params are initialised using torch.Tensor.requires_grad_(). this tells PyTorch that we will want gradients to be calculated with respect to these params.

Weights, and bias, will also be initialised as random values, to be altered when training.


```python
def init_params(size, multiplier=1.0): return (torch.randn(size)*multiplier).requires_grad_()

weights = init_params(128*128,1)
bias = init_params(1)
```

We can use `@` operator for multiplying each vector in xb matrix by weights, rather than doing a `for` loop over the matrix (as this is very slow).


```python
def linear_model(xb): return xb@weights + bias
```


```python
preds = linear_model(train_x)
preds
```




    tensor([-17.7085, -23.1593,  39.8399,  23.0836, -14.8054, -15.5319,  -8.1738,
            -17.4093, -28.8678,  -3.7090, -28.3204,  16.7366,  -4.1800, -21.9037,
             -8.2536, -12.8513,  -9.2549, -13.8841,  15.4846, -16.7070],
           grad_fn=<AddBackward0>)



### End sidebar
## Creating a Model
We know our model is likely to be more complex than a single linear function. We also know that a single nonlinearity with two linear layers is enough to approximate any function (it can be mathematically proven that such a setup can solve any computable problem to an arbitrarily high level of accuracy). For now we will do just the bare minimum and create a model as such.

without using PyTorch or fastai, we would perhaps create a model like so:
>Note that `torch.max` will directly return anything above the value of the argument passed to it, otherwise it will just pass the argument passed to it. In other words any negative values are converted to zero's. This is the nonlinearity that we are adding


```python
def simple_net(x):
    res = x@w1 + b1
    res = res.max(tensor(0.0))
    res = res@w2 + b2
    return res
        
```


```python
w1 = init_params(128*128,5)
b1 = init_params(5)
w2 = init_params(5,1)
b2 = init_params(1)
```

In the linear model, the sidebar above, we are able to run our model on batches like so:
`def linear_model(xb): return xb@weights + bias`
but if we were to do the same with our `simple_net` we would soon run into an issue.

This is because the first line `res = x@w1 + b1` will do a matrix multiplication of each item in the batch, against each 128x128 matrix in `w1`.

The third line `res = res@w2 + b2` would take all 5 of the outputs from the second and first lines, but it would return just a single value.

So for all the images in a batch we receive one prediction?

No. This model is actually meant to be run on one image at a time. The number of 128*128 matrices in `w1` is actually just a way of adding 'hidden layers' within the first layer. So what happens is we receive a prediction for the image, for each matrix we put in `w1`. The second linear function (the third line) is therefore, in a way, finding a way to select the stronges prediction produced by the first linear function. We can choose any number of matrices for `w1` with varying levels of accuracy.

In order to run the model on batches, I created a function that will run it in batches and produce a tensor of the results.

>NOTE: I deduced the above logic from three things:  
    1. How the linear model is applied.  
    2. The comments about simple_net made by BobMcDear on [the first page of this thread on the fastai forum](https://forums.fast.ai/t/effect-of-second-layer-on-first-layer-results-in-simple-neural-net/98930).  
    3. The comment made about simple_net by akashgshastri on [the first page of this thread on the fastai forum](https://forums.fast.ai/t/beginner-q-simple-net-vs-cnn-learner-for-rgb-images/80764).  
    As it stands currently, I am open to being corrected on this logic...


```python
def get_batch_predictions(xb):
    return torch.stack([simple_net(x) for x in xb])
```

## Training the Model
Now we also need a loss function that will show how far our prediction is from the truth. Since all our labels/ground truths are values of either 0 or 1, we can normalise our predictions for values that only lie between 0 and 1 also. For this we can use the sigmoid function.


```python
def sigmoid(x): return 1/(1+torch.exp(-x))
```


```python
def loss_function(predictions, targets):
    predictions = predictions.sigmoid()
    return torch.where(targets==1, 1-predictions, predictions).mean()
```

It is the loss that we eventually call `.backward()` on in order to get our gradient with respect to each of the params (`w1`, `b1`, `w2`, and `b2`).

Lets prepare a batch out of our training for giving this method a go.


```python
batch_x = train_x[:5]
batch_preds = get_batch_predictions(batch_x)
batch_targets = train_y[:5]

batch_x,batch_preds,batch_targets
```




    (tensor([[0.4484, 0.4471, 0.4444,  ..., 0.4719, 0.4784, 0.4758],
             [0.2209, 0.2222, 0.2248,  ..., 0.4000, 0.4157, 0.4405],
             [0.7686, 0.7739, 0.7791,  ..., 0.6261, 0.5869, 0.5516],
             [0.5830, 0.5895, 0.5974,  ..., 0.2902, 0.3007, 0.5137],
             [0.8366, 0.8366, 0.8366,  ..., 0.5412, 0.5255, 0.5098]]),
     tensor([[-1149.1401],
             [ -343.6340],
             [-1202.1691],
             [ -801.5582],
             [-1316.0537]], grad_fn=<StackBackward0>),
     tensor([[0],
             [0],
             [0],
             [0],
             [0]]))




```python
loss = loss_function(batch_preds, batch_targets)
loss
```




    tensor(0., grad_fn=<MeanBackward0>)




```python
loss.backward()

w1.grad.shape,w1.grad.mean(),b1.grad,w2.grad.shape,w2.grad.mean(),b2.grad
```




    (torch.Size([16384]),
     tensor(0.),
     tensor([-0., 0., 0., -0., -0.]),
     torch.Size([5]),
     tensor(0.),
     tensor([0.]))



now lets do all that in one function


```python
def calc_gradient(xb,yb,model):
    preds = model(xb)
    loss = loss_function(preds,yb)
    loss.backward()
```


```python
calc_gradient(train_x[:5],train_y[:5],get_batch_predictions)

w1.grad.mean(),b1.grad.mean(),w2.grad.mean(),b2.grad.mean()
```




    (tensor(0.), tensor(0.), tensor(0.), tensor(0.))




```python
w1.grad.zero_()
b1.grad.zero_()
w2.grad.zero_()
b2.grad.zero_()
```




    tensor([0.])



now we can write a train epoch function that does this all in one go.


```python
def train_epoch(model, learning_rate, params):
    for xb,yb in training_dl:
        calc_gradient(xb ,yb, model)
        for p in params:
            p.data -= p.grad*learning_rate
            p.grad.zero_()
```

lets also create a function to check accuracy of each batch.


```python
def batch_accuracy(preds_b, targets_b):
    preds_normalised = preds_b.sigmoid()
    correct = (preds_normalised>0.5) == targets_b
    return correct.float().mean()
    
batch_accuracy(batch_preds, batch_targets)
```




    tensor(1.)



we can also create a function to show how accurate are model is after each training epoch. this would be done by testing it agains our validation data.


```python
### round() rounds the first arugment a number of decimal digits provided by the second argument
def validate_epoch(model):
    accuracies = [batch_accuracy(model(xb), yb) for xb, yb in valid_dl]
    return round(torch.stack(accuracies).mean().item(), 4)
```


```python
validate_epoch(get_batch_predictions)
```




    0.5




```python
lr = 1.
params = w1,b1,w2,b2

# train_epoch(get_batch_predictions, lr, params)
# validate_epoch(get_batch_predictions)
```

now lets try it over a few epochs.


```python
for i in range(10):
    train_epoch(get_batch_predictions, lr, params)
    print(validate_epoch(get_batch_predictions), end=' ')
```

    0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 

Clearly this is not indicative of a model that will improve with training. I tried various ways of tinkering and debugging my `simple_net` but to no avail. I could only conclude that `simple_net` is not sufficient for the task I have, or, if it is to be sufficient, then it requires a lot more computing power than I am throwing at it (which could be by way of more weights matrices and more training epochs).

So the next thing I tried was the exampt same `simple_net` but declared and trained the purely fastai and PyTorch way.

## The PyTorch & fastai way

using PyTorch instead, we can create it the following way:


```python
simple_net = nn.Sequential(
    nn.Linear(128*128,20),
    nn.ReLU(),
    nn.Linear(20,1)
)
```

fastai has as built in Stochastic Gradient Descent optimiser, SGD.

And then fastai also provides us with Learner that we can use to create put everything together:


```python
learn = Learner(dls, simple_net, opt_func=SGD, loss_func=loss_function, metrics=batch_accuracy)
```

now we can use fastai's Learner.fit instead or our for loop of train_epoch.


```python
learn.fit(10,lr=lr)
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
      <th>batch_accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.543056</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.525924</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.515540</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.508081</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.506870</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.506186</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.503422</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.501898</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.501641</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.500227</td>
      <td>0.500000</td>
      <td>0.500000</td>
      <td>00:00</td>
    </tr>
  </tbody>
</table>


So I received pretty much the same result as when I did it using a more _manual_ methodology.

## Using resnet18
And now for the second out of my proposed methods, I will take full advantage of fastai. I will use the `DataBlock` api for creating my DataLoaders and I will use `resnet18` as my model. I'll also use `vision_learner` to train it.

>Note I also used `DataBlock` as part of my image classifier that you can see in my [previous blog post](https://gurtaj1.github.io/blog/2021/09/04/first_post_icon_classifier.html)


```python
data_block = DataBlock(
    blocks=(ImageBlock,CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=2),
    get_y=parent_label,
    item_tfms=Resize(128))
```

We can now use the `dataloaders` method on our `data_black`. As you may see, the data block has defined the following:
- **blocks**: our inputs are `ImageBlock`'s and the label's are `CategoryBlock`'s
- **get_items**: the function we use for getting our images is `get_image_files` (the same one we used in our manual data prepartion).
- **splitter**: we randomnly split the data and take the declared percantage of it to use just for validation. Setting the `seed` means that we will always use the same set of data as our validation, ensuring the model does not get trained on it.
- **get_y**: we set this to parent_label so that what ever name the parent folder of the data has, is what that data will be labeled as.
- **item_tfms**: here we declare how we want our images transformed and normalised.

>Note: in our manual data preperation we created our own validation data set and training data set. As you see above here `DataBlock` is instructed to extract a validation set automatically for us. For that reason we will now access a folder that has all the same data we used in our manual method, but this time they are seperated into 'positive' and 'negative' category as a whole. They are not further split into training and validation.


```python
dls=data_block.dataloaders(path/'all')
```

Lets take a look at some of the images in our validation set.


```python
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_75_0.png)
    


All looking good so far. At the moment the data_block just has each image cropped at a random position, at 128 pixels by 128 pixels. Now lets take a look at some of the different ways we could transform our data, before we use it to train our model.

Here's how it looks with the images 'squished':


```python
data_block = data_block.new(item_tfms=Resize(128, ResizeMethod.Squish))
dls = data_block.dataloaders(path/'all')
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_77_0.png)
    


Here's how it looks with the images 'padded' inorder to fill any space that may be left when minimising them inorder to fit within the specified 128 by 128 size:


```python
data_block = data_block.new(item_tfms=Resize(128, ResizeMethod.Pad, pad_mode='zeros'))
dls = data_block.dataloaders(path/'all',bs=5)
dls.valid.show_batch(max_n=4, nrows=1)
```


    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_79_0.png)
    


Since the actual image in my image library that I am looking for is not skewed or stretched in anyway, I decided to go for the 'padded' mode with my data.

>Note how this time I included `bs=5` in the `dataloaders` method call. Without this i kept getting `nan` for my `train_loss` when training the model. The following three lines of code were what I used to help me detect what was wrong here:  
    `x,y = learn.dls.one_batch()`  
    `out = learn.model(x)`  
    `learn.loss_func(out, y)`  
    which gave the error message: `ValueError: This DataLoader does not contain any batches`.  
    The debugging code was suggested by KevinB in the first page of [this thread on the fastai forum](https://forums.fast.ai/t/train-loss-and-valid-loss-all-nan/91080)


```python
# x,y = learn.dls.one_batch()
# out = learn.model(x)
# learn.loss_func(out, y)
```

I then proceeded to train a `resnet18` model on my data using `vision_learner`:

>Note before successfully doing so it turned out that I had to downgrade my version of `torchvision` to 0.12 so that the methods I was opting to use, would work. I also had to upgrade my MacOS operating system as all apple silicon running on anything below 12.3 MacOS did not have Pytorch GPU support.


```python
#!!!ACTIVATE CORRECT ENVIRONMENT BEFORE RUNNING THIS CELL!!!#

# !pip install --upgrade torchvision==0.12
```


```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(10)
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.761049</td>
      <td>2.855227</td>
      <td>0.833333</td>
      <td>00:03</td>
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
      <th>error_rate</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.013831</td>
      <td>2.266986</td>
      <td>0.666667</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.923140</td>
      <td>1.505093</td>
      <td>0.666667</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.193987</td>
      <td>1.579342</td>
      <td>0.500000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1.121834</td>
      <td>1.969770</td>
      <td>0.500000</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1.127938</td>
      <td>2.468466</td>
      <td>0.666667</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>5</td>
      <td>1.091505</td>
      <td>2.685832</td>
      <td>0.833333</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>6</td>
      <td>1.096134</td>
      <td>2.706148</td>
      <td>0.833333</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>7</td>
      <td>1.015126</td>
      <td>2.648789</td>
      <td>0.833333</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.935627</td>
      <td>2.475007</td>
      <td>0.833333</td>
      <td>00:02</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.947918</td>
      <td>2.624727</td>
      <td>0.833333</td>
      <td>00:02</td>
    </tr>
  </tbody>
</table>


A very underwhelming result, just like in the first method via my `simple_net`.

Non the less I thought I would confirm just how confused this model is by plotting a confusion matrix:


```python
interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()
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








    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_86_4.png)
    


Why not take a look at the top losses whilst were at it!


```python
interp.plot_top_losses(5, nrows=1)
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








    
![png](https://raw.githubusercontent.com/gurtaj1/blog/master/post%20assets/output_88_2.png)
    


## Conclusion
I can only conclude that I, thus far, am using models that are not sufficient for the purpose that I require them for (object detection). I clearly have a lot more to learn about different neural networks and will endeavour to pay close attention to this matter in my continued study.
