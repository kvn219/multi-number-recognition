# Multi-Number-Recognition
#### Using Street View Housing Numbers (SVHN)

<p align="center">
    <img src="https://github.com/kvn219/multi-number-recognition/blob/master/assets/raw_train_images.png" hieght="640" width="576"/>
</p>

This repository contains my capstone project for Udacity's Machine Learning Engineering Nanodegree. Below is a brief description of the dataset and instructions to reproduce my results.


## Project Overview

The purpose of the capstone project was to examine street view images of house numbers and predict all the numbers present in the image. The evaluation metric for the project is per digit accuracy.  In other words, the number of correct digits predicted divided by the total number of digits in the dataset.  Our best model achieves 95% per digit accuracy after ten epochs of training (going through the dataset ten times).   

## Notebooks to Browse

[1. Explore raw images](https://github.com/kvn219/multi-number-recognition/blob/master/notebooks/1_Explore_Raw_Images.ipynb)

[2. Samples analysis](https://github.com/kvn219/multi-number-recognition/blob/master/notebooks/2_Samples_Analysis.ipynb)

[3. Model evaluation](https://github.com/kvn219/multi-number-recognition/blob/master/notebooks/3_Model_Evaluation.ipynb)

[4. Overfitting](https://github.com/kvn219/multi-number-recognition/blob/master/notebooks/4_Overfitting.ipynb)


## Requirements
This project works on [Anaconda's](http://continuum.io/downloads) distribution of Python 3.5

__Once you have Anaconda installed:__

#### 1) Create a virtual environment call capstone_kvn219 with the required packages. 

~~~
$ conda env create capstone_kvn219 -f environment.yml
~~~

#### 2) Activate the newly created environment.

~~~
$ source activate capstone_kvn219
~~~

#### 3) Download the data.
  
 You'll have two options.  The recommended option is to download the TFRecords, I've preprocessed.  The second, is to download and process the data on your own.

#### 3a) Download the data from dropbox (recommended).
 
- Go to records/README.md
- Download train, test, and validation data and place them in the records/ directory.

#### 3b) Download the data your self (not recommended). 
~~~
cd svhn
python download.py
python wrangle.py
cd ..
$ python convert_to_records.py
~~~

#### 4) Train the model

~~~
$ python main.py --name=demo --mode=train --log=run1 --num_epochs=10
~~~
Next you'll see the following output.  This is to help you keep track of parameters.  
Note: Some directories will not apply if you downloaded the data from dropbox.  
~~~
$ python main.py --name=demo --mode=train --log=run1 --num_epochs=10

MULTI-SVHN CLASSIFICATION
Running project: demo
Mode: train
Is training: True
Log name: run1
Checkpoint dir: /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/
Loading data from train.tfrecords
Num epochs: 10
Batch size: 36
Learning rate: 0.0001
Random crop: False
Grayscale: False
Channels: 3

Starting new train session
Training...
~~~

Training takes awhile, so you should see something like this after 15 mins.

~~~
...
Starting new train session
Training...

[20:12:03] Step:  1000, Loss: 005.53591, Seq Acc: 0.08, Dig Acc: 0.65
[20:14:19] Step:  2000, Loss: 003.99407, Seq Acc: 0.19, Dig Acc: 0.76
[20:16:40] Step:  3000, Loss: 004.14222, Seq Acc: 0.22, Dig Acc: 0.76
[20:18:58] Step:  4000, Loss: 002.56051, Seq Acc: 0.53, Dig Acc: 0.87
~~~
 
 We've set up the program like this so that you can stop the program [KeyboardInterrupt].  And pick up where you left off later by using the same command.
 
 ~~~
 $ python main.py --name=demo --mode=train --log=run1 --num_epochs=10
 ...
Restored model from: /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-4080
Training...

[20:23:12] Step:  5000, Loss: 002.25958, Seq Acc: 0.47, Dig Acc: 0.85
[20:25:40] Step:  6000, Loss: 001.39092, Seq Acc: 0.64, Dig Acc: 0.91
...
 ~~~
 
#### 5) Validation

To get validation results, keep the same name, change the mode to "valid" and reduce the num_epochs to 1.
~~~
$ python main.py --name=demo --mode=valid --num_epochs=1
~~~

~~~
0it [00:00, ?it/s]
Restoring...
 /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-1000
Stopping evaluation at    1 epochs,   1 steps.
Total time to run: 0:00:16.118897

1it [00:16, 16.76s/it]
Restoring...
 /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-2000
Stopping evaluation at    1 epochs,   1 steps.
Total time to run: 0:00:15.927491

2it [00:33, 16.70s/it]
Restoring...
 /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-3000
Stopping evaluation at    1 epochs,   1 steps.
Total time to run: 0:00:17.427519

3it [00:51, 17.16s/it]
Restoring...
 /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-4000
Stopping evaluation at    1 epochs,   1 steps.
Total time to run: 0:00:16.658947

4it [01:08, 17.21s/it]
Restoring...
 /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/demo/demo-5000
Stopping evaluation at    1 epochs,   1 steps.
Total time to run: 0:00:17.136208

5it [01:26, 17.41s/it]
Results saved to /Users/kvn219/PycharmProjects/multi-number-recognition/results/demo/valid-demo.csv

   checkpoint  digit_accuracy  sequence_accuracy
0        1000        0.657715           0.075794
1        2000        0.740248           0.202088
2        3000        0.810117           0.373621
3        4000        0.846223           0.482516
4        5000        0.869377           0.551180
(capstone)
~~~
Your results will be saved to /results/valid-name_of_task.csv

#### 6) Test

The same applies for testing.  You'll just need to change the mode.
~~~
$ python main.py --name=test --mode=test --num_epochs=1
~~~

#### 7) Tensorboard

![alt text](https://github.com/kvn219/multi-number-recognition/blob/master/assets/demo.gif)

To view the model in tensorboard, you can enter the following:
~~~
$ tensorboard --logdir=./logs --port 6006

Starting TensorBoard b'39' on port 6006
(You can navigate to http://192.168.1.9:6006)
WARNING:tensorflow
...
~~~
Go to your browser and go to the url: http://192.168.1.9:6006/

# Extras

We also provide a supplemental metric: sequential accuracy, which measures the number of correct predicted sequences divided by the number of total sequences in the dataset.


# Resources and Attribution

 Two good papers on the subject are from [Goodfellow et al. (2014)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42241.pdf) and [Netzer et al. (2011)](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/37648.pdf).   And for more recent articles:  [Jaderberg et al. (2015)](http://papers.nips.cc/paper/5854-spatial-transformer-networks)  and [Ba et al. (2014)](https://arxiv.org/pdf/1412.7755v2.pdf) are also very useful resources. Jaderberg presents the use of Spatial Transformer Networks and Ba attention-based models. Two great examples of SVHN projects are from fellow Udacity students from [camigord](https://github.com/camigord/ML_CapstoneProject) and [hangyao](https://github.com/hangyao/street_view_house_numbers).
 
 Much of the work in this repo is directly inspired by the work of the papers and projects mentioned above.
