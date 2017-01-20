#!/usr/bin/env bash
python main.py --name=demo --log=run1 --mode=train --num_epochs=1

#kvn219 at Kevins-MBP in ~/PycharmProjects/multi-number-recognition on master [?]
#$ python main.py --name=grayscale --log=lrate0001 --mode=train --num_epochs=15 --grayscale
#
#MULTI-SVHN CLASSIFICATION
#Running project: grayscale
#Mode: train
#Is training: True
#Logger: lrate0001
#Checkpoint dir: /Users/kvn219/PycharmProjects/multi-number-recognition/checkpoints/grayscale/
#Loading data from train.tfrecords
#Num epochs: 15
#Batch size: 36
#Learning rate: 0.0001
#Random crop: False
#Grayscale: True
#Channels: 1
#
#Starting new train session
#Training...
#
#[21:49:33] Step:  1000, Loss: 004.11600, Seq Acc: 0.14, Dig Acc: 0.74
#[21:52:00] Step:  2000, Loss: 003.20880, Seq Acc: 0.28, Dig Acc: 0.81
#[21:54:25] Step:  3000, Loss: 003.41800, Seq Acc: 0.28, Dig Acc: 0.78
#[21:56:50] Step:  4000, Loss: 002.73470, Seq Acc: 0.53, Dig Acc: 0.81
#[21:59:26] Step:  5000, Loss: 002.15485, Seq Acc: 0.47, Dig Acc: 0.86
#[22:02:00] Step:  6000, Loss: 001.93764, Seq Acc: 0.58, Dig Acc: 0.88
#[22:04:36] Step:  7000, Loss: 001.61397, Seq Acc: 0.69, Dig Acc: 0.91
#[22:07:13] Step:  8000, Loss: 001.51119, Seq Acc: 0.67, Dig Acc: 0.91
#[22:09:38] Step:  9000, Loss: 001.08460, Seq Acc: 0.69, Dig Acc: 0.92
#[22:12:05] Step: 10000, Loss: 001.24557, Seq Acc: 0.72, Dig Acc: 0.93
#[22:14:30] Step: 11000, Loss: 001.70842, Seq Acc: 0.64, Dig Acc: 0.91
#[22:16:58] Step: 12000, Loss: 001.45446, Seq Acc: 0.72, Dig Acc: 0.92
#[22:19:25] Step: 13000, Loss: 001.80966, Seq Acc: 0.72, Dig Acc: 0.92
#[22:21:55] Step: 14000, Loss: 000.47047, Seq Acc: 0.92, Dig Acc: 0.98
#[22:24:23] Step: 15000, Loss: 000.76695, Seq Acc: 0.78, Dig Acc: 0.94
#[22:26:54] Step: 16000, Loss: 001.24322, Seq Acc: 0.69, Dig Acc: 0.91
#[22:29:30] Step: 17000, Loss: 000.63813, Seq Acc: 0.78, Dig Acc: 0.94
#[22:32:02] Step: 18000, Loss: 001.05948, Seq Acc: 0.75, Dig Acc: 0.93
#[22:34:33] Step: 19000, Loss: 000.86522, Seq Acc: 0.81, Dig Acc: 0.94
#[22:37:07] Step: 20000, Loss: 001.18221, Seq Acc: 0.69, Dig Acc: 0.93
#[22:39:34] Step: 21000, Loss: 000.73875, Seq Acc: 0.81, Dig Acc: 0.95
#[22:42:00] Step: 22000, Loss: 001.07387, Seq Acc: 0.75, Dig Acc: 0.93
#[22:44:28] Step: 23000, Loss: 001.42619, Seq Acc: 0.78, Dig Acc: 0.92
#[22:47:01] Step: 24000, Loss: 001.11561, Seq Acc: 0.78, Dig Acc: 0.94
#[22:49:28] Step: 25000, Loss: 000.71768, Seq Acc: 0.86, Dig Acc: 0.97
#[22:51:57] Step: 26000, Loss: 000.88451, Seq Acc: 0.83, Dig Acc: 0.96
#[22:54:27] Step: 27000, Loss: 002.18220, Seq Acc: 0.83, Dig Acc: 0.94
#[22:57:00] Step: 28000, Loss: 000.34998, Seq Acc: 0.94, Dig Acc: 0.98
#[22:59:32] Step: 29000, Loss: 001.36419, Seq Acc: 0.75, Dig Acc: 0.93
#[23:02:14] Step: 30000, Loss: 000.82556, Seq Acc: 0.83, Dig Acc: 0.95
#[23:04:57] Step: 31000, Loss: 001.05126, Seq Acc: 0.72, Dig Acc: 0.93
#[23:07:42] Step: 32000, Loss: 000.58116, Seq Acc: 0.89, Dig Acc: 0.98
#[23:10:22] Step: 33000, Loss: 000.90778, Seq Acc: 0.81, Dig Acc: 0.95
#[23:12:57] Step: 34000, Loss: 000.55902, Seq Acc: 0.86, Dig Acc: 0.97
#[23:15:34] Step: 35000, Loss: 000.34160, Seq Acc: 0.94, Dig Acc: 0.99
#[23:18:05] Step: 36000, Loss: 000.70535, Seq Acc: 0.72, Dig Acc: 0.93
#[23:20:50] Step: 37000, Loss: 000.68446, Seq Acc: 0.83, Dig Acc: 0.96
#[23:23:22] Step: 38000, Loss: 000.36481, Seq Acc: 0.92, Dig Acc: 0.98
#[23:25:53] Step: 39000, Loss: 000.69565, Seq Acc: 0.86, Dig Acc: 0.97
#[23:28:14] Step: 40000, Loss: 000.50935, Seq Acc: 0.89, Dig Acc: 0.97
#[23:30:33] Step: 41000, Loss: 000.43204, Seq Acc: 0.86, Dig Acc: 0.97
#[23:32:52] Step: 42000, Loss: 000.68956, Seq Acc: 0.92, Dig Acc: 0.97
#[23:35:11] Step: 43000, Loss: 001.27975, Seq Acc: 0.69, Dig Acc: 0.92
#[23:37:30] Step: 44000, Loss: 000.32484, Seq Acc: 0.89, Dig Acc: 0.98
#[23:39:48] Step: 45000, Loss: 000.66037, Seq Acc: 0.83, Dig Acc: 0.96
#[23:42:05] Step: 46000, Loss: 000.50126, Seq Acc: 0.92, Dig Acc: 0.97
#[23:44:23] Step: 47000, Loss: 001.04638, Seq Acc: 0.78, Dig Acc: 0.94
#[23:46:41] Step: 48000, Loss: 000.75738, Seq Acc: 0.89, Dig Acc: 0.97
#[23:48:58] Step: 49000, Loss: 000.53394, Seq Acc: 0.83, Dig Acc: 0.97
#[23:51:16] Step: 50000, Loss: 000.20359, Seq Acc: 0.92, Dig Acc: 0.98
#[23:53:34] Step: 51000, Loss: 000.37284, Seq Acc: 0.92, Dig Acc: 0.98
#[23:55:51] Step: 52000, Loss: 000.40561, Seq Acc: 0.89, Dig Acc: 0.97
#[23:58:09] Step: 53000, Loss: 000.59903, Seq Acc: 0.94, Dig Acc: 0.97
#[00:00:26] Step: 54000, Loss: 000.82204, Seq Acc: 0.81, Dig Acc: 0.95
#[00:02:43] Step: 55000, Loss: 000.46429, Seq Acc: 0.83, Dig Acc: 0.96
#[00:05:00] Step: 56000, Loss: 000.44149, Seq Acc: 0.92, Dig Acc: 0.98
#[00:07:17] Step: 57000, Loss: 000.99733, Seq Acc: 0.78, Dig Acc: 0.94
#[00:09:34] Step: 58000, Loss: 000.27153, Seq Acc: 0.92, Dig Acc: 0.98
#[00:11:52] Step: 59000, Loss: 000.34306, Seq Acc: 0.89, Dig Acc: 0.97
#[00:14:09] Step: 60000, Loss: 000.77103, Seq Acc: 0.83, Dig Acc: 0.95
#[00:16:28] Step: 61000, Loss: 000.34287, Seq Acc: 0.94, Dig Acc: 0.98
#[00:18:45] Step: 62000, Loss: 000.47860, Seq Acc: 0.89, Dig Acc: 0.98
#[00:21:03] Step: 63000, Loss: 000.56967, Seq Acc: 0.89, Dig Acc: 0.98
#[00:23:21] Step: 64000, Loss: 000.52041, Seq Acc: 0.92, Dig Acc: 0.97
#[00:25:38] Step: 65000, Loss: 000.17911, Seq Acc: 0.94, Dig Acc: 0.99
#[00:28:06] Step: 66000, Loss: 000.33035, Seq Acc: 0.92, Dig Acc: 0.98
#[00:30:36] Step: 67000, Loss: 002.28640, Seq Acc: 0.75, Dig Acc: 0.93
#[00:33:07] Step: 68000, Loss: 000.60623, Seq Acc: 0.83, Dig Acc: 0.97
#[00:35:29] Step: 69000, Loss: 000.74583, Seq Acc: 0.92, Dig Acc: 0.97
#[00:37:46] Step: 70000, Loss: 000.29726, Seq Acc: 0.89, Dig Acc: 0.98
#[00:40:02] Step: 71000, Loss: 000.48197, Seq Acc: 0.92, Dig Acc: 0.98
#[00:42:19] Step: 72000, Loss: 000.32517, Seq Acc: 0.89, Dig Acc: 0.96
#[00:44:35] Step: 73000, Loss: 000.48915, Seq Acc: 0.89, Dig Acc: 0.97
#[00:46:51] Step: 74000, Loss: 000.58024, Seq Acc: 0.83, Dig Acc: 0.96
#[00:49:08] Step: 75000, Loss: 000.37826, Seq Acc: 0.92, Dig Acc: 0.98
#[00:51:24] Step: 76000, Loss: 001.16864, Seq Acc: 0.86, Dig Acc: 0.97
#[00:53:41] Step: 77000, Loss: 000.21738, Seq Acc: 0.94, Dig Acc: 0.99
#[00:55:58] Step: 78000, Loss: 000.12565, Seq Acc: 0.97, Dig Acc: 0.99
#[00:58:22] Step: 79000, Loss: 000.14335, Seq Acc: 0.94, Dig Acc: 0.99
#[01:00:49] Step: 80000, Loss: 000.36520, Seq Acc: 0.86, Dig Acc: 0.97
#[01:03:22] Step: 81000, Loss: 000.37449, Seq Acc: 0.92, Dig Acc: 0.98
#[01:05:43] Step: 82000, Loss: 000.12966, Seq Acc: 0.97, Dig Acc: 0.99
#[01:08:00] Step: 83000, Loss: 000.13364, Seq Acc: 0.97, Dig Acc: 0.99
#[01:10:16] Step: 84000, Loss: 000.43217, Seq Acc: 0.92, Dig Acc: 0.98
#[01:12:35] Step: 85000, Loss: 000.28366, Seq Acc: 0.86, Dig Acc: 0.97
#[01:14:55] Step: 86000, Loss: 000.53918, Seq Acc: 0.81, Dig Acc: 0.96
#[01:17:17] Step: 87000, Loss: 000.45932, Seq Acc: 0.86, Dig Acc: 0.96
#[01:19:38] Step: 88000, Loss: 000.27797, Seq Acc: 0.89, Dig Acc: 0.98
#[01:21:56] Step: 89000, Loss: 000.20260, Seq Acc: 0.94, Dig Acc: 0.98
#[01:24:18] Step: 90000, Loss: 000.20315, Seq Acc: 0.94, Dig Acc: 0.99
#[01:26:36] Step: 91000, Loss: 000.54340, Seq Acc: 0.89, Dig Acc: 0.97
#[01:28:54] Step: 92000, Loss: 000.48221, Seq Acc: 0.94, Dig Acc: 0.98
#[01:31:10] Step: 93000, Loss: 000.46957, Seq Acc: 0.89, Dig Acc: 0.98
#Done training for   15 epochs, 93228 steps.
#Finished training.
#Total time to run: 3:44:57.838850