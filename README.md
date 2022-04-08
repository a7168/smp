Smart Meter Project
======
NBEATS for load data

Directory structure
------
![directory structure](https://user-images.githubusercontent.com/56117848/159694068-b48137c8-961e-4eae-b2f9-849a7694cf3c.png)


Environment
------

| Package        |        Version |
| :------------- | -------------: |
| python         |          3.9.7 |
| pytorch        |         1.10.2 |
| cudatoolkit    |         11.3.1 |
| numpy          |         1.21.2 |
| pandas         |          1.4.1 |
| matplotlib     |          3.5.1 |
| tensorboard    |          2.8.0 |
| requests       |         2.27.1 |
| xlrd           |          2.0.1 |

Run on terminal
------
for training task:
```
python train.py [<options>]
```
for detecting task:
```
python detect.py
```

References
------
[1] [nbeats paper](https://openreview.net/forum?id=r1ecqn4YwB)  
[2] [nbeats source code](https://github.com/philipperemy/n-beats)  
[3] [IHEPC dataset](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
