**Smart Meter Project**
======
Several tasks related to smart meters project, including
+ access data from remote database
+ visualize statistics load data as figures
+ train anomaly detection model for each resident
+ show the reconstruction error between model result and input data


**Directory structure**
------
![folder](https://user-images.githubusercontent.com/56117848/183457454-4c4d838a-26bb-4730-a117-42236662e54c.png)


**Dependency** 
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

If you are using conda to manage package and in linux, run
```
source check.sh
```
to check current dependency package and version.  
Or run
```
conda list | findstr /rc:'^python ' /rc:'pytorch ' /rc:'cudatoolkit' /rc:'numpy ' /rc:'pandas' /rc:'matplotlib ' /rc:'tensorboard ' /rc:'request ' /rc:'xlrd'
```
in windows cmd.  
If the specific package is not show in return list, that means package haven't been installed yet, or not in this environment but in base.  
![dependency](https://user-images.githubusercontent.com/56117848/183455976-339e6750-d55f-409c-9b45-a5c47d74bad2.png)

**Tasks**
------
There are several functions in `main.py`, each function represent a corresponding task.  
To do the certain task, uncomment the related line in `if __name__=='__main__:'` block and comment other functions, then run the `main.py` by
```
python main.py
```
+ access data  
`use debug or interactive mode would be better, since need to update several times when network is unstable.`
    1. use ssh connect to specified host
    2. run openproxyserver.bat to open tunnel as proxy server
    3. make sure `110resident.xls` and `connection.json` in `access` folder
    4. run `access_api_data` function in `main.py`  
    expected result would be like:
![access](https://user-images.githubusercontent.com/56117848/183474632-6921d12c-f2a7-4107-9956-27fbdceaaf08.png)
+ visualize statistics load data  
    1. make sure there are `data_xxx.csv` and `info.csv` in `dataset/TMbase` folder
    2. adjust parameters of `plot_load_analysis` function in main script
    3. run  `main.py`  
    example result, figure-2021 of `--show ymwh` flag
![2021_ymwh](https://user-images.githubusercontent.com/56117848/183475715-56def52a-fed0-4aeb-b13b-7d265978ba9c.png)
+ train reconstruction model  
We modify nbeats model to output backcast and fit the re-construction based anomaly detection task. And we use CNN structure with `down-sampling factor` to control relation between input and representation instead of `backcast_length` and `forecast_length`. So the origin fully-connected structure nbeats model may not work fine and should be deprecate.
    1. make sure there are `data_xxx.csv` in `dataset/TMbase` folder
    2. adjust parameters of `train_model` function in main script  
    `make_argv` actually return 2-layer list, outside is for each resident from `cond1` and inside is arguments corresponding to the resident to train the model.
        + choose GPU device index
        + log `context_visualization` may consume large space
    3. run  `main.py`  
    + Information in training process is logged in `exp/{expname}/log/{name}.csv` 
    + the model weight is saved in `exp/{expname}/model/{name}.mdl`
    + To visualize the information, run tensorboard by
        ```
        tensorboard --logdir=exp/{expname}/run [--port]
        ```
        `{expname}` need to be replace to experimental name, and specify port number if the default port is conflict

+ detection  
    1. make sure there are `data_xxx.csv` in `dataset/TMbase` folder and model weight under `exp/{expname}/model`
    2. adjust parameters in each task:
        + compute anomaly ratio by given threshold list run `detect_compute_ratio` function
        + apply other residents' data on each model run `detect_apply_on_other_data` function
        + show reconstruction result and error run `detect_user_period` function
        + parameter `output_place` of function `detect_compute_ratio` and `detect_apply_on_other_data` can adjust figure output place,  
        `None` for directly ouput to window or plot panel in editor, or given string to log in tensorboard.
+ other minor tasks
    + plot_user_data
    + plot_model_basis
    + plot_model_result


**References**
------
[1] [nbeats paper](https://openreview.net/forum?id=r1ecqn4YwB)  
[2] [nbeats source code](https://github.com/philipperemy/n-beats)  
[3] [IHEPC dataset](https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption)
