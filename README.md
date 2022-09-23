## MammoFL_MICCAI2022

VICoROB_MAIA team implementation (Following the [challenge reference implementation](https://github.com/Project-MONAI/tutorials/tree/main/federated_learning/breast_density_challenge)) for
[ACR-NVIDIA-NCI Breast Density FL challenge](http://BreastDensityFL.acr.org).

Held in conjunction with [MICCAI 2022](https://conferences.miccai.org/2022/en/).


------------------------------------------------
## 1. Run Training using [NVFlare](https://github.com/NVIDIA/NVFlare) on Toy Dataset

### 1.1 Download example data
Follow the steps described in [./data/README.md](./data/README.md) to download an example breast density mammography dataset.
Note, the data used in the actual challenge is different. 
Challenge provide a set of random data splits. Please download them using
```
python3 ./code/pt/utils/download_datalists_and_predictions.py
```
After download, they will be available as `./data/dataset_blinded_site-*.json` which follows the same format as what
will be used in the challenge.
Please do not modify the data list filenames in the configs as they will be the same during the challenge.

Note, the location of the dataset and data lists will be given by the system.
Do not change the locations given in [config_fed_client.json](./code/configs/mammo_fedavg/config/config_fed_client.json):
```
  "DATASET_ROOT": "/data/preprocessed",
  "DATALIST_PREFIX": "/data/dataset_blinded_",
```

### 1.2 Build container
The argument specifies the FQDN (Fully Qualified Domain Name) of the FL server. Use `localhost` when simulating FL on your machine.
```
./build_docker.sh localhost
```
Note, all code and pretrained models need to be included in the docker image.
The virtual machines running the containers will not have public internet access during training.
For an example, please see the `download_model.py` used to download ImageNet pretrained weights in this example.

The Dockerfile is submitted using the [MedICI platform](https://www.medici-challenges.org).
For detailed instructions, see the [challenge website](http://BreastDensityFL.acr.org).

### 1.3 Run server and clients containers, and start training
Run all commands at once using. Note this will also create separate logs under `./logs`
```
./run_all_fl.sh
```
Note, the GPU index to use for each client is specified inside `run_all_fl.sh`.
See the individual `run_docker_site-*.sh` commands described below.
Note, the server script will automatically kill all running container used in this example
and final results will be placed under `./result_server`.

(optional) Run each command in a separate terminals to get site-specific printouts in separate windows.

The argument for each shell script specifies the GPU index to be used.
```
./run_docker_server.sh
./run_docker_site-1.sh 0
./run_docker_site-2.sh 1
./run_docker_site-3.sh 0
```

### 1.4 (Optional) Visualize training using TensorBoard
After training completed, the training curves can be visualized using
```
tensorboard --logdir=./result_server
```
A visualization of the global accuracy and [Kappa](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html) validation scores for each site with the provided example data is shown below.
The current setup runs on a machine with two NVIDIA GPUs with 12GB memory each.
The runtime for this experiment is about 45 minutes.
You can adjust the argument to the `run_docker_site-*.sh` scripts to specify different
GPU indices if needed in your environment.

![](./figs/example_data_val_global_acc_kappa.png)

### 1.5 (Optional) Kill all containers
If you didn't use `run_all_fl.sh`, all containers can be killed by running
```
docker kill server site-1 site-2 site-3
```


------------------------------------------------
## 2. Modify the FL algorithm

You can modify and extend the provided example code under [./code/pt](./code/pt).

You could use other components available at [NVFlare](https://github.com/NVIDIA/NVFlare)
or enhance the training pipeline using your custom code or features of other libraries.

See the [NVFlare examples](https://github.com/NVIDIA/NVFlare/tree/main/examples) for features that could be utilized in this challenge.

### 2.1 Debugging the learning algorithm

The example NVFlare `Learner` class is implemented at [./code/pt/learners/mammo_learner.py](./code/pt/learners/mammo_learner.py).
You can debug the file using the `MockClientEngine` as shown in the script by running
```
python3 code/pt/learners/mammo_learner.py
```
Furthermore, you can test it inside the container, by first running
```
./run_docker_debug.sh
```
Note, set `inside_container = True` to reflect the changed filepaths inside the container.


------------------------------------------------
## Challenge evaluation
The script used for evaluating different submissions is available at [challenge_evaluate.py](./challenge_evaluate.py)
