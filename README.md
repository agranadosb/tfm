# TFM

This project contains the implementation of the Final project of the Master.


## Installation

This project only works for linux. It has been tested on Ubuntu 18.04 - 20.04 - 22.04.

To properly run the functionality of the project, you need to have the python
version 3.8 or greater installed.

To install the project, you need to create a virtual environment:

```shell
pip3 install virtualenv

python3 -m venv venv

source venv/bin/activate
```

and then install the requirements:

```shell
# You need to install the pytorch version that fits your system. The next
# command is for a system with a GPU and CUDA 11.6 installed. If you have
# another system, please check the pytorch website to install the correct
# version (https://pytorch.org/).
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116

pip install -r requirements.txt
```

If you are going to monitor the training process, you need to install the
following requirements:

```shell
pip install wandb
```

`Wandb` is the tool used to generate the plots and the metrics of the training
process. You can use the personal account (it's free) [https://wandb.ai/site/pricing](https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup)
and install locally the `wandb` tool.

You can check how to use it on the next link: [https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup](https://docs.wandb.ai/guides/hosting/how-to-guides/basic-setup)

## Run

This project has some functionalities to test the Master's project. You can
check the different options with the following command:

```shell
python3 __main__.py --help
```
