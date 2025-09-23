## Person Detection and Tracking Using Artificial Intelligence

> [Runze Guo](https://github.com/Alanze) [Jun Li](https://github.com/mistlirry) [Jacqueline Neo](https://github.com/jacqueline-neo) [Ziling Luo](https://github.com/ESTELLA219) [Guanqiao Ren](https://github.com/ERgq5739)

This repository contains the code and report for the project of EE6008 We use the model [yolov8](https://github.com/ultralytics/ultralytics)+Deepsort & ByteTrack for person detection and tracking

### Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

### Getting start

Install the `ultralytics` package, including all [requirements](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml), in a [**Python>=3.8**](https://www.python.org/) environment with [**PyTorch>=1.8**](https://pytorch.org/get-started/locally/).

```bash
pip install ultralytics
```

### Datasets

Install [MOT20](https://motchallenge.net/data/MOT20/) and [MOT17](https://motchallenge.net/data/MOT17/) and change the datasets using 

```bash
python data_process.py
```

### Train and Test

As an example, the datasets can be unzipped at `./datasets`

To view training results and loss plots, please check the `./run`

- Train the yolov8 model:

```bash
yolo task=detect mode=train model=./yolov8n.pt data=".yaml" workers= epochs= batch=
```

- Test the yolov8 model

```bash
yolo detect predict model= source=
```