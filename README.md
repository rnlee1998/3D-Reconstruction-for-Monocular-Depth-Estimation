# 3D Reconstruction for Monocular-Depth-Estimation


## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

## Prepare
* depth image (nyu or kitti)

## Run
*Nyu 3D reconstruction*
```py
python ./tools/misc/visualize_point-cloud.py --output-dir point-cloud --dataset nyu --exp_name bts --depth_raw_path "/mnt/data2/datasets/results/nyu/raw/"
```

*Kitti 3D reconstruction*
```py
python ./tools/misc/visualize_point-cloud.py --output-dir point-cloud --dataset kitti --exp_name bts --depth_raw_path "/mnt/data2/datasets/results/kitti/raw/"
```

## Reference
https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox


