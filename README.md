# 3D Reconstruction for Monocular-Depth-Estimation


## Installation

Please refer to [get_started.md](docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.

or like this

my environment: torch=1.9.1,CUDA=11.4
```
python setup.py build
python setup.py install
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```

## Prepare
* depth image (nyu or kitti)

attention:The depth map names should match Monocular-Depth-Estimation-Toolbox/splits/*.txt

## Run
*Nyu 3D reconstruction*
```py
python ./tools/misc/visualize_point-cloud.py --output-dir point_cloud --dataset nyu --exp_name bts --depth_raw_path "/mnt/data2/datasets/results/nyu/raw/"
```

*Kitti 3D reconstruction*
```py
python ./tools/misc/visualize_point-cloud.py --output-dir point_cloud --dataset kitti --exp_name bts --depth_raw_path "/mnt/data2/datasets/results/kitti/raw/"
```

## Reference
https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox


