# MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation

<div style="display: flex;">
  <img src="/images/drums_mi_2_spiral_550000_rgb.gif" alt="Image" width="200">
  <img src="/images/ship_mi_spiral_500000_rgb.gif" alt="Image" width="200">
  <img src="/images/lego_mi_final_spiral_500000_rgb.gif" alt="Image" width="200">
</div>

| arXiv |
| :---- |
| [MultiPlaneNeRF: Neural Radiance Field with Non-Trainable Representation](https://arxiv.org/pdf/2305.10579.pdf)|


### Abstract
*NeRF is a popular model that efficiently represents 3D objects from 2D images. However, vanilla NeRF has a few important limitations. NeRF must be trained on each object separately. The training time is long since we encode the object’s shape and color in neural network weights. Moreover, NeRF does not generalize well to unseen data. In this paper, we present MultiPlaneNeRF – a first model
that simultaneously solves all the above problems. Our model works directly on 2D images. We project 3D points on 2D images to produce non-trainable representations. The projection step is not parametrized, and a very shallow decoder can efficiently process the representation. Using existing images as part of NeRF can significantly reduce the number of parameters since we train only a small implicit decoder. Furthermore, we can train MultiPlaneNeRF on a large data set and force our implicit decoder to generalize across many objects. Consequently, we can only replace the 2D images (without additional training) to produce a NeRF representation of the new object. In the experimental section, we demonstrate that MultiPlaneNeRF achieves comparable results to state-of-the-art models for synthesizing new views and has generalization properties.*

 <img src="/images/image.png" alt="Image" width="400">

Code based on NeRF pytorch implementation by yenchenlin: https://github.com/yenchenlin/nerf-pytorch, that implements the method of MultiPlaneNeRF paper.

## MultiPlaneGAN

In this repo, we add an implementation of MultiPlaneGAN  https://github.com/gmum/MultiPlaneGan

## Requirements
- Dependencies stored in `requirements.txt`.
- Python 3.9.12
- CUDA

## Usage

### Installation
Create new conda environment and install requirements specified in: `pip install -r requirements.txt`

Download official NeRF dataset

### Running - single object
Official NeRF data, can be dowloaded for `lego` model by running:
```
bash download_example_data.sh
```

To run a training for MultiPlaneNeRF, run the Python script:

```
python run_nerf.py --config configs/lego.txt
```

### Running - generalized

We use custom dataset adapted from Shapenet, which contains cars, chairs and planes classes. Each class has 50 images with size 200x200 and corelated pose for each render.

[Download dataset here.](https://ujchmura-my.sharepoint.com/:u:/g/personal/przemyslaw_spurek_uj_edu_pl/ETy5BPpf4ZFLorYEpXxhRRcBY1ASvCqDCgEX_h75Um6MlA?e=MTJdaj)

Put data folders in local path: `./data/multiple` (same path as `datadir` in config files)

To run a training for MultiPlaneNeRF in generalized mode, run the Python script:

```
python run_nerf_many_generalize.py --config configs/generalized_cars.txt
```
