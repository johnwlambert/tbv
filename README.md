## Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection (NeurIPS '21, Official Repo)
[John Lambert](https://johnwlambert.github.io/), [James Hays](https://www.cc.gatech.edu/~hays/)

This repository contains the source code for training and evaluating models described in the NeurIPS '21 paper *Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection*. [[Paper]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6f4922f45568161a8cdf4ad2299f6d23-Paper-round2.pdf) [[Supplementary Material]](https://openreview.net/attachment?id=cXCZnLjDm4s&name=supplementary_material)

The Trust but Verify (TbV) dataset is publicly available for download, as part of the [**Argoverse 2.0** family of datasets](https://www.argoverse.org/av2.html). Download instructions can be found [here](https://github.com/argoai/av2-api/blob/main/src/av2/datasets/tbv/README.md#downloading-tbv). You can find a short invited talk at the CVPR 2021 VOCVALC workshop summarizing our work [here on Youtube](https://youtu.be/JeTZbCuyeM8?t=3735).

## Table of Contents

- [Dataset Overview](#dataset-overview)
- [Installation](#installation)
- [Downloading the dataset](#download-dataset)
- [Rendering Training/Val/Test Data](#render-data)
- [Training Models](#training-models)
- [Evaluating Models](#evaluating-models)
- [Pre-trained Model Accuracies](#pretrained-model-accuracies)
- [Generating `seamseg` semantic segmentation label maps](#generating-seamseg-label-maps)
- [Citing](#citing)
- [FAQ](#faq)
- [License](#license)


<a name='dataset-overview'></a>
## Dataset Overview

The Trust but Verify (TbV) dataset is the first public dataset for the task of high-definition (HD) map change detection, i.e. determining when sensor data and map data are no longer in agreement with one another due to real-world changes. We collected TbV by mining thousands of hours of data from over 9 months of autonomous vehicle fleet operations.

[//]: # (92me_Jd8_hstacked, cJiXg_jDONA, 92me_Jd8_hstacked_map, 0f0c_Aom_hstack, xpxg_4ab4, Hl998_zU1P_hstacked)

An example from Pittsburgh:
<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/160265919-99b300fa-d7b9-4eaa-8889-46f8f142ee52.gif" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/160266065-7cbccc84-daa7-4129-92da-0f50569f0aed.gif" height="215">
</p>

Examples from Palo Alto and Miami:
<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/160265925-174a2958-e8d6-42ac-84a2-8e012b9e4e71.gif" height="215">
  <img src="https://user-images.githubusercontent.com/29715011/160264648-cfad3571-3bfa-4acb-9831-d8f8693a7dc8.gif" height="215">
</p>

Examples from Miami:
<p align="center">
  <img src="https://user-images.githubusercontent.com/29715011/160264672-3e725689-395f-4fd5-847f-d9ba273e8c5c.gif" height="215">
  <img src="https://user-images.githubusercontent.com/29715011/160264678-760a8167-0b1e-442d-8d73-e0487868c40a.gif" height="215">
</p>

The dataset, consisting of maps and logs collected in six North American cities, is one of the largest AV datasets to date with more than 7.9 million images and will be made available to the public, along with code and models under the the CC BY-NC-SA 4.0 license. Above, we show before-and-after images that showcase a few examples of map changes featured in TbV logs.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

<a name='installation'></a>
## Installation

First, clone the repo:
```bash
git clone https://github.com/johnwlambert/tbv.git
```
Next, install Miniconda or Anaconda, and create the conda environment:
```bash
conda env create -f environment_linux.yml
conda env create -f environment_mac.yml
```
Note: rendering data is only supported on Linux with a CUDA-supported GPU.
```bash
cd tbv
pip install -e .
```
When you clone the repo, the structure should be as follows:
```
- tbv/
 |--- setup.py
 |--- tbv-raytracing/
   |---setup.py
   |---pybind11/
```
Next, install `mseg-api` anywhere on your machine using:
```bash
git clone https://github.com/mseg-dataset/mseg-api.git
cd mseg-api
pip install -e .
cd ..
```
Next, install `mseg-semantic` anywhere on your machine using:
```bash
git clone https://github.com/mseg-dataset/mseg-semantic.git
pip install -e .
```
Next, install `av2` ([`av2-api`](https://github.com/argoai/av2-api/blob/main/README.md) is the official repo for Argoverse 2.0):
```bash
pip install av2==0.1.0
```

Next, install Eigen. On Linux, `sudo apt install libeigen3-dev`. Next, `cd tbv-raytracing` and download `pybind11` via `git clone https://github.com/pybind/pybind11.git`, where it should be downloaded into the second-level `tbv-raytracing` dir.

Ensure your `nvcc` compiler is at least version `11.3, V11.3.109`, and then compile the GPU library using `setup.py` as follows:
```bash
python setup.py bdist_wheel
pip install dist/tbv_raytracing-0.0.1-cp{PY_VERSION}-cp{PY_VERSION}-linux_x86_64.whl
```
e.g. this file could be named one of the following:
```python
pip install dist/tbv_raytracing-0.0.1-cp38-cp38-linux_x86_64.whl
pip install dist/tbv_raytracing-0.0.1-cp39-cp39-linux_x86_64.whl 
pip install dist/tbv_raytracing-0.0.1-cp310-cp310-linux_x86_64.whl
```

<a name='download-dataset'></a>
## Downloading the dataset

Download the dataset per the instructions found [here](https://github.com/argoai/av2-api/blob/main/src/av2/datasets/tbv/README.md#downloading-tbv).

Create a folder, and then `logs/`

<a name='render-data'></a>
## Rendering Training/Val/Test Data

To render data in a bird's eye view, run
```bash
python scripts/run_dataset_rendering_job.py --config_name bev_config.yaml
```

To render data in the ego-view, run
```bash
python scripts/run_dataset_rendering_job.py --config_name egoview_config.yaml
```

For training data w/ augmentations, ensure the following fields are set to `render_test_set_only: False` and `jitter_vector_map: True`.


[//]: # ()
[//]: # (i47t2l9q6nJQNDFbD6iI0MrQ11yq4JNB__Autumn_2020_315971536859131000_rgb_interpTruelinear_projmethodray_tracing)
[//]: # (i47t2l9q6nJQNDFbD6iI0MrQ11yq4JNB__Autumn_2020_315971536859131000_semantics_interpTruenearest_projmethodray_tracing)
[//]: # (ujhUH2flle6ctyPZawAe9GiSzoeqkAJX__Summer_2020_315971131960015000_rgb_interpTruelinear_projmethodray_tracing)
[//]: # (ujhUH2flle6ctyPZawAe9GiSzoeqkAJX__Summer_2020_315971131960015000_semantics_interpTruenearest_projmethodray_tracing)

<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/165433493-219d9070-c157-4b05-9862-70fe5b6eec3d.jpg" height="200">
  <img src="https://user-images.githubusercontent.com/16724970/165433531-26994c65-62e2-4ba4-b2ce-6c7669d0d864.png" height="200">
  <img src="https://user-images.githubusercontent.com/16724970/165433535-84be378b-620e-4523-97f6-01cb37536c1d.jpg" height="200">
  <img src="https://user-images.githubusercontent.com/16724970/165433547-9479430c-432c-4722-bdfa-acec0c3dab44.png" height="200">
</p>

Program output will be saved in a `logging_output` directory.

We use the following abbreviations for city names featured in TbV:
| City Name | Abbreviation | 
| :-------: | :----------: |
| Washington, DC | WDC |
| Miami, FL | MIA |
| Pittsburgh, PA | PIT |
| Palo Alto, CA | PAO | 
| Austin, TX | ATX | 
| Detroit, MI | DTW |


<a name='training-models'></a>
## Training Models

After rendering a dataset, you're ready to train models. Start training by running:
```bash
python scripts/train.py \
    --training_config_name {CONFIG_UUID}.yaml \
    --rendering_config_name {CONFIG_UUID}.yaml
```

<a name='evaluating-models'></a>
## Evaluating Models

**Pretrained Models** are available [here](https://github.com/johnwlambert/tbv/releases/tag/v0.1_tbv_pretrained_models). Each model has an associated rendering config, training config, and model checkpoint file, all with the same uuid (e.g. `0589cca2-72aa-4626-9a05-af60eeea5fb6`).

To run model inference with a model trained to operate on the **ego-view**:
```bash
python scripts/test.py \
    --rendering_config_name 6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8.yaml \
    --training_config_name 6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8.yaml \
    --gpu_ids 0 \
    --save_inference_viz False \
    --split val \
    --ckpt_fpath ~/Downloads/6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8.pth \
    --filter_eval_by_visibility True
```

To run model inference with a model trained to operate on the **BEV**:
```bash
python scripts/test.py \
    --rendering_config_name 0589cca2-72aa-4626-9a05-af60eeea5fb6.yaml \
    --training_config_name 0589cca2-72aa-4626-9a05-af60eeea5fb6.yaml \
    --gpu_ids 0 \
    --save_inference_viz False \
    --split val \
    --ckpt_fpath ~/Downloads/0589cca2-72aa-4626-9a05-af60eeea5fb6.pth \
    --filter_eval_by_visibility False \
```

<a name='pretrained-model-accuracies'></a>
## Pre-trained Model Accuracies
Below, we provide the accuracies of the released pre-trained models on the val and test sets. Mean accuracies below are over the visible region (see asterisk *). All use early-fusion, except otherwise specified.

| Model UUID |  Model Description | Sensor Input | Map Input | Semantic Label Map Input | (Real) Val mAcc* | (Real) Test mAcc* |
| :--------: | :----------------: | :----: | :--:| :--------:| :--------:| :---------:|
| 6d3bfc13-1da4-49f0-bf6d-f6b1fc9647d8 | egoview, dropout either map or semantics, 100% prob | :white_check_mark:| :white_check_mark:| :white_check_mark:| 0.7031 | 0.7013 |
| 9dcfaa3f-a6af-4295-93ca-24a84d6b9c2d | egoview, 224x224, larger batch size |:white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6916 | 0.6843 |
| 98e50a71-c7b2-411d-99f4-781826488a26 | egoview, blurred input, independent map dropout_prob: 0.5, independent semantics dropout prob: 0.0 | :white_check_mark:| :white_check_mark:| :white_check_mark:| 0.7138 | 0.6826 |
| 2a3550a4-7b3d-4ab1-8165-e20d7cb069c9 | egoview, early fusion, but dropout either semantics or sensor, 0% prob | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6923 | 0.6747    
| 0061c32d-da98-4583-a311-8f2fc37b6655 | egoview, independent map dropout_prob: 0.0, independent semantics dropout prob: 0.5 | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6850 | 0.6697 |
| 44e55ee6-76da-4995-8fff-f4a2a4c3a8af | egoview, independent semantics dropout prob: 0.75, independent map dropout prob: 0.75 | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6735 | 0.6766 |
| 4d4f41a2-4bfe-42f2-88d0-1db253eeb9be | egoview, high res, all 3 modalities w/ dropout, 448x448 | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6732 | 0.6589 |
| 17fd2c0a-fee5-47c5-92cc-8b37f4479a8b | egoview, independent semantics dropout prob: 0.75, independent map dropout prob: 0.50 | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6683 | 0.6606 |
| b3ef41e8-db72-4e12-808d-353e4cd54280 | egoview 224x224  | | :white_check_mark:| :white_check_mark: | 0.6677 | 0.6183 |
| 88f42746-3374-40f7-a015-625652ca62c8 | egoview, blurred, independent semantics dropout prob: 0.25, independent map dropout prob: 0.25 | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6762 | 0.6604 |
| fe460247-d73f-4519-8643-ff38f95fb3b7 | egoview, 0% dropout, and with blur | :white_check_mark:| :white_check_mark:| :white_check_mark: | 0.6781 | 0.6373 |
| e3411e4e-87a6-4539-9ebb-1fcd6f99e601 | egoview, Resnet-50 | :white_check_mark: | :white_check_mark: | | 0.6505 | 0.6442 |
| 36b24988-5c54-46b4-9c22-cd48f70ae9f6 | egoview, 120 epochs |:white_check_mark: | :white_check_mark: | | 0.6533 | 0.6169 |
| 9b170dcf-6ba8-41d0-9ff2-fc3faf92e514 | egoview, 60 epochs,  no multiple negatives | :white_check_mark: | :white_check_mark: | | 0.6085 | 0.6362 |
| b5a2198c-5368-4feb-85c5-843a5646ecfa | egoview, map-only | | :white_check_mark: | | 0.5512 | 0.5364 |
| 615683e4-8431-4b31-ba1d-3523e6165fa9 | egoview, late fusion | :white_check_mark: | :white_check_mark: | | 0.5453 | 0.4963 |
| 0589cca2-72aa-4626-9a05-af60eeea5fb6 | BEV,  Resnet-50, 30 epochs | :white_check_mark: | :white_check_mark: | | 0.6588 | 0.6448 |
| e0ac04a5-c883-4497-81ef-e89ef3d23fdb | BEV, late fusion | :white_check_mark: | :white_check_mark: | | 0.6207 | 0.5450 |


<a name='generating-seamseg-label-maps'></a>
## Generating `seamseg` semantic segmentation label maps
Some models use `seamseg` label maps. To generate them, follow the steps below:

Clone the following fork of `seamseg`: https://github.com/johnwlambert/seamseg

<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/162601727-0d5ede18-6092-4942-92b8-310d7ffc3956.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/162601729-32df0429-0882-4b62-89b7-5ca4ed28664a.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/162601730-8961720d-8c7b-4cf3-8d7d-6adb0028c68f.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/162601731-53cc14c1-aa50-41ae-a22c-09b0b57b4338.jpeg" height="215">
</p>

Download the `seamseg` `seamseg_r50_vistas.zip` model [here](https://drive.google.com/file/d/1ULhd_CZ24L8FnI9lZ2H6Xuf03n6NA_-Y/view), or using the following bash commands:

```bash
export GDRIVE_FILEID='1ULhd_CZ24L8FnI9lZ2H6Xuf03n6NA_-Y'
export GDRIVE_URL='https://docs.google.com/uc?export=download&id='$GDRIVE_FILEID
wget --save-cookies cookies.txt $GDRIVE_URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
wget --load-cookies cookies.txt -O seamseg_r50_vistas.zip $GDRIVE_URL'&confirm='$(<confirm.txt)
```

Install `inplace-abn`: `pip install git+https://github.com/mapillary/inplace_abn.git`, and unzip the downloaded `.zip` file:

```bash
mkdir seamseg_pretrained_models
unzip seamseg_r50_vistas.zip -d seamseg_pretrained_models
```
You should see
```bash
ls -l seamseg_pretrained_models 
config.ini
metadata.bin
seamseg_r50_vistas.tar
```
Pass `/path/to/seamseg_pretrained_models` as `seamseg_model_dirpath` to `python run_seamseg_over_logs.py` as follows:
```bash
cd seamseg/scripts
mkdir ../logs
python ../../tbv-staging/scripts/run_seamseg_over_logs.py --tbv-dataroot /tbv_dataset/logs_raw --seamseg_output_dataroot /tbv_dataset/seamseg_output --num-processes 1 --split test --seamseg_model_dirpath /path/to/seamseg_pretrained_models
```

<a name='citing'></a>
## Citing this work
```BibTeX
@inproceedings{Lambert21neurips_TrustButVerifyHDMapChangeDetection,
 author = {Lambert, John W. and Hays, James},
 booktitle = {Advances in Neural Information Processing Systems Track on Datasets and Benchmarks},
 title = {{Trust, but Verify}: Cross-Modality Fusion for HD Map Change Detection},
 url = {https://openreview.net/pdf?id=cXCZnLjDm4s},
 year = {2021}
}
```

<a name='faq'></a>
## FAQ:

**Q**: Is there a file that indicates different log pairs and what has changed?

**A**: We provide a clustering of logs by spatial location [here](tbv/scene_clusters.py). A few things to note:

- Each log within a cluster shares some significant visual overlap with other logs within its cluster.
- These are not necessarily before/after pairs. In some cases, all logs in a cluster may be "after" a change.
- Each cluster has at least one log in the val or test set.
- Logs of each cluster are provided in chronological order.

**Q**: Where can I find the data splits?

**A**: Official train, val, test data splits are available [here](tbv/splits.py). There are 799 train logs, 111 val logs, and 133 test logs.

**Q**: Will labels be released for which logs include change/no change? Wanted to verify that the training sets have no changes. Therefore, all we need to know is if a log is in the training set to know the label?

**A**: Yes. Val split labels can be found [here](labeled_data/tbv_val_split_annotations.json). All train logs are *positive* logs that contain no changes. Most of the val and test logs contain at least some change (*negatives*), although some are positive "before" logs.

**Q**: TbV doesn't necessarily have before and after sensor data? So it seems we're just checking if the corresponding vector map is up-to-date or not?

**A**: Correct. We certainly do have many pairs of sensor data before/after in the dataset, but our goal was to be able to not have to store all past sensor data when we want to make an online map change prediction (for the TbV paper's experiments, we assume online online sensor data, and an onboard map).

**Q**: How you make your decision on change: In your paper, you mention that each change task is given a buffer of sensor data from time 0 to t, but in your model architectures in figure 3, I can't figure out how you incorporate the buffer. Is it at each time stamp, a change decision is made, and then you average the decision from all of the time stamps?
**A**: Using the buffer is not strictly necessary, but in some cases, it can be useful to have. Having a buffer of past info is also fairly realistic w.r.t. onboard settings. For the bird's eye view models we trained, we used a ring buffer to keep around the past 3d points w/ their RGB values, to make a richer input texture map (see code [here](tbv/rendering/orthoimagery_generator.py#L113)). For the ego-view models, we didn't use a buffer of sensor data, but there would be ways to feed into a buffer of data as input. We discuss this a bit [Appendix F, page 5 of the supplement](https://openreview.net/attachment?id=cXCZnLjDm4s&name=supplementary_material).

**Q**: I can't compile the `tbv_raytracing` package?
**A**: Check that the version of your driver (`cat /proc/driver/nvidia/version`) is compatible with your `cuda-toolkit` version (`torch.version.cuda`), according to the [NVIDIA compatibility CUDA/driver docs](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html).


[//]: # (row 1)
[//]: # (YEDRWy1MYuf5IONz4gQmQwAVuVQzkovm__2020-07-02-Z1F0055, YEDRWy1MYuf5IONz4gQmQwAVuVQzkovm__2020-11-10-Z1F0014, blank)
[//]: # (Nr6t0auYyTEC42fJNIqhkaSasyGjfV6E__2020-06-22-Z1F0064, Nr6t0auYyTEC42fJNIqhkaSasyGjfV6E__2020-10-12-Z1F0052)

<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/131888454-791e26a0-ba8c-4152-b510-eac6acc6e8b0.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131888462-f372ba49-4ea7-426a-922f-6aa805983616.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131890666-e917acce-9329-49d4-aaf4-22d93b0165f8.jpg" height="65">
  <img src="https://user-images.githubusercontent.com/16724970/131889151-6cd72465-0fec-4be6-882e-18078c9ad48c.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131889153-c76d60fd-24a0-4f61-82c6-114e616ca9b6.jpeg" height="215">
</p>

[//]: # (row 2)
[//]: # (bjalGQhAZWMLh50K0poYHX6GcXxnJPom__2020-06-23-Z1F0042, 9nS3_LogId79188___2020_10_22)
[//]: # (pbADFDy5ElABBs4vTFGnGtkQjTqIDKyD__2020-06-10-Z1F0049, pbADFDy5ElABBs4vTFGnGtkQjTqIDKyD__2020-07-23-Z1F0012____v2)

<p align="center">
  <img src="https://user-images.githubusercontent.com/16724970/131891417-8da39671-adc1-45d2-bd91-e4b67f6673a4.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131891426-8737c7c7-40f5-4246-a101-30ac90c9743c.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131890666-e917acce-9329-49d4-aaf4-22d93b0165f8.jpg" height="65">
  <img src="https://user-images.githubusercontent.com/16724970/131893541-0ef130ee-7b37-45d5-b8fa-8cf600f0f39f.jpeg" height="215">
  <img src="https://user-images.githubusercontent.com/16724970/131893551-1e6142a1-ac96-4843-a3a2-e8083a035e3b.jpeg" height="215">
</p>


<a name='license'></a>
## License

All code within this repository and all data included in the TbV Dataset are governed by the CC BY-NC-SA 4.0 license. By downloading the software, you are agreeing to the terms of this license agreement. If you do not agree with these terms, you may not use or download this software. It may not be used for any commercial purpose. See **[LICENSE](./LICENSE)** for more details.

This code and dataset are owned by Argo AI, LLC (Licensor), but are distributed by John Lambert with Argo's permission under CC BY-NC-SA 4.0. 

Exclusive Remedy and Limitation of Liability: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's (you or your organization) use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.

Disclaimer of warranties: The software is provided "as-is" without warranty of any kind including any warranties of performance or merchantability or fitness for a particular use or purpose or of non-infringement. Licensee bears all risk relating to quality and performance of the software and related materials.

Copyright: The Software is owned by Licensor and is protected by United States copyright laws and applicable international treaties and/or conventions.

