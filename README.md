## Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection (NeurIPS '21, Official Repo)
[John Lambert](https://johnwlambert.github.io/), [James Hays](https://www.cc.gatech.edu/~hays/)

This repository contains the source code for training and evaluating models described in the NeurIPS '21 paper *Trust, but Verify: Cross-Modality Fusion for HD Map Change Detection*. [[Paper]](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6f4922f45568161a8cdf4ad2299f6d23-Paper-round2.pdf) [[Supplementary Material]](https://openreview.net/attachment?id=cXCZnLjDm4s&name=supplementary_material)

The Trust but Verify (TbV) dataset is publicly available for download, as part of the [**Argoverse 2.0** family of datasets](https://www.argoverse.org/av2.html). Download instructions can be found [here](https://github.com/argoai/av2-api/blob/main/src/av2/datasets/tbv/README.md#downloading-tbv). You can find a short invited talk at the CVPR 2021 VOCVALC workshop summarizing our work [here on Youtube](https://youtu.be/JeTZbCuyeM8?t=3735).

## Dataset Overview

The Trust but Verify (TbV) dataset is the first public dataset for the task of high-definition (HD) map change detection, i.e. determining when sensor data and map data are no longer in agreement with one another due to real-world changes. We collected TbV by mining thousands of hours of data from over 9 months of autonomous vehicle fleet operations.

![92me_Jd8_hstacked](https://user-images.githubusercontent.com/16724970/160265919-99b300fa-d7b9-4eaa-8889-46f8f142ee52.gif)
![cJiXg_jDONA](https://user-images.githubusercontent.com/16724970/160265925-174a2958-e8d6-42ac-84a2-8e012b9e4e71.gif)
![92me_Jd8_hstacked_map](https://user-images.githubusercontent.com/16724970/160266065-7cbccc84-daa7-4129-92da-0f50569f0aed.gif)

![0f0c_Aom_hstack](https://user-images.githubusercontent.com/29715011/160264648-cfad3571-3bfa-4acb-9831-d8f8693a7dc8.gif)
![xpxg_4ab4](https://user-images.githubusercontent.com/29715011/160264672-3e725689-395f-4fd5-847f-d9ba273e8c5c.gif)
![Hl998_zU1P_hstacked](https://user-images.githubusercontent.com/29715011/160264678-760a8167-0b1e-442d-8d73-e0487868c40a.gif)


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

The dataset, consisting of maps and logs collected in six North American cities, is one of the largest AV datasets to date with more than 7.9 million images and will be made available to the public, along with code and models under the the CC BY-NC-SA 4.0 license. Above, we show before-and-after images that showcase a few examples of map changes featured in TbV logs.

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

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
Compile the GPU library using `setup.py` as follows:
```bash
python setup.py bdist_wheel
pip install dist/tbv_raytracing-0.0.1-cp{PY_VERSION}-cp{PY_VERSION}-linux_x86_64.whl
```
e.g. this file could be named one of the following:
```python
pip install dist/tbv_raytracing-0.0.1-cp38-cp38-linux_x86_64.whl
pip install dist/tbv_raytracing-0.0.1-cp310-cp310-linux_x86_64.whl
```

## Downloading the dataset

Download the dataset per the instructions found [here](https://github.com/argoai/av2-api/blob/main/src/av2/datasets/tbv/README.md#downloading-tbv).

Create a folder, and then `logs/`

## Rendering Training Data

To render data in a bird's eye view, run
```bash
python scripts/run_dataset_rendering_job.py --config_name bev_config.yaml
```

To render data in the ego-view, run
```bash
python scripts/run_dataset_rendering_job.py --config_name egoview_config.yaml
```

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


## Training Models

After rendering a dataset, you're ready to train models. Start training by running:
```bash
python scripts/train.py --config_name {CONFIG_NAME}
```

## Model Evaluation

```bash
python scripts/test.py --config_name {CONFIG_NAME}
```

**Pretrained Models**: will be released shortly.

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

## License

All code within this repository and all data included in the TbV Dataset are governed by the CC BY-NC-SA 4.0 license. By downloading the software, you are agreeing to the terms of this license agreement. If you do not agree with these terms, you may not use or download this software. It may not be used for any commercial purpose. See **[LICENSE](./LICENSE)** for more details.

This code and dataset are owned by Argo AI, LLC (Licensor), but are distributed by John Lambert with Argo's permission under CC BY-NC-SA 4.0. 

Exclusive Remedy and Limitation of Liability: To the maximum extent permitted under applicable law, Licensor shall not be liable for direct, indirect, special, incidental, or consequential damages or lost profits related to Licensee's (you or your organization) use of and/or inability to use the Software, even if Licensor is advised of the possibility of such damage.

Disclaimer of warranties: The software is provided "as-is" without warranty of any kind including any warranties of performance or merchantability or fitness for a particular use or purpose or of non-infringement. Licensee bears all risk relating to quality and performance of the software and related materials.

Copyright: The Software is owned by Licensor and is protected by United States copyright laws and applicable international treaties and/or conventions.


## FAQ:
**Q**: How you make your decision on change: In your paper, you mention that each change task is given a buffer of sensor data from time 0 to t, but in your model architectures in figure 3, I can't figure out how you incorporate the buffer. Is it at each time stamp, a change decision is made, and then you average the decision from all of the time stamps?
**A**: Using the buffer is not strictly necessary, but in some cases, it can be useful to have. Having a buffer of past info is also fairly realistic w.r.t. onboard settings. For the bird's eye view models we trained, we used a ring buffer to keep around the past 3d points w/ their RGB values, to make a richer input texture map (see code [here](https://github.com/johnwlambert/tbv/blob/main/tbv/rendering/make_bev_rgb_img.py#L118-L123). For the ego-view models, we didn't use a buffer of sensor data, but there would be ways to feed into a buffer of data as input. We discuss this a bit [Appendix F, page 5 of the supplement](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/file/6f4922f45568161a8cdf4ad2299f6d23-Supplemental-round2.zip).
