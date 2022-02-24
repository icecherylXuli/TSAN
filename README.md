# Transcoded Video Restoration by Temporal Spatial Auxiliary Network

#### Li Xu, Gang He, Jinjia Zhou, Weiying Xie, Yunsong Li, Yu-Wing Tai

#### Paper: https://arxiv.org/abs/2112.07948

## 1. Abstract

In most video platforms, such as Youtube, Kwai, and Tik-Tok, the played videos usually have undergone multiple video encodings such as hardware encoding by recording devices, software encoding by video editing apps, and single/multiple video transcoding by video application servers. Previous works in compressed video restoration typically assume the compression artifacts are caused by one-time encoding. Thus, the derived solution usually does not work very well in practice. In this paper, we propose a new method, temporal spatial auxiliary network (TSAN), for transcoded video restoration. Our method considers the unique traits between video encoding and transcoding, and we consider the initial shallow encoded videos as the intermediate labels to assist the network to conduct self-supervised attention training. In addition, we employ adjacent multi-frame information and propose the temporal deformable alignment and pyramidal spatial fusion for transcoded video restoration. The experimental results demonstrate that the performance of the proposed method is superior to that of the previous techniques.

## 2. Pre-request

### 2.1. Environment

- Ubuntu 16.04
- CUDA 10.1
- PyTorch 1.16
- Packages: tqdm, lmdb, pyyaml, opencv-python, scikit-image

### 2.2. DCNv2

**Build DCNv2.**

```bash
$ cd ops/dcn/
$ bash build.sh
```

**(Optional) Check if DCNv2 works.**

```bash
$ python simple_check.py
```

> The DCNv2 source files here is different from the [open-sourced version](https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch) due to incompatibility.[[issue]](https://github.com/open-mmlab/mmediting/issues/84#issuecomment-644974315)



## 3. Test

Test data and pre-trained models: [[Google Drive]](https://drive.google.com/drive/folders/1uPTuwkZTYBjq4Wm6ayaFYB0ByEumFvMd?usp=sharing) 

```bash
$ python test.py
```

## 4. Quantitative Results

![1645521230323](./images/Quantitative_Results.png)



## 5. Qualitative Results

![1645521230323](https://github.com/icecherylXuli/TSAN/blob/main/images/Qualitative_Results.PNG)



## Acknowledgement

The code is based on [STDF](https://github.com/ryanxingql/stdf-pytorch) and [BasicVSR](https://github.com/ckkelvinchan/BasicVSR-IconVSR).

### 
