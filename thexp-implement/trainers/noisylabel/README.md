# Noisy Label
## MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks

[[paper]](https://arxiv.org/abs/1712.05055) [[official]](https://github.com/google/mentornet)
 - [ ] Todo 
 
 
## Probabilistic End-to-end Noise Correction for Learning with Noisy Labels(Pencil)
[[paper]](https://arxiv.org/abs/1903.07788) [[official]](https://github.com/ljmiao/PENCIL)

**my reimplement**
```
python3 trainers/noisylabel/pencil.py
```
 
## Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels (MentorMix)
[[paper]](https://arxiv.org/abs/1911.09781) [[official]](https://github.com/google-research/google-research/tree/a28d3e008df5f023d915cb644c5ab02c57599957/mentormix)

 - [ ] Todo


## Distilling Effective Supervision from Severe Label Noise (IEG)

> Another name "IEG: Robust neural net training with severe label noises" 

 [[paper]](https://arxiv.org/abs/1910.00701) [[official]](https://github.com/google-research/google-research/tree/master/ieg)
 
use wideresnet-28-2, and reproduce the result on cifar10 with 40%, 80% synthetic noisy.(I can't run wideresnet-28-10 on my single GPU.) 
 
 

**my reimplement**
```
python3 trainers/noisylabel/ieg_noisy_label.py --noisy_ratio=0.8
```

 **cifar10 result** 
 
 |noisy ratio|results|
 |---|---|
 |0.4|95.01|
 |0.8|94.81|
 
> train cifar100 may need larger model(like WRN28-10 or others).

## O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks

 [[paper]](https://openaccess.thecvf.com/content_ICCV_2019/papers/Huang_O2U-Net_A_Simple_Noisy_Label_Detection_Approach_for_Deep_Neural_ICCV_2019_paper.pdf)
[[official]](https://github.com/hjimce/O2U-Net)
 
 **my reimplement**
```
python3 trainers/noisylabel/o2u.py --noisy_ratio=0.8
```

> Currently can't reproduce the results.


## DivideMix: Learning with Noisy Labels as Semi-supervised Learning
 
 [[paper]](https://arxiv.org/abs/2002.07394) [[official]](https://github.com/LiJunnan1992/DivideMix)

```
python3 trainers/noisylabel/dividemix.py --noisy_ratio=0.8
```

 - [ ] run