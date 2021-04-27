# Thexp-implement

Reimplement of some papers I interested. All implementations are based on my another pytorch framework [thexp](https://github.com/sailist/thexp), and are as elegant and simple as I can.

## How to run
```
git clone https://github.com/sailist/thexp-implement
```

You need to instal my another library [thexp](https://github.com/sailist/thexp).
```
pip install thexp
```

When first run, you need to specify the dataset root: open `data/dataxy.py`, and change the value of the variable `root`. That's the only thing you need to do before running some scripts.  

And, find the script in `trainers/` and directly run it.

For example, if you want to reproduce the result of fixmatch, you can find it in `trainers/semisupervised/fixmatch.py`, and run it by using the code below:
```
python trainers/semisupervised/fixmatch.py

# or
cd trainers/semisupervised
python fixmatch.py
``` 

> the working directory will be changed automaticaly, so you just need to run it.

## Implementation list

Here list the paper reimplemented in this repo.

### Supervised baseline
including WRN-28-2, WRN-28-10, and Resnet{20, 32, 44, 50, 56},

Only use basic cross-entropy loss to optimize the model, and the data use four common augmentation methods:weak, weak+mixup, strong, strong+mixup

> `weak` means random horizontal flip and random crop, and `strong` means `weak` + `RandAugment` , please see the code for details.

```
python3 trainers/supervised/mixup.py
python3 trainers/supervised/strong_aug.py
python3 trainers/supervised/weak_aug.py
python3 trainers/supervised/strong_mixup.py
```

For most case, Strong or Strong+mixup will reach the best results. 

> [1] RandAugment: Practical automated data augmentation
  with a reduced search space, https://arxiv.org/pdf/1909.13719.pdf
>
> [2] mixup: Beyond Empirical Risk Minimization, https://arxiv.org/abs/1710.09412 

### Semi-Supervised
see [semisupervised/README.md](https://github.com/thexp/thexp-implement/blob/master/thexp-implement/trainers/noisylabel/README.md) for details

reimplement list:
 - Interpolation Consistency Training for Semi-Supervised Learning
 - MixMatch: A Holistic Approach to Semi-Supervised Learning
 - FixMatch: Simplifying Semi-Supervised Learning with Consistency and Conﬁdence

### Noisy Label
see [noisylabel/README.md](https://github.com/thexp/thexp-implement/blob/master/thexp-implement/trainers/noisylabel/README.md) for details

reimplement list:
 - MentorNet: Learning Data-Driven Curriculum for Very Deep Neural Networks
 - Probabilistic End-to-end Noise Correction for Learning with Noisy Labels(Pencil)
 - Beyond Synthetic Noise: Deep Learning on Controlled Noisy Labels (MentorMix)
 - Distilling Effective Supervision from Severe Label Noise (IEG)
 - O2U-Net: A Simple Noisy Label Detection Approach for Deep Neural Networks

 
### Meta-learning
see [metalearning/README.md](https://github.com/thexp/thexp-implement/blob/master/thexp-implement/trainers/metalearning/README.md) for details

reimplement list:
 - Distilling Effective Supervision from Severe Label Noise (IEG)
 - Learning to Reweight Examples for Robust Deep Learning (L2R)

 