# Conversational Emotion-Cause Pair Extraction with Guided Mixture of Experts

This repository is the official implementation of [Conversational Emotion-Cause Pair Extraction with Guided Mixture of Experts]. 

![model_v5_2](https://user-images.githubusercontent.com/57910677/218022327-6bf55a88-d780-40a5-b2c8-878361567a32.png)

## Dependencies

- Python 3.9.7
- PyTorch 1.11.0
- NumPy 1.20.3
- sickit-learn 0.24.2
- transformers 4.11.3
- CUDA 11.4
- tqdm 4.62.3

## Usage

To train (or test) the model in the paper, run this command:
```
python main.py
```

The following command automatically performs training and evaluation for all data.
```
python train.py
```

## Dataset
The dataset used in this paper is RECCON dataset. See [this link](https://github.com/declare-lab/RECCON) for more information.

## Citation

```
DongJin Jeong and JinYeong Bak. 2023. Conversational Emotion-Cause Pair Extraction with Guided Mixture of Experts. In Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 3288–3298, Dubrovnik, Croatia. Association for Computational Linguistics.
```
