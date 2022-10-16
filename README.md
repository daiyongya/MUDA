### MUDA for Sentiment Analysis
This repo contains the source code for our paper:

[**Adversarial Training Based Multi-Source Unsupervised Domain Adaptation for Sentiment Analysis**](https://ojs.aaai.org//index.php/AAAI/article/view/6262)

### Requirements:
- Python 3.6
- PyTorch 0.4
- PyTorchNet
- scipy
- tqdm (for progress bar)

## Model Overview
### Brief Introduction
In this paper, we focus on the multi-source unsupervised domain adaptation for sentiment analysis and desire to combine the hypotheses of multiple 
labeled source domains to derive a good hypothesis for an unlabeled target domain. For this purpose, we introduce two transfer learning frameworks. 
The first framework is Weighting Scheme based Unsupervised Domain Adaptation (WS-UDA), in which we integrate the source classifiers to annotate 
pseudo labels for target instances directly. Our second framework is a Two-Stage Training based Unsupervised Domain Adaptation method (2ST-UDA), 
which further utilize pseudo labels to train a target-specific extractor.

Our model is divided into two parts. The first part is to get the pre-trained model of each module, and the second part is to use our 
pre-trained model to get the results of our two transfer learning frameworks.

## Training Process

### Training Tips: 
Using Microsoft's open source tuning tool [nni](https://github.com/microsoft/nni), the final result has a fluctuation of ±0.5%

### Before Running

The pre-trained word embeddings file exceeds the 100MB limit of github, and is thus provided as a gzipped tar ball.
Please run the following command to extract it first:

```
tar -xvf data/w2v/word2vec.tar.gz -C data/w2v/
```
Before starting to run the program, you must set the values of base_save_dir and exp2_model_save_file (exp3_model_save_file) 
to store the model and parameter files during the training process.

### Get The Pre-trained Model For Amazon review dataset 
```bash
cd code/
python3 get_pre-trained_model_exp2.py
```

### Exp 1: WS-UDA on the Amazon review dataset
```bash
cd code/
python3 WS-UDA_exp2.py
```

### Exp 2: 2ST-UDA on the Amazon review dataset
```bash
cd code/
python3 2ST-UDA_exp2.py
```

### Get The Pre-trained Model For Amazon review dataset 
```bash
cd code/
python3 get_pre-trained_model_exp3.py
```

### Exp 3: WS-UDA on the FDU-MTL dataset
```bash
cd code/
python3 WS-UDA_exp3.py
```

### Exp 4: 2ST-UDA on the FDU-MTL dataset
```bash
cd code/
python3 2ST-UDA_exp3.py
```