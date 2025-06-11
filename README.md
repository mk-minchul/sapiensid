# SapiensID

SapiensID is an official repository for SapiensID: Foundation for Human Recognition CVPR 2025 (https://arxiv.org/pdf/2504.04708).

```
@inproceedings{kim2025sapiensid,
  title={SapiensID: Foundation for Human Recognition},
  author={Kim, Minchul and Ye, Dingqiang and Su, Yiyang and Liu, Feng and Liu, Xiaoming},
  booktitle={CVPR},
  year={2025}
}
```

There are two main components:
- SapiensID: modeling
- WebBody: dataset

## SapiensID
The core modeling and evaluation framework that provides:
- Person re-identification models and pipelines
- Pretrained models included
- Comprehensive evaluation on multiple datasets including:
  - PRCC
  - WebBody
  - Market-1501
  - MSMT17
  - LTCC
  - CelebReID
  - DeepChange
  - CCDA
  - CCVID
  - (we provide code for creating validation sets for each dataset but do not distribute the datasets)
- Support for both single and multi-GPU evaluation
- Feature extraction and metric computation
- Model training and inference pipelines

Refer to [tasks/sapiensID/README.md](tasks/sapiensID/README.md) for more details.

## WebBody
A dataset creation and management tool that:
- Downloads and processes images from web sources
- Handles image resizing and quality control
- Manages data organization and storage
- Supports parallel processing for large-scale dataset creation
- Integrates with wandb for monitoring download progress
- 

Refer to [WebBody/README.md](tasks/WebBody/README.md) for more details.