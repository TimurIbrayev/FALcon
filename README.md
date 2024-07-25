# Exploring Foveation and Saccade for Improved Weakly-Supervised Localization
**FALcon** is an active object localization framework incorporating the bio-plausible mechanisms of foveation and saccades for improved and resilient weakly supervised object localization.

The work was presented at _NeurIPS 2023 Workshop on Gaze Meets ML (GMML)_ and published at _Proceedings of Machine Learning Research (PMLR)_. The manuscript can be found at [this link](https://proceedings.mlr.press/v226/ibrayev24a/ibrayev24a.pdf).
<div align="center">
  <img src="poster/FALcon_poster_NeurIPS23_GMML_snippet.jpg" alt="Poster Snippet" width="900"/>
</div>
This is the official repository providing the code, model weights, and data to enable reproducibility of the work.

## Required Packages
- Python 3.8 or higher
- PyTorch 1.9 or higher

## Data
The project works on the following datasets, with the files expected to be in the corresponding structures:

<details>
  <summary>CUB-200-2011</summary>
  
  ```
  ├── README
  ├── attributes
  │   ├── attributes.txt
  │   ├── certainties.txt
  │   ├── class_attribute_labels_continuous.txt
  │   └── image_attribute_labels.txt
  ├── bounding_boxes.txt
  ├── classes.txt
  ├── image_class_labels.txt
  ├── images
  │   ├── 001.Black_footed_Albatross
  │   ├── 002.Laysan_Albatross
  │   ├── 003.Sooty_Albatross
  │   ├── ...
  │   └── 200.Common_Yellowthroat
  ├── images.txt
  ├── parts
  │   ├── part_click_locs.txt
  │   ├── part_locs.txt
  │   └── parts.txt
  └── train_test_val_split.txt
  ```
</details>

<details>
  <summary>ImageNet</summary>
  
  ```
├── ILSVRC2012_devkit_t12
├── ILSVRC2012_devkit_t3
├── anno_train
│   ├── n01440764
│   ├── ...
│   └── n15075141
├── anno_val
│   ├── n01440764
│   ├── ...
│   └── n15075141
├── test
├── train
│   ├── n01440764
│   ├── ...
│   └── n15075141
├── val
│   ├── n01440764
│   ├── ...
│   └── n15075141
├── anno_valprep.sh
└── valprep.sh
  ```
</details>


## Usage

The framework is implemented as a set of separate script for each individual operation, e.g. ```FALcon_train_cub``` for training FALcon on CUB-200-2011 dataset samples.

The operations of different scripts are controlled by one of the configuration scripts, e.g. ```FALcon_config_cub``` for the operations on CUB-200-2011 dataset samples.

Please, address the dropdown below for the short description of scripts and directories in the repository.
  <details>
  <summary>Descriptions of scripts</summary>
  
  | Script name | Script role |
  |-------------|-------------|
  |```FALcon_train_{dataset}```| Trains FALcon on the corresponding dataset. |
  |```FALcon_config_*```| Specifies different operational parameters for the execution of scripts. |
  |```FALcon_test_as_WSOL```| Tests FALcon on the dataset specified in ```FALcon_config_test_as_WSOL```.  |
  |```FALcon_models```| Contains class of VGG-like models for the implementation of FALcon. |
  |```FALcon_collect_{dataset}```| Runs FALcon on the samples of the corresponding dataset. Used to partition "long runs" of data collection. |
  |```AVS_functions```| Contains functions specific to the framework. |
  |```cls_models```| Contains various models used for the classification by FALcon and/or PSOL. |
  |```psol_*```| Performs the similar operations as the corresponding FALcon scripts, but using only PSOL framework. |
  |```utils/```| Contains a set of helper functions, not necessarily specific to the proposed framework. |
  |```PSOL/```| Contains a copy of PSOL github code, which incorporates our re-implemented scripts for training models on CUB-200-2011 dataset. |

</details>

_Please, note:_ the code in this repository was cleaned up and restructured for a better readibility, which might cause some errors with references to imports and/or data. If you face difficulties, please let us know by submitting an issue!

## Model Weights
▶ Please, use [this link](https://purdue0-my.sharepoint.com/:f:/g/personal/tibrayev_purdue_edu/EmflqrsUu5xEiomrjtCGsaABLYI-hRoQnCduhlQ41c6ffw) to find a shared OneDrive folder, which contain model parameters trained on CUB-200-2011 or ImageNet2012 datasets. The structure of the shared drive mirrors the structure of this repository, meaning that the ```.pth``` checkpoints should be placed into the corresponding folders.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Contact Information
Please, do not hesitate to reach out the corresponding author, Timur Ibrayev, via [email](mailto:tibrayev@purdue.edu?subject=[GitHub]%20FALcon%20repo) or on [LinkedIn](https://www.linkedin.com/in/timuribrayev).

Alternatively, feel free to open an issue or propose a new feature here on the GitHub. 

We would appreciate your feedback and contributions!

## Citation
If you use the framework in your work, please consider citing the original research paper:

```
@inproceedings{pmlr-v226-ibrayev24a,
  title = {Exploring Foveation and Saccade for Improved Weakly-Supervised Localization},
  author = {Ibrayev, Timur and Nagaraj, Manish and Mukherjee, Amitangshu and Roy, Kaushik},
  booktitle = {Proceedings of The 2nd Gaze Meets ML workshop},
  pages = {61--89},
  year = {2024},
  editor = {Madu Blessing, Amarachi and Wu, Joy and Zanca, Dario and Krupinski, Elizabeth and Kashyap, Satyananda and Karargyris, Alexandros},
  volume = {226},
  series = {Proceedings of Machine Learning Research},
  month = {16 Dec},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v226/ibrayev24a.html}
}
