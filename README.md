# To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models

Authors: Simone Gaisbauer, Prabin Gyawali

This repository contains the code to reproduce our results presented at the 13th International Conference on Mobile Mapping Technology (MMT) in Xiamen, on June 20-22, 2025. It is the result of the project work Photogrammetry and Remote Sensing at TUM in the winter semester 2024/25. 
> Gaisbauer, S., Gyawali, P., Zhang, Q., Wysocki, O., & Jutzi, B. (2025). To Glue or Not to Glue? Classical vs Learned Image Matching for Mobile Mapping Cameras to Textured Semantic 3D Building Models. arXiv preprint arXiv:2505.17973.

## Credits for the code
Our code is largely based on the repository [Glue Factory](https://github.com/cvg/glue-factory) (state 03.01.25) by the authors Philipp Lindenberger, Paul-Edouard Sarlin, Rémi Pautrat, and Iago Suárez. The following articles are associated with this repository:

> Lindenberger, P., Sarlin, P.-E., Pollefeys, M., 2023. Lightglue: Local feature matching at light speed. Proceedings of the IEEE/CVF International Conference on Computer Vision, 17627–17638.

> Pautrat*, R., Súarez*, I., Yu, Y., Pollefeys, M., Larsson, V., 2023. GlueStick: Robust Image Matching by Sticking Points and Lines Together. International Conference on Computer Vision (ICCV).

We include their [readme](gluefactory_docs/gluefactory_README.md), [license](gluefactory_docs/gluefactory_LICENSE), and [instructions on running the evaluations](gluefactory_docs/gluefactory_evaluation.md) in this repository.

### Used as-is from Glue Factory
Since Glue Factory provides a bunch of functionalities, we only include the files necessary for our evaluations in this repository.

The following methods are directly used from Glue Factory:
- Evaluation for HPatches and Megadepth1500
- Learnable methods SuperPoint, SuperGlue, LightGlue, LoFTR, DISK
- Classical methods SIFT
- Visualization methods


### Modifications: Additional methods and datasets for evaluation
Our modifications concern the addition of the following methods and config files for evaluating them on HPatches and Megadepth1500.  
- Classical matchers: Nearest neighbours and FLANN from OpenCV
- Classical detectors and descriptors: AKAZE and ORB from OpenCV
- Learnable methods: Added grayscale conversion for LoFTR

We also made minor modifications to make things run for our application, e.g. in the config structure such that it is compatible with our datasets and methods.

Our second contribution is the addition of the TUM facade dataset. Our code for the custom evaluation for this dataset follows the Glue Factory code for image pair evaluation. We added the following methods:
- Tum evaluation pipeline
- Geometry for absolute poses
- Robust estimator for the absolute pose using OpenCV
- Custom two-view plot for visualizing the projection error of the matches


### Further dependencies
We specify the used libraries in the [dependencies.txt](/dependencies.txt) file. They can be installed with pip using:
```bash
pip install -r dependencies.txt
```

### License information
The Glue Factory repository itself is under [Apache-2.0 license](/gluefactory_docs/gluefactory_LICENSE). [SuperPoint](https://github.com/magicleap/SuperPointPretrainedNetwork) and [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork) follow restrictive licenses ([SuperPoint license](https://github.com/magicleap/SuperPointPretrainedNetwork?tab=License-1-ov-file#License-1-ov-file), [SuperGlue license](https://github.com/magicleap/SuperGluePretrainedNetwork?tab=License-1-ov-file)). These methods are included in Glue Factory in `gluefactory_nonfree` and we did not make any changes to these files. As noted in [Glue Factory's readme](gluefactory_README.md), also other methods might follow their own license.

Due to the restrictive licenses of the nonfree methods, we do not include this folder in our repository. If you would like to test the non-free methods with this repository, please retrieve the `gluefactory_nonfree` directly from the [Glue Factory's original repository](https://github.com/cvg/glue-factory).

## Credits for the datasets

### Generic datasets

We use the datasets HPatches and Megadepth1500 as provided by Glue Factory. The datasets are automatically downloaded when running inspect or eval on them.

### Custom TUM dataset

We evaluate on two custom datsets facade texture - car image pairs and facade texture - UAV image pairs. The facade images are already included in this repository, whereas the car images and UAV images are automatically downloaded when running inspect or eval on them.

#### Facade textures

We use facade texture images of LoD2 CityGML models from the [TUM2TWIN](https://tum2t.win/) project. They were retrieved from [tum2twin-datasets](https://gitlab.lrz.de/tum-gis/tum2twin-datasets/-/tree/main/citygml/lod2-textured-building-datasets) and are subject to [CC BY 4.0](https://gitlab.lrz.de/tum-gis/tum2twin-datasets/-/blob/main/LICENSE) license.

> Wysocki, O., Schwab, B., Biswanath, M. K., Zhang, Q., Zhu, J., Froech, T., ... & Jutzi, B. (2025). TUM2TWIN: Introducing the Large-Scale Multimodal Urban Digital Twin Benchmark Dataset. arXiv preprint arXiv:2505.07396.

#### UAV images

We use a small subset of images from the photogrammetric UAV reconstruction from the [TUM2TWIN](https://tum2t.win/) project. The full datset is available [here](https://zenodo.org/records/14899378) and is subject to [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode). The associated publication is:

> Anders, K., Wang, J., Wysocki, O., Huang, X., & Liu, S. (2025). UAV Laser Scanning and Photogrammetry of TUM Downtown Campus (1.1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14899378

#### Car images

The 343 car-based images for our second dataset were provided by the company [3D Mapping Solutions]("https://www.3d-mapping.de/"). We make them available with their permission under the followig license: Mobile Mapping Images of TUM main campus © 2025 by <a href="https://www.3d-mapping.de/">3D Mapping Solutions</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-sa/4.0/">CC BY-NC-SA 4.0</a>.

## Run instructions

The following commands are used to run and inspect the evaluation on the TUM facade - car/UAV image pairs. Please note that the selection of car or UAV images is done via different config files, which have an suffix "_uav" for the UAV dataset. For more detailled instructions on how to run gluefactory, we refer to their [evaluation instructions](gluefactory_evaluation.md).

- To run the TUM facade image evaluation with one of the [config files](gluefactory/configs):
```bash
python -m gluefactory.eval.tum --conf [config]
```
- To inspect the TUM facade image evaluation with several previously run configurations:
```bash
python -m gluefactory.eval.inspect tum [config1] [config2] [...]
```

### Example for TUM facade-car dataset

- Run the TUM evaluation with Superpoint+Lightglue on the facade-car dataset:
```bash
python -m gluefactory.eval.tum --conf superpoint+lightglue-official
```
- Run the TUM evaluation with sift+FLANN on the facade-car dataset:
```bash
python -m gluefactory.eval.tum --conf sift+FLANN
```
- Inspect both results:
```bash
python -m gluefactory.eval.inspect tum superpoint+lightglue-official sift+FLANN 
```

### Example for TUM facade-uav dataset

- Run the TUM evaluation with Superpoint+Lightglue on the facade-UAV dataset:
```bash
python -m gluefactory.eval.tum --conf superpoint+lightglue-official_uav
```
- Run the TUM evaluation with sift+FLANN on the facade-UAV dataset:
```bash
python -m gluefactory.eval.tum --conf sift+FLANN_uav
```
- Inspect both results:
```bash
python -m gluefactory.eval.inspect tum superpoint+lightglue-official_uav sift+FLANN_uav 
```
