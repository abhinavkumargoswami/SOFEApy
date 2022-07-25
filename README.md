
# Project Title

A brief description of what this project does and who it's for


# SOFEApy

This package is a python implementation of the original Single-shot Optical Flow Estimation Algorithm (SOFEA), A Non-iterative and Robust Optical Flow Estimation Algorithm for Dynamic Vision Sensors. The original implementation is in MATLAB. It is available at https://bitbucket.org/wengfei_nus/sofea/src/master/
A brief introduction of the method is in the following video.
[![SOFEA](https://img.youtube.com/vi/mPItrhMn0JQ/0.jpg)](https://www.youtube.com/watch?v=mPItrhMn0JQ)

The CVPR 2020 workshop paper is available [here](https://openaccess.thecvf.com/content_CVPRW_2020/html/w6/Low_SOFEA_A_Non-Iterative_and_Robust_Optical_Flow_Estimation_Algorithm_for_CVPRW_2020_paper.html)

## Citation
Weng Fei Low, Zhi Gao, Cheng Xiang and Bharath Ramesh. "SOFEA: A Non-Iterative and Robust Optical Flow Estimation Algorithm for Dynamic Vision Sensors". In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2020, pp. 82-83.

```
@InProceedings{Low_2020_CVPR_Workshops,
author = {Low, Weng Fei and Gao, Zhi and Xiang, Cheng and Ramesh, Bharath},
title = {SOFEA: A Non-Iterative and Robust Optical Flow Estimation Algorithm for Dynamic Vision Sensors},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2020}
}
```

## Requirements
Python 3 with the following packages installed:
* scipy
* plotly
* numpy
* mat73
## Installation

Run the following command to install SOFEApy

```cmd
pip install -r requirements.txt
```

## Usage
Run the following command.
Select the name of the sequence using arg1.
Visualise the local spatial neighbourhood about the EOI using arg2.
```cmd
python main.py [-arg1] [-arg2] 
```
The result is saved as a mat file.

## Output

https://user-images.githubusercontent.com/63021565/180638651-29bd4e79-06c5-4e8e-8c8a-d46b936c9735.mp4

## License

[GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/)

