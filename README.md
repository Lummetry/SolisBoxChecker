# VaporBoxCheck
VaporBoxCheck is a mini-app developed in Python that uses Pytorch and Tensorflow frameworks. This application can be used as a test tool in order to test different deployment setups on a wide range of hardware and software setups.

## Prerequisites:
### System Prerequisites:
* Python >= v3.7
#### _If GPU is available:_
* NVIDIA driver compatible with CUDA >= v10.2 (Info URL: https://docs.nvidia.com/deploy/cuda-compatibility/index.html)
* CUDA >= v10.2 (Info URL: https://developer.nvidia.com/cuda-downloads)
* CUDNN >= v7.6.5 (Info URL: https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

### Box Libraries:
* OpenCV >= 3.4.1 (Info URL: https://pypi.org/project/opencv-python/)
* Pytorch >= 1.7.1 (Info URL: https://pytorch.org/get-started/locally/)
* Torchvision >= 0.8.2 (Info URL: https://pytorch.org/get-started/locally/)
* TensorflowGPU/Tensorflow >= 2.0.0 (Info URL: https://www.tensorflow.org/install)

## Running application:
1. Create environment:
    <br>`conda create -n vaporbox anaconda opencv python=3.7`
    <br>`conda activate vaporbox`
    <br>`conda install -c anaconda tensorflow-gpu==2.1.0`
    <br>`conda install pytorch torchvision -c pytorch`
2. Git clone the project (including submodules): 
    <br>`git clone https://github.com/Lummetry/VaporBoxCheck.git --recurse-submodules`
3. Go to project folder: `cd VaporBoxCheck`
4. Run: `python main.py`
5. Wait for script to run and check results
6. Send results to Lummetry Team as indicated in the script results

## Example outputs:
Below you can find a range of outputs after testing the script on different operating systems, hardware platforms and environment setups.

1. Windows Results:

![Windows Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/windows.png)

2. Ubuntu Results:

![Ubuntu Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/ubuntu.png)

3. MacOS Results:

![MacOS Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/macos.png)
