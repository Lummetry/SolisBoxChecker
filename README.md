# VaporBoxCheck
VaporBoxCheck is a mini-app developed in Python that uses Pytorch and Tensorflow frameworks. This application can be used as a test tool in order to test different deployment setups on a wide range of hardware and software setups.

## Preparing NVIDIA Jetson environment:
In order to setup NVIDIA Jetson envrionment you can access following pdf for a stept-by-step tutorial:
[Jetson Setup](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/JetsonSetup.pdf).
Please note that this step-by-step tutorial has been done using a TX2 Developer Kit device.

## Preparing non-Jetson environments:
1. Install Anaconda or Minicoda
2. Run environment setup:
```
    conda create -n vaporbox anaconda opencv python=3.7
    conda activate vaporbox
    conda install -c anaconda tensorflow-gpu==2.1.0
    conda install pytorch torchvision -c pytorch
```
    
## Running VaporBoxCheck
1. Git clone the project (including submodules): 
    <br>`git clone https://github.com/Lummetry/VaporBoxCheck.git --recurse-submodules`
2. Go to project folder: `cd VaporBoxCheck`
3. Run: `python run.py`
4. Wait for script to run and check results
5. Send results to Lummetry Team as indicated in the script results

## Example outputs:
Below you can find a range of outputs after testing the script on different operating systems, hardware platforms and environment setups.

1. Windows Results:

![Windows Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/windows.png)

2. Ubuntu Results:

![Ubuntu Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/ubuntu.png)

3. MacOS Results:

![MacOS Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/macos.png)

4. JetsonTX2 Results:

![JetsonTX2 Results](https://github.com/Lummetry/VaporBoxCheck/blob/main/_vapor_box_check/_output/jetson.png)
