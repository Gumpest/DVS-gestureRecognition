# DVS-gestureRecognition-ANN2SNN

已完成工作：

使用DVS CeleX_MP获取脉冲信号，输入SNN中，获得手势。

通过SNN || ANN的方法，训练dvs手势集，部署在C++上，接入摄像机的脉冲，完成推理。

目标：

将网络通过ANN2SNN工具链，得到SNN的配置帧，完成部署。

## 自制数据集(DVSGesture)

* 使用DVS摄像机拍摄 (CeleX_MP)
* 格式csv ：<x , y, p , t>

* 共有5种手势 (有待扩充，尤其顺逆时针手势、环境的变化、距离的变化)

* DataTrain：5 * 10, DataTest: 5 * 2

* 环境为晴天，拍摄距离约为2m

* **需要自制数据集的原因是：使用IBM数据集进行训练后，无法使用CeleX_MP获得的数据正确推理，正确率极低**

  **推断原因：两款摄像头架构不同，首先DVS128是随机输出事件，时间戳对应着当时像素发生变化的时间（in-pixel）；而CeleX_MP是按行输出事件，同时时间戳递增，这会造成Event的输出顺序和实际触发顺序不一致；因此识别率处于较低水平。**

```python
The demo data captured by CeleX5_MP
mapping = {
    0: 'arm roll',
    1: 'arm updown',
    2: 'hand clap',
    3: 'right hand clockwise ',
    4: 'right hand wave',
}
Author : Zhangyuan
```

附1：[芯仑摄像头SDK库](https://github.com/CelePixel/CeleX5-MIPI)（C++ API用于实时推理 & GUI用于制作数据集）

附2：[IBM DVS_Gesture](http://www.research.ibm.com/dvsgesture/)

附3：CeleX_SDK_Getting_Started_Guide

DVS_SDK 使用说明 分辨率为：1280*800 工作频率为 70MHz

环境配置 Visual Studio 2015 + OpenCV 3.3.0 + QT 5.8.0

## 数据集的训练(SNN&ANN)

**DVS获得的数据帧，天生含有时序信息，因此与SNN天然适配**

DVS（Dynamic Vision Sensor）输出为事件流<X , Y , P , T>，通过对自然光强变化的判定输出仿生的动态特征脉冲信号

因此供选定参考文章模型五个，均为SNN（脉冲神经网络）：论文仓库已提供

1、Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks （性能最佳，不开源）

2、Effective Sensor Fusion with Event-Based Sensors and Deep Network Architectures （使用theano，但是载入的数据集是传统MNIST）

3、Direct Training for Spiking Neural Networks: Faster, Larger, Better （无开源代码）

4、SLAYER: Spike Layer Error Reassignment in Time（附加开源代码与数据集）

5、Synaptic Plasticity Dynamics for Deep Continuous Local Learning（代码开源，且MNIST、Gesture都可以跑）

**最终**使用来自清华大学类脑中心提出的模型 – **STBP**，因其识别率最高、性能最佳，但是需要复现编写代码。

该模型使用频率编码，定义了不可导的激活函数，因此SpikeAct需要重载。

### 训练方法一

使用数据格式为<x,y,p,t>

文件夹DVSgesture_xypt

### 训练方法二

使用数据格式为<x,y,t>

文件夹DVSgesture_xyt

### 训练方法三

使用ann，同时将时间步映射在特征图上

文件夹DVSgesture-ANN2SNN

### 文件架构说明

* 项目文件
  * gesture -- 数据集
  * gesture.py -- 数据格式预处理 DVS类
  * train.py -- 训练
  * eval.py -- 推理
  * transScript.py -- 模型转换 py->C++

## 模型的使用与C++部署

**1、Libtorch（C++ API）下载**

注意：libtorch版本<=torch版本；因仅做推断，选择非CUDA版本即可，WINDOWS环境下只可使用debug版本，VS2015

本机安装torch.__version__=1.4.0，对应安装libtorch1.3.0（Win_CPU_DEBUG）

https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.3.0.zip

**2、将模型转换为序列化的PyTorch模型**

① 将PyTorch模型转换为Torch Script

注意：device改为CPU，因为libtorch是CPU版本，Forward函数依赖于控制流。

曾经尝试单一Trace或者注释的方法，即使PY端不报错，C++端依然加载时会异常。

因此使用混用方法，对更新函数进行改写，去掉bool（），并对其注释；生成模型使用追踪。

② 将Script Module序列化为一个文件

traced_script_module.save("model.pt")

**3、在C++中加载Script Module**

环境配置：pytorch 1.4 + libtorch 1.3 + VS 2015 + cmake 3.18

在libtorch文件夹内创建文件夹example，在example中创建build文件夹；

创建C++文件和CMAKELISTS.txt，并将.pt文件放在同一文件夹

在命令行输入CMD中的内容：

cmake –DCMAKE_PREFIX_PATH=(路径) -DCMAKE_BUILD_TYPE=Debug() –G “Visual Studio 14 Win64” ..

将libtorch中的lib文件夹中的dll复制进debug文件夹中

模型加载成功

**4、编写C++部署代码**

① 编写**输出**部分：

选取推理概率最高的输出，并使用Map结构对照输出手势结果。

② 编写**输入**部分：

**a、** 首先是离线模式，需要编写CSV文件读取以及<x,y,p,t>的选取，形成数据流，将其转换为Tensor并输入至加载好的模型内；

结果：数据成功输入并获得正确结果；

**b、** 接着是在线模式，接入DVS_SDK的getVectorData数据流。

初始SDK是在类中的重载虚函数编写的，单独线程，初步思路是定义全局变量在重载函数内存满后放到主线程中，发现数据会存在漏读和内存读取异常；

解决办法：改写SDK，定义一个互斥锁，管理线程读写共享内存的顺序。设置主线程永循环，在循环内读取数据；若时间<win，压入全局变量内；若时间>win，形成数据流并推断输出；

**c、**  已解决问题是：

推理时间过长，实时性较差，可能是因Debug模式且Libtorch API处理线程问题混乱。

首先尝试设置并行计算的线程数，默认全部打开

* 优化：torch::NoGradGuard no_grad;  推理时间45s左右

* 原因：它会把一部分时间用来计算梯度

* 优化：更改dt与step

  dt = 30，step = 70；默认 45s

  dt = 50，step = 40；训练效果中等，准确率停留在80%（放弃）

  dt = 40，step = 40；训练效果优秀，准确率停留在100%    推理时间22.5s左右

* 原因：for（step）相当于内部循环推理step次，时间线性增长

**5、优化DVS输出为脉冲** 

**输入静态图 & 建帧时长30ms**

Python 推理时间为0.043s左右

C++   推理时间为0.46s左右

**输入三帧图像 & 建帧时长30ms & 三帧图像输入三个channel中**

Python 推理时间为0.048s左右
