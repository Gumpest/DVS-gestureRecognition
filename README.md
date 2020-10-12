# DVS-gestureRecognition-ANN2SNN
使用DVS_MP获取脉冲信号，输入SNN中，获得手势。

通过ANN的方法，训练dvs手势集，将网络通过ANN2SNN工具链，得到SNN的配置帧，完成部署。
## 数据集
使用CeleX_dvs_mp在白天拍摄，光照环境单一；
## 网络
使用三层ANN网络
## 项目流程
将三帧图片依次放在三个通道中，赋予其时空信息；

建帧时长30ms

部署在CPP+Libtorch平台上
