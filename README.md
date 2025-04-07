# 🌊 水质智能检测系统 - 基于深度学习的图像分类

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![TensorFlow 2.10](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)](https://www.tensorflow.org/)
[![OpenCV 4.6](https://img.shields.io/badge/OpenCV-4.6-green.svg)](https://opencv.org/)

一个基于CNN的水质分类系统，能够自动识别5种不同水质类别，准确率达85%+。包含完整的数据处理、模型训练、可视化分析和预测模块。

![预测示例](https://via.placeholder.com/600x400/009688/FFFFFF?text=Actual:4+Predicted:4) <!-- 建议替换为实际预测对比图 -->

## 🚀 项目亮点
- ​**工业级解决方案**：完整实现从数据处理到部署预测的完整AI pipeline
- ​**高性能模型**：采用深度卷积神经网络，测试集准确率85%+
- ​**可视化分析**：提供训练过程可视化、预测结果可视化模块
- ​**工程化实践**：包含模型持久化、路径优化等生产环境适配功能

## 📚 目录结构
```bash
Water-Quality-Detection/
├── data/                   # 原始水质图片数据集
├── saved_model/            # 训练好的模型文件
├── utils/                  # 工具函数
├── training_visualization/ # 训练过程可视化图表
├── config.py               # 路径配置
├── model.py                # 模型架构定义
├── train.py                # 模型训练脚本
└── predict.py              # 预测演示脚本
