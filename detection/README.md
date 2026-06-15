/home/weiqiang/AI/ocr-majorbio/
|-- data/                  # 存放你的数据集
|   |-- train_images/      # 训练图片
|   |-- train_labels/      # 训练标签 (例如, .txt 文件)
|   |-- val_images/        # 验证图片
|   |-- val_labels/        # 验证标签
|-- models/                # 存放模型定义脚本
|   |-- east_model.py      # 模型架构
|-- utils/                 # 工具脚本
|   |-- dataset.py         # 自定义Dataset类，用于加载数据
|   |-- transforms.py      # 数据预处理和增强
|-- train.py               # 训练脚本
|-- eval.py                # 评估脚本
|-- requirements.txt       # 项目依赖
