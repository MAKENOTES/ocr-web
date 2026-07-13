1、图片样本红色背景白色字体，需优化。

## UI 白字/小文字预处理契约

当前采用“统一前处理 crop -> 再训练 -> 推理同样 crop”的方案。

训练集、验证集、推理阶段都使用 `recognition/utils/white_text_crop.py` 中的确定性白字裁剪逻辑，主要处理“有色背景 + 白色文字”的菜单、按钮和小文字截图。

`recognition/components/dataset.py` 通过 `use_auto_crop=True` 接入同一套 crop。训练阶段不再使用随机 UI padding 增强，避免训练分布和推理分布继续不一致。

推理阶段 `recognition/inference.py` 使用 `WhiteTextCropInferenceDataset`，并将裁剪后的核查图保存到 `recognition/data/crops`。
