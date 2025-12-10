---
title: Final Highlight Test
date: 2025-11-30 23:00:00
tags: [Test]
---

测试高亮功能。请确保下方有三个反引号，并且紧跟着 python 单词。

```python
import torch
import numpy as np

# 这是一个定义函数的例子
def calculate_loss(prediction, target):
    """
    计算均方误差
    """
    error = prediction - target
    return torch.mean(error ** 2)

if __name__ == "__main__":
    # 初始化变量
    x = torch.tensor([1.0, 2.0, 3.0])
    y = torch.tensor([1.1, 1.9, 3.2])
    
    # 调用函数
    loss = calculate_loss(x, y)
    print(f"Current Loss: {loss:.4f}")
```

测试结束。