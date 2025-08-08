# Torch Weights 权重加载模块分析

## 文件位置
`/Users/georgezhou/Downloads/gpt-oss/gpt_oss/torch/weights.py`

## 概述
这是一个专门处理 MXFP4 (Mixed-Precision 4-bit Floating Point) 量化权重的加载模块。它负责从 SafeTensors 格式的检查点文件中加载模型权重，并支持高效的 MXFP4 量化格式解码。

## 核心常量定义

### MXFP4 相关常量
**位置**: 第 8-14 行
```python
BYTES_PER_BLOCK = 16  # 32个FP4数字打包在16字节中
FP4_VALUES = [
    +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,  # 正值
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # 负值
]
```
- **BYTES_PER_BLOCK**: 每个 MXFP4 块包含 32 个 FP4 数字，压缩到 16 字节
- **FP4_VALUES**: FP4 格式支持的 16 个浮点值的查找表

### 参数名映射
**位置**: 第 16-25 行
```python
PARAM_NAME_MAP = {
    f"block.{n}.mlp.mlp1_bias": f"block.{n}.mlp.mlp1_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp1_weight": (f"block.{n}.mlp.mlp1_weight.blocks", f"block.{n}.mlp.mlp1_weight.scales") for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_bias": f"block.{n}.mlp.mlp2_bias" for n in range(36)
} | {
    f"block.{n}.mlp.mlp2_weight": (f"block.{n}.mlp.mlp2_weight.blocks", f"block.{n}.mlp.mlp2_weight.scales") for n in range(36)
}
```
将逻辑参数名映射到检查点中的实际张量名，支持 36 个 transformer 块。

## 核心类: Checkpoint

### 构造函数 `__init__`
**位置**: 第 29-50 行
**功能**: 初始化检查点加载器

#### 实现过程:
1. **设备字符串构建** (第 30-35 行):
   ```python
   device_str = (
       device.type if device.index is None
       else device.type + ":" + str(device.index)
   )
   ```

2. **SafeTensors 文件扫描** (第 37-42 行):
   扫描检查点目录中所有 `.safetensors` 文件

3. **张量映射构建** (第 43-49 行):
   ```python
   tensor_name_to_file = {}
   for safetensor_file in safetensor_files:
       with safe_open(safetensor_file, framework="pt", device=device_str) as f:
           for key in f.keys():
               tensor_name_to_file[key] = safetensor_file
   ```

### 主要方法

#### `get(name: str) -> torch.Tensor`
**位置**: 第 52-59 行
**功能**: 根据参数名获取张量

实现逻辑:
1. 使用参数名映射查找实际张量名
2. 如果返回元组 (blocks_name, scales_name)，调用 `_get_mxfp4_tensor`
3. 如果返回单个名称，调用 `_get_tensor`

#### `_get_tensor(name: str) -> torch.Tensor`
**位置**: 第 61-66 行
**功能**: 加载普通张量（偏置等）

#### `_get_mxfp4_tensor(...) -> torch.Tensor`
**位置**: 第 68-117 行
**功能**: 解码 MXFP4 量化权重

##### 核心解码算法:
1. **数据加载** (第 83-84 行):
   ```python
   blocks = self._get_tensor(blocks_name)
   scales = self._get_tensor(scales_name).to(torch.int32) - 127
   ```

2. **查找表创建** (第 90 行):
   ```python
   lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
   ```

3. **分块处理** (第 100-116 行):
   ```python
   for r0 in range(0, rows_total, rows_per_chunk):
       # 提取低4位和高4位
       idx_lo = (blk & 0x0F).to(torch.long)
       idx_hi = (blk >> 4).to(torch.long)
       
       # 查找表映射
       sub[:, 0::2] = lut[idx_lo]
       sub[:, 1::2] = lut[idx_hi]
       
       # 应用缩放因子
       torch.ldexp(sub, exp, out=sub)
   ```

## MXFP4 量化原理

### 数据格式
1. **Blocks**: 每字节包含两个 4 位索引（低4位和高4位）
2. **Scales**: 每组对应的指数缩放因子
3. **解码**: `value = FP4_VALUES[index] * 2^(scale-127)`

### 内存效率
- **压缩比**: 16:1 (相对于 FP16)
- **分块处理**: 避免大张量一次性加载到内存
- **交错存储**: 支持 SwiGLU 激活的高效计算

## 优化版本对比

### `_get_mxfp4_tensor` (内存优化版)
**位置**: 第 68-117 行
- 分块处理，内存使用少
- 适合大型模型推理

### `_get_mxfp4_tensor_copy` (简化版)
**位置**: 第 119-137 行
- 一次性处理所有数据
- 内存使用多但代码简洁

## 与其他模块的关系

### 依赖模块
- `torch`: PyTorch 框架
- `safetensors`: 安全张量格式
- `math`: 数学运算

### 被调用者
- `gpt_oss/torch/model.py`: Torch 模型实现
- `gpt_oss/triton/model.py`: Triton 模型实现

## 性能特征

### 内存优化
- 分块加载避免 OOM
- 支持不同设备 (CPU/GPU)
- 内存映射文件读取

### 计算优化
- 查找表快速解码
- 向量化操作
- 原地计算减少内存分配

## 使用示例

```python
import torch
from gpt_oss.torch.weights import Checkpoint

# 初始化检查点加载器
device = torch.device("cuda:0")
checkpoint = Checkpoint("path/to/checkpoint/", device)

# 加载普通权重 (bias)
bias = checkpoint.get("block.0.mlp.mlp1_bias")

# 加载量化权重 (自动解码 MXFP4)
weight = checkpoint.get("block.0.mlp.mlp1_weight")
```

## 技术亮点

1. **高效量化**: MXFP4 格式在保持精度的同时大幅减少内存使用
2. **灵活映射**: 支持不同的参数命名约定
3. **分块处理**: 处理超大权重时的内存安全
4. **设备感知**: 自动适配不同的 PyTorch 设备