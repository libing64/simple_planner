# Simple PushT Planner

这是一个简单的PushT仿真环境和规划器实现。

## 项目结构

- `pusht_env.py`: PushT仿真环境实现 (基于 Gymnasium, Pymunk, Pygame)
- `planner.py`: 简单的规划器实现 (将T型块推到目标位置)
- `demo.py`: 演示脚本
- `environment.yml`: Conda环境配置文件
- `requirements.txt`: Python依赖列表
- `create_env.sh`: 创建环境的脚本

## 安装指南

1. 创建Conda环境:
   ```bash
   bash create_env.sh
   ```
   或者手动创建:
   ```bash
   conda env create -f environment.yml
   ```

2. 激活环境:
   ```bash
   conda activate pusht_env
   ```

## 运行演示

运行演示脚本以查看规划器效果:
```bash
python demo.py
```

## 手动控制

手动控制Agent推动T型块:
```bash
python manual_control.py
```
- 使用方向键 (↑ ↓ ← →) 控制Agent移动
- 按 `R` 重置环境
- 按 `Q` 退出

## 实现细节

- **环境**: 
  - 观察空间: Agent位置, T型块位置, T型块角度
  - 动作空间: Agent速度 (vx, vy)
  - 奖励: 基于到目标的距离和角度差

- **规划器**: 
  - 使用简单的基于状态机的逻辑
  - 如果Agent在T型块前方，先绕行到后方
  - 计算推行方向和速度
