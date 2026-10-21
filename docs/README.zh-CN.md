> Detailed development and operations reference. Start with the [project overview](../README.md).

# CUDA / MUSA 蒙特卡洛求二维 Laplace 场

Sutton “肥皂泡”例子的连续版本对应 Dirichlet 问题：

```text
∇²u = 0            （区域内部）
u = g               （边界）
u(x) = E[g(X_τ)]    （随机游走首次碰到边界时的边界值期望）
```

程序让一个 GPU block 负责一个网格点，block 内各线程独立生成随机游走并归约平均值。随后计算 `E=-∇u`，输出 `field.csv`，列为 `x,y,u,Ex,Ey`。数值 kernel 共用同一份源码，`gpu_backend.h` 负责 CUDA/MUSA API 映射。

## 编译运行

没有 GPU 工具链时，可以用相同算法和输出格式的 CPU 参考程序验证案例：

```bash
make cpu
./mc_laplace_cpu 32 32 4096
```

需要 CUDA Toolkit 和支持 CUDA 的 NVIDIA GPU：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/mc_laplace 128 128 4096
```

也可以直接构建 CUDA 目标：

```bash
make cuda
./mc_laplace_cuda 128 128 4096
```

### MUSA

需要 MUSA SDK（含 `mcc`、MUSA Runtime 和 muRAND）。默认 SDK 路径为 `/usr/local/musa`：

```bash
make musa
./mc_laplace_musa 128 128 4096
```

若 SDK 在其他位置：

```bash
make musa MUSA_HOME=/opt/musa
```

可先用 `musaInfo` 检查驱动与设备，再确认以下文件存在：

```bash
test -x "$MUSA_HOME/bin/mcc"
test -f "$MUSA_HOME/include/musa_runtime.h"
test -f "$MUSA_HOME/include/murand_kernel.h"
```

MUSA 构建会定义 `USE_MUSA`，调用 `musaMalloc/musaMemcpy` 和设备侧 `murand_init/murand`，并链接 `-lmurand -lmusart`。CUDA 构建仍使用原来的 CUDA Runtime 与 cuRAND。

参数依次是 `nx ny 每点轨迹数 最大步数`；第四项可省略。

## 解析解验证案例与可视化

默认案例的区域是单位正方形 `[0,1]²`，四条边采用 Dirichlet 条件：

```text
u(x,y) = x²-y²
```

因为 `∇²u=2-2=0`，内部解析解仍是 `u=x²-y²`，解析场强为 `E=(-2x,2y)`。程序会打印势场 RMSE 和最大绝对误差。`field.csv` 包含模拟值、解析值、势场误差及场强误差。

安装 NumPy 和 Matplotlib 后生成可视化：

```bash
python3 -m pip install numpy matplotlib
python3 visualize.py field.csv
```

结果保存在 `field_validation.png`，包含模拟势场、解析势场、带符号误差、绝对误差以及两个场强分量。需要同时打开窗口时增加 `--show`。

若系统安装了 R，也可以不安装 Python 绘图库直接运行：

```bash
Rscript visualize.R field.csv
```

`make plot` 会优先使用 R，否则调用 Python 脚本。

## 换成任意边界

在 `main` 中设置两个同尺寸数组：`boundary[p]=1` 表示吸收边界，`values[p]` 是该处 Dirichlet 值。外框必须全部是边界；内部也可以放置任意形状的边界。只计算单个/少量查询点时，应把 kernel 改为只启动这些点，而不是计算整张网格。

误差按 `O(1/sqrt(N))` 衰减；轨迹数增加 4 倍，统计误差约减半。复杂区域或高分辨率下，逐格随机游走可能很慢，连续域的 Walk-on-Spheres 通常更合适。
# ElectronicSimulator
