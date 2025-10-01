# Ceres Ball Trajectory Demo

该示例程序使用 OpenCV 对 `resources/video.mp4` 中的小球进行逐帧检测，并借助 Ceres Solver 拟合以下带阻力的弹道模型：

$$
\begin{cases}
\Delta t = t - t_0\\
x(t) = x_0 + \dfrac{v_{x0}}{k}\left(1 - e^{-k \Delta t}\right)\\
y(t) = y_0 + \dfrac{v_{y0} + \frac{g}{k}}{k}\left(1 - e^{-k \Delta t}\right) - \dfrac{g}{k}\Delta t
\end{cases}
$$

求解得到的参数包括 $x_0, y_0, v_{x0}, v_{y0}, g, k$（其中 $g \in [100,1000](px/s^2)$，$k \in [0.01, 1](1/s)$）。程序会输出拟合曲线图像以及叠加拟合结果的视频到 `results` 目录。

## 运行

```bash
./build/ceres_project
```

运行后会在 `results/` 目录生成以下文件：

- `trajectory_plot.png`：拟合后的小球轨迹二维图（绿色为检测点，红色为拟合曲线）。
- `video_with_fit.mp4`：在原视频上叠加检测点与拟合结果的输出视频。

程序会在终端打印拟合得到的参数以及带阻力模型方程，其中 $y_{up}$ 以屏幕向上为正；若需转换为图像像素坐标，可使用 $y_{image} = H - y_{up}$（$H$ 为帧高）。

若检测精度不足，可以调整 `src/main.cpp` 中 `detectBallCenter` 函数的参数（例如霍夫圆检测或自适应阈值的参数），以适配不同光照或小球颜色。

## 结果
- 当采用霍夫圆形检测来确定圆心，结果为：

$$ \Delta t = t - t_0 $$

$$ x(t) = 163.718259 + \frac{253.039512}{0.066628} \left(1 - e^{-0.066628 \Delta t}\right) $$

$$ y(t) = 591.678024 + \frac{349.187519 + \frac{499.435454}{0.066628}}{0.066628} \left(1 - e^{-0.066628 \Delta t}\right) - \frac{499.435454}{0.066628} \Delta t $$

- 平均残差值为 0.937
- x0=163.718259, y0=591.678024, vx0=253.039512, vy0=349.187519, g=499.435454, k=0.066628
## 注意
需要注意的是，由于采取的坐标架不同，x0,y0会有不同。例如，以在图片坐标系下，左上角为原点，第一帧图像中小球位置x0,y0分别为163,128但为了画图的方便，为定义的坐标架为以左下角为原点，此时x0,y0分别为164,591.与标准答案有冲突属于正常现象，清学长仔细考察！

## 性能 (Performance)
下述性能数据基于当前最小化核心实现（仅检测 + 参数拟合，无绘图/视频输出），在一次运行中的测量结果：

```
timing_ms detect=2177 solve=0 total=2182 frames=210 avg_detect_per_frame_ms=10.367
```

说明：
- 检测阶段耗时约 2177 ms，占总时间绝大部分；Ceres 求解在该数据规模下（210 帧、420 个残差）耗时≈0 ms（相对检测可忽略）。
- 平均每帧检测耗时 ~10.37 ms；对应理论最大处理帧率 ≈96 FPS（单线程检测 + 现有参数）——已满足 60 FPS 视频的离线处理需求。
- HoughCircles 命中时往往一次成功，失败时回退到自适应阈值 + 连通域质心法，这两个分支构成主要 CPU 时间。

已采用的代码级优化：
1. 预分配：对时间戳与坐标 vector 使用 reserve 减少 reallocation。
2. 参数初始化：利用首末帧估计初速度，减少求解器早期抖动迭代（间接节省部分求解时间）。
3. 线程配置：Ceres 使用 `hardware_concurrency()` 配置多线程（虽然本例残差较少，影响有限）。
4. 关闭不必要输出：最小化 I/O 干扰真实检测与求解耗时。
5. Release 编译：`-O3 -march=native -DNDEBUG -flto`（若开启 profiling 模式会切换为 `-O2 -g -fno-omit-frame-pointer` 以利 perf 分析）。
