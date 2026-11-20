# DMC vision 在线重建图集（2026-11-20）

已按用户限定接入 **DMC vision**：使用在线物理 decoder 和 Dreamer 原生 image decoder 时，每100,000个控制台 Step 保存一次并排图集。Proprio 不启用，不增加 observation→state 展示 decoder，也不自动开启原本关闭的物理 decoder。

## 与旧图集的对应关系

已阅读 `logdir/dyn_recon/physics_probe_stage5_three_groups_20261106/comparison_images/README.md`、相关 manifest、`physics_probe/compare_reconstruction_images.py`、物理渲染脚本及文档。旧流程先生成两路来源图，核对 feature 和 episode/起点一致，再合并原始像素。旧 proprio 右侧额外经过 observation→state decoder；用户本次明确仅对 vision 启用，因此无需这条辅助链路。

新功能在训练里一次生成共享想象 feature，然后同时送入：

- 左：float64 在线物理 decoder → qpos/qvel/act → 仿真副本静态截图；
- 右：Dreamer 原生 image decoder → RGB，最近邻放大。

默认列深度8、10、12、14。按用户后续要求，已改为每期严格进行五次想象重建实验、生成一张五行图片；优先使用不同普通、非末帧环境槽，起点不足时循环使用，但每行独立采样想象动作/潜在状态，不复制已有图片。同属于该训练模型的 seed，不是五个独立训练模型。不依赖 Advantage、候选池或 guard 接受情况；失败格明确标记。两路均不是未来真实画面，也不构成固定测试集的精度评估。

## 入口与产物

- `configs/configs.yaml:40`：周期、来源数、深度、尺寸和 seed，默认随在线 physics decoder 开启。
- `trainer.py`：采集器当前 posterior latent 产生后、模型更新前检查里程碑；只处理达到/越过的100k倍数，不在Step0生成、不重复同一里程碑。
- `Dreamer.reconstruction_visualization()`：隔离 CPU/当前 CUDA RNG，无额外 `obs_step`；只推理，不更新参数、梯度、normalizer 或 replay。
- `OnlinePhysicsEnv.render_reconstructions()`：独立 MuJoCo model/data 副本，设置预测 core、解析推进时间/上一 control；不执行动作，不调用分支 guard。核对真实环境 integration state/model 以及截图前后的副本状态保持不变。
- `reconstruction_visualization.py`：并排 PNG、README、HTML 索引、manifest 和 `samples.pt`。原生64×64像素最近邻放大到320×320；物理320×320像素直接排列。

各训练目录的 `reconstruction_visualization/README.md` 或 `index.html` 是总入口，每期存入 `step_000100000/` 等子目录。manifest 包含真实 Step、模型更新数、来源、相机、失败原因及共享 feature 哈希；tensor 文件保存 feature、动作链、float64 core 和两路像素。

默认 `dyn_rec=on`、`decoder_only` 的 DMC vision 会启用；`dyn_rec=off` 没有物理 decoder，默认不生成。无原生 image decoder 的 representation 跳过。图集失败单独记 `error.json` 并打印原因，训练继续；同里程碑不反复重试，不覆盖旧图集。修改供新启动进程使用，本次没有重启或改变已有训练。

## 验证

- 初版全仓库：**193项 CPU unittest 通过，79.351秒**，见 `unittest.log`。后续五行调整单独重跑全部10项可视化回归，5.104秒通过；未重复无关的全仓库测试。
- 新增9项回归：vision限定、物理head依赖、共享feature、Torch RNG/参数/源latent不变、100k/200k触发及不重复、终止/重建槽排除、无普通来源延后、失败原因、原始像素5倍最近邻放大、训练循环接线、RPC槽选择、真实MuJoCo状态/计时/task RNG/guard不变。
- GPU6 EGL：实际生成4对320×320物理截图/原生输出，零动作；已目视检查完整PNG。验收使用未训练CPU小模型，Step2仅模拟调度，**不是训练重建质量结果**。一次图集生成0.674秒，不据此推算正式模型训练开销。
- Black 120/unstable、Flake8、isort 和 `git diff --check` 通过；dreamer整文件的既有格式只检查新增方法。

EGL产物与明确信息：`egl_smoke/smoke.json`、`egl_smoke/reconstruction_visualization/step_000000002/comparison.png`。

五行修订另做实际GPU6 EGL验收：单个可用环境起点也生成五次实验、每行四对图，共20对，合成一张2624×2160 PNG，目视确认五行以及不同采样结果。零训练更新、零仿真动作，生成耗时2.576秒；仍是未训练CPU小模型的功能验收。新产物：`egl_five_trials/smoke.json`、`egl_five_trials/reconstruction_visualization/step_000000002/comparison.png`。

CPU验证固定CPU88、nice19，OMP/MKL/OpenBLAS各一线程。最初沙箱尝试无法写指定临时目录，Python回退到`/tmp`；测试临时目录随上下文清理。正式39项回归及193项全量回归在权限允许的环境重跑，并显式断言 `tempfile.gettempdir() == /mnt/phx2/tmp`，禁止回退。EGL验收也使用 `/mnt/phx2/tmp`，模型只在CPU上推理，没有启动训练或选择实验checkpoint。

仅用于图集索引的长HTML字面量在全量回归后做了等价拆分以通过行长检查；最后重新通过全部相关静态检查，没有改变图集或训练行为。

使用说明见 `docs/online_reconstruction.md` 的“DMC vision 训练中的重建可视化”。
