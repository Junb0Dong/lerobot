# `my_dataset_merged` 数据集检查（2026-08-30）

## 当前状态

已检查用户提供的本地 LeRobot v3.0 数据集。数据集包含 53 个 episode、43,326 个数值帧、30 FPS、1 个 task、14D action 和 14D observation state。共有 3 路 640×480 RGB 视频：`head`、`left_wrist`、`right_wrist`，编码元数据声明为 AV1/yuv420p、无音频、pyav backend。

## 关键发现

- 数值数据完整性基本正常：`index` 连续覆盖 0–43,325、无重复；`episode_index` 覆盖 0–52；action/state 无 NaN 或 Inf。
- episode 长度为 661–1,069 帧，平均 817.47 帧，约 22.0–35.6 秒，合计约 24.1 分钟。
- action/state 均为左臂 6D + 右臂 6D + 头部 2D 的位置特征。两个头部维度在全数据中恒定，物理单位未在元数据声明。
- `observation.state` 作为当前测得位置、`action` 作为目标/控制位置而使用相同 14D 空间是正常的行为克隆 contract；二者并非重复标签。XLeRobot 源配置默认 `use_degrees=False` 时，臂/头使用校准后的 `[-100, 100]` 归一化位置，只有显式 `use_degrees=True` 才是角度制；仅凭数据文件无法确定录制时采用了哪种模式。
- 源码语义更具体：`get_observation()` 从 Feetech 的 `Present_Position` 读取 state；`send_action()` 将 `.pos` 字段作为绝对目标，写入 `Goal_Position`，不是增量动作。默认 `max_relative_target=None` 时不做相对目标安全裁剪。记录器实际保存的是发送调用前的 `action_values`，若配置了 action processor 或安全裁剪，文件中的 action 不一定等于最终写入寄存器的值。
- 实际数据中 action 的前 12 个关节维度有 2,813 个值超出 `[-100, 100]`，而 state 没有超出；因此不能把 action 直接当作已被电机接受的原始目标或保证在合法归一化范围内。若采用 `RANGE_M100_100`，底层 Feetech 写入时会将其裁剪到 `[-100,100]` 后再按 calibration 映射到原始编码器值。
- `data/chunk-000` 下的三个 parquet 是存储分片，不代表三个动作块：`file-000` 为 episode 0–15（14,100 帧），`file-001` 为 16–26（9,034 帧），`file-002` 为 27–52（20,192 帧）。当前文件都远小于 `data_files_size_in_mb=100`，更像是合并/写出过程保留了源文件边界或关闭了数据拼接；三者总计仍是完整 43,326 帧。
- 53×3 个视频文件均存在；通过 MP4 容器的 `stsz/stts` 检查发现 18/53 个 episode 的视频帧数多于对应数值帧数，总视频帧 50,844，而数值帧 43,326。训练前应确认视频尾部是否为无效录制。
- `stats.json` 的 action/state 统计与实际数据一致，但 `episode_index`/`index` 的全局统计仍是旧范围（最大值 25/20,191，而实际为 52/43,325）。这两列不用于归一化，但元数据需要留意。

## 验证方式与后续

- 使用 `pyarrow` 读取三个 data parquet、episode parquet 和 `tasks.parquet`，直接重算帧数、边界、分组和数值统计。
- 使用 Python 标准库解析 MP4 容器中的视频轨道时长、sample 数和时间表；未做完整像素解码验证。
- 后续若要训练，优先用实际 LeRobot reader 读取 `dataset[0]`，再随机抽取首/中/尾帧核对 action、state 与视频同步；确认无误后再清理多余视频尾帧或重建 episode 视频元数据。

## 2026-08-30 派生 ActionCodec 数据集

- 已通过只读审计：53 个 episode、43,326 个数值帧、14D action/state、30 FPS；18 个视频异常 episode 在有效数值 timestamp 范围内三路均可解码，帧 timestamp 误差为 0，因此保留视频文件原样，仅忽略数值范围外的尾部视频帧。
- 已生成 `/home/ainot02/xzd/lerobot_v060/local_datasets/my_dataset_merged_actioncodec`。action 裁剪了 2,813/606,564 个元素（0.004637598011091987），state 保持逐元素不变；`meta/stats.json`、episode action/state stats、`index`/`episode_index` 全局统计已重算。
- 脚本：`src/lerobot/scripts/prepare_actioncodec_dataset.py`；测试：`tests/actioncodec/test_prepare_actioncodec_dataset.py`。派生目录中的 159 个视频与原目录逐文件字节一致，LeRobotDataset 首/中/尾样本三路图像读取通过。
