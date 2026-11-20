# E. 雷达数据盘点（独立域迁移数据源评估）

盘点对象：`/data1/user_wzk/datasets/` 下的三个雷达候选数据集。目标：选一个"纯雷达、复 IQ、可单标签分类"的数据集做独立域迁移。

| 数据集 | 任务性质 | 样本格式 | 长度 | 类别 | 复 IQ? | 官方划分 | 结论 |
|---|---|---|---|---|---|---|---|
| RadChar(-Tiny) | 脉内调制分类 | complex128 IQ | 512 @3.2MHz | 5 类（相参脉冲串/Barker/多相 Barker/Frank/LFM） | ✅ | 无（已自建分层划分） | ✅ 采用（主数据） |
| RadarCommDataset | 雷达+通信混合 | float32 **实值**单通道 | 256 | 9 类（amdsb/amssb/ask/bpsk/dsss_cck/dsss_oqpsk/fmcw/gfsk/pulsed） | ❌ 实值非复 IQ | 无 | ❌ 格式不符（实值）+ 通信雷达混合 |
| EM-Radar-mixed | 盲源分离(BSS) | float32 IQ | 1024 | 源类型（iq_mix/iq_src1/iq_src2） | ✅ | 无 | ❌ 是源分离任务，非单标签分类 |

## RadChar 详情（选用）

- 文件：`RadChar-Tiny.h5`，`iq`(50000,512) complex128，`labels`(50000,) 结构体 `[index, signal_type, number_of_pulses, pulse_width, time_delay, PRI, signal_to_noise_ratio]`。
- 5 类各 10000 条，**完全均衡**；SNR 整数 −20..20dB（41 档）。
- 每类内连续随机参数（脉宽/脉数/PRI/时延），512 采样即一条完整波形 → **一条波形 = 一条记录**，无需滑窗，天然记录级。
- 无跨样本重复（50000 个 IQ 哈希全部唯一）。
- Tiny ⊂ Small ⊂ Baseline ⊂ Large（嵌套子集），只取 Tiny 单变体，避免嵌套泄漏。

## 预处理产物

- `prep_radchar.py` 输出：`/data1/user_wzk/datasets/Radchar/radchar_tiny_rml.h5`（X[N,512,2]/Y[N,5] one-hot/Z[N,1] SNR，NLC 布局）。
- 分层划分（按 signal_type，seed=0，60/20/20）→ train/dev/eval = 30000/10000/10000，索引存 `round2_overnight/data/radchar_{train,dev,eval}.npy`。
- manifest `round2_overnight/configs/radchar.json`；probe 配置 `round2_overnight/configs/radar_probe.json`（5 类、L=512）。

## 迁移协议（E）

- 复用 round-1 冻结编码器（G/Q × {random, MAE, denoise} × 3 seed），不向编码器输入 SNR/类别/源信息（SNR 仅用于低/高信噪比分档报告）。
- 全标签 probe + 每类 10 条的 5 次抽样 probe；早停/超参只看 radar dev，锁定后雷达 eval 评一次。
