# /projects/SSM 项目规则

先读 SERVER_AGENT_PROMPT.md、docs/EXPERIMENT_PROTOCOL.md。
本目录是新实验项目，不得覆盖其他/projects项目。数据来自既有cls_dev，禁止重新随机划分来替代。所有真实训练前审计历史Global与预处理；候选默认配置不是旧实验事实。
保持G/S0/S1/S2主对照；物理融合S2_concat为辅助。自定义SSM不得命名为标准Mamba或忠实S5。未验证前不能声称频偏等变、频偏补偿或效率提升。
正式实验前跑tests、合成pipeline、CUDA检查与真实smoke。服务器负责具体验证与修复。保存配置、源码hash、索引hash、所有结果和失败日志。不得伪造运行、数据或结果。
修复后重跑相关验证，不在同一个run中途改变源码与协议。新配置或新代码使用新campaign。test不用于调参。任务范围内自主推进，遇到不可获取的关键数据/协议再明确报告阻塞。
