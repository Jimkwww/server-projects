# Round6 逐种子配对差值
Δ = 前者 − 后者，逐种子（17/42/2026）；mean ± sample-std。

| 对比 | 指标 | seed17 | seed42 | seed2026 | mean ± std |
|---|---|---|---|---|---|---|
| G-denoise − G-MAE | comm_frozen | -0.0131 | -0.0106 | -0.0083 | -0.0107 ± 0.0024 |
| Q-denoise − Q-MAE | comm_frozen | -0.0346 | -0.0282 | -0.0330 | -0.0319 ± 0.0033 |
| G-joint − G-MAE | comm_frozen | -0.0087 | -0.0141 | 0.0016 | -0.0071 ± 0.0080 |
| G-joint − G-denoise | comm_frozen | 0.0045 | -0.0035 | 0.0099 | 0.0036 ± 0.0068 |
| Q-joint − Q-MAE | comm_frozen | -0.0260 | -0.0202 | -0.0221 | -0.0228 ± 0.0029 |
| Q-joint − Q-denoise | comm_frozen | 0.0087 | 0.0080 | 0.0109 | 0.0092 ± 0.0015 |
| Q-bi-MAE − Q-MAE | comm_frozen | -0.0311 | -0.0228 | -0.0372 | -0.0303 ± 0.0072 |
| Q-bi-denoise − Q-denoise | comm_frozen | -0.0003 | 0.0029 | -0.0115 | -0.0030 ± 0.0076 |
| Q-bi-joint − Q-joint | comm_frozen | -0.0038 | -0.0234 | -0.0115 | -0.0129 ± 0.0098 |
| G-denoise − G-MAE | radar_frozen | 0.0428 | 0.0251 | 0.0000 | 0.0226 ± 0.0215 |
| Q-denoise − Q-MAE | radar_frozen | 0.0109 | 0.0008 | 0.0070 | 0.0062 ± 0.0051 |
| G-joint − G-MAE | radar_frozen | 0.0368 | 0.0098 | -0.0133 | 0.0111 ± 0.0251 |
| G-joint − G-denoise | radar_frozen | -0.0060 | -0.0153 | -0.0133 | -0.0115 ± 0.0049 |
| Q-joint − Q-MAE | radar_frozen | 0.0116 | -0.0071 | -0.0004 | 0.0014 ± 0.0095 |
| Q-joint − Q-denoise | radar_frozen | 0.0007 | -0.0079 | -0.0074 | -0.0049 ± 0.0048 |
| Q-bi-MAE − Q-MAE | radar_frozen | -0.0034 | -0.0251 | -0.0202 | -0.0162 ± 0.0114 |
| Q-bi-denoise − Q-denoise | radar_frozen | -0.0075 | -0.0091 | -0.0095 | -0.0087 ± 0.0011 |
| Q-bi-joint − Q-joint | radar_frozen | -0.0123 | -0.0120 | -0.0089 | -0.0111 ± 0.0019 |
| G-denoise − G-MAE | comm_ft | -0.0349 | -0.0471 | -0.0798 | -0.0540 ± 0.0232 |
| Q-denoise − Q-MAE | comm_ft | -0.0292 | -0.0240 | -0.0202 | -0.0245 ± 0.0045 |
| G-joint − G-MAE | comm_ft | -0.0253 | -0.0519 | -0.0170 | -0.0314 ± 0.0182 |
| G-joint − G-denoise | comm_ft | 0.0096 | -0.0048 | 0.0628 | 0.0225 ± 0.0356 |
| Q-joint − Q-MAE | comm_ft | 0.0090 | 0.0087 | -0.0276 | -0.0033 ± 0.0210 |
| Q-joint − Q-denoise | comm_ft | 0.0381 | 0.0327 | -0.0074 | 0.0212 ± 0.0249 |
| Q-bi-MAE − Q-MAE | comm_ft | 0.0144 | 0.0285 | 0.0083 | 0.0171 ± 0.0104 |
| Q-bi-denoise − Q-denoise | comm_ft | 0.0067 | 0.0397 | -0.0340 | 0.0042 ± 0.0369 |
| Q-bi-joint − Q-joint | comm_ft | -0.0202 | -0.0122 | 0.0173 | -0.0050 ± 0.0197 |
| G-denoise − G-MAE | radar_ft | -0.0033 | -0.0033 | -0.0034 | -0.0033 ± 0.0001 |
| Q-denoise − Q-MAE | radar_ft | -0.0034 | 0.0000 | 0.0118 | 0.0028 ± 0.0080 |
| G-joint − G-MAE | radar_ft | 0.0016 | -0.0007 | 0.0088 | 0.0032 ± 0.0050 |
| G-joint − G-denoise | radar_ft | 0.0049 | 0.0026 | 0.0122 | 0.0066 ± 0.0050 |
| Q-joint − Q-MAE | radar_ft | -0.0017 | -0.0045 | 0.0013 | -0.0016 ± 0.0029 |
| Q-joint − Q-denoise | radar_ft | 0.0017 | -0.0045 | -0.0105 | -0.0044 ± 0.0061 |
| Q-bi-MAE − Q-MAE | radar_ft | 0.0015 | 0.0078 | 0.0035 | 0.0043 ± 0.0032 |
| Q-bi-denoise − Q-denoise | radar_ft | 0.0036 | 0.0024 | -0.0074 | -0.0005 ± 0.0060 |
| Q-bi-joint − Q-joint | radar_ft | -0.0005 | 0.0055 | 0.0054 | 0.0035 ± 0.0034 |
