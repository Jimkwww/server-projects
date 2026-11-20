# SSM_L round-1 acceptance report

- device: `cuda`
- all_pass: **True**

- [x] **1. patch/decode round-trip + old-project alignment** — roundtrip=True impulse_order=True init_bit_identical={'G': True, 'Q': True}
- [x] **2. noise statistics** — mean<0.02=True var_eq=True uncorr=True power_ratio_±0.3dB=True zero_input=True tiny=True reproducible=True ratios=0.0dB:0.07,5.0dB:-4.93,10.0dB:-9.93,15.0dB:-14.93
- [x] **3. mask leakage + Q-denoise recompute + Q filter gradients + probe freeze** — mask_leakage_free=True denoise_recompute=True Q_32_scalars=True grad_flow_pretrain=True grad_flow_downstream=True probe_frozen=True
- [x] **4. no-NaN loss decrease + shared init + sample order + checkpoint reload** — loss_decreases={'G-mae': True, 'G-denoise': True, 'Q-mae': True, 'Q-denoise': True} shared_init=True sample_order=True reload_consistent=True
