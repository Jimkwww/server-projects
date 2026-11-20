# Round5 frozen readout trend (per checkpoint)
Protocol: freeze → mean-pool → StandardScaler(train) + L2 LR(C=1). RML main = val; radar main = eval(test). Mean ± sample-std over 3 seeds.

## F1

### rml (val)
| backbone | step7800 | epoch20 | epoch30 | epoch40 | epoch50 |
|---|---|---|---|---|---|
| G | 0.3282 | 0.3724 | 0.3908 | 0.4029 | 0.4015 |
| G_Qlearn | 0.3172 | 0.3595 | 0.3719 | 0.3796 | 0.3798 |
| Q | 0.3442 | 0.3668 | 0.3810 | 0.3859 | 0.3875 |

### radar (test)
| backbone | step7800 | epoch20 | epoch30 | epoch40 | epoch50 |
|---|---|---|---|---|---|
| G | 0.7766 | 0.7844 | 0.7792 | 0.7820 | 0.7825 |
| G_Qlearn | 0.8097 | 0.8105 | 0.8095 | 0.8081 | 0.8080 |
| Q | 0.8108 | 0.8129 | 0.8153 | 0.8153 | 0.8148 |

## F2

### rml (val)
| backbone | step7800 | epoch20 | epoch30 | epoch40 | epoch50 |
|---|---|---|---|---|---|
| G | 0.3276 | 0.3469 | 0.3671 | 0.3787 | 0.3774 |
| G_Qlearn | 0.3246 | 0.3519 | 0.3658 | 0.3665 | 0.3675 |
| Q | 0.3384 | 0.3547 | 0.3594 | 0.3631 | 0.3620 |

### radar (test)
| backbone | step7800 | epoch20 | epoch30 | epoch40 | epoch50 |
|---|---|---|---|---|---|
| G | 0.7802 | 0.7845 | 0.7890 | 0.7874 | 0.7874 |
| G_Qlearn | 0.7866 | 0.7960 | 0.7920 | 0.7916 | 0.7905 |
| Q | 0.8091 | 0.8058 | 0.8040 | 0.8041 | 0.8040 |
