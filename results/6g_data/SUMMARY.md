# Augmentation benchmark — cross-scenario summary

_71 rows from 9 scenarios, 2 algorithms._


## Algorithm: `RandomForest`

| scenario            |     BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   Δ AutoFeat−BASE |
|:--------------------|---------:|---------------:|----------------------:|-----------:|------------------:|
| scenario1           |   0.9649 |         0.9461 |                0.9462 |     0.9649 |            0      |
| scenario2c          |   0.456  |         0.9378 |                0.951  |     0.903  |            0.447  |
| scenarioA_lat95     |   0.9457 |         0.9317 |                0.9391 |     0.9769 |            0.0312 |
| scenarioA_lat99     |   0.9408 |         0.9197 |                0.927  |     0.9734 |            0.0326 |
| scenarioB_amf_seg01 |   0.9424 |         0.9442 |                0.8469 |     0.9415 |           -0.0009 |
| scenarioK_csi       | nan      |         1      |                1      |     1      |          nan      |
| scenarioN_target_n  |   0.9739 |         0.9752 |                0.9413 |     0.9749 |            0.001  |
| scenarioR_resource  |   0.456  |         0.4527 |                0.4559 |     0.9714 |            0.5154 |
| scenarioU_unrelated |   0.9408 |         0.9408 |                0.9556 |     0.9734 |            0.0326 |

## Algorithm: `XGBoost`

| scenario            |   BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   Δ AutoFeat−BASE |
|:--------------------|-------:|---------------:|----------------------:|-----------:|------------------:|
| scenario1           | 0.9662 |         0.948  |                0.9492 |     0.9662 |            0      |
| scenario2c          | 0.5395 |         0.9455 |                0.9472 |     0.9417 |            0.4022 |
| scenarioA_lat95     | 0.9551 |         0.9607 |                0.9604 |     0.9796 |            0.0245 |
| scenarioA_lat99     | 0.9508 |         0.9534 |                0.9552 |     0.9765 |            0.0257 |
| scenarioB_amf_seg01 | 0.9412 |         0.9346 |                0.7941 |     0.9162 |           -0.025  |
| scenarioK_csi       | 0      |         0.9896 |                0.9948 |     1      |            1      |
| scenarioN_target_n  | 0.9786 |         0.9792 |                0.9212 |     0.9794 |            0.0008 |
| scenarioR_resource  | 0.5395 |         0.5389 |                0.5373 |     0.9724 |            0.4329 |
| scenarioU_unrelated | 0.9508 |         0.9508 |                0.9536 |     0.9765 |            0.0257 |


## Scenario context

| Scenario | Purpose | Expected | Target |
|---|---|---|---|
| `scenario1` | Cross-service workload augmentation | refuse | `lat99` |
| `scenario2c` | Feature recovery (self-join via time, proxy-free lake) | showcase | `lat99` |
| `scenarioA_lat95` | Cross-app temporal (target lat95, proxy-free lakes) | partial | `lat95` |
| `scenarioA_lat99` | Cross-app temporal (target lat99, proxy-free lakes) | partial | `lat99` |
| `scenarioB_amf_seg01` | Within-app amf segments (join on n, proxy-free lake) | partial | `lat99` |
| `scenarioK_csi` | KUL MaMIMO CSI compression (no identity columns in lake) | showcase | `target_x` |
| `scenarioN_target_n` | Inverse problem (target = workload n) | refuse | `n` |
| `scenarioR_resource` | Cross-app resource contention (proxy-free positive) | showcase | `lat99` |
| `scenarioU_unrelated` | Heterogeneous unrelated lake (honest negative) | refuse | `lat99` |
