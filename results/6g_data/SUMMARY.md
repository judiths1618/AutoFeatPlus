# Augmentation benchmark — cross-scenario summary

_89 rows from 9 scenarios, 2 algorithms._


## Algorithm: `RandomForest`

| scenario            |     BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   AutoFeatPlus |   Δ AutoFeat−BASE |   Δ AutoFeatPlus−BASE |
|:--------------------|---------:|---------------:|----------------------:|-----------:|---------------:|------------------:|----------------------:|
| scenario1           |   0.9649 |         0.9461 |                0.9462 |     0.9649 |         0.9474 |            0      |               -0.0175 |
| scenario2c          |   0.456  |         0.9378 |                0.951  |     0.903  |         0.9371 |            0.447  |                0.4811 |
| scenarioA_lat95     |   0.9457 |         0.9317 |                0.9391 |     0.9769 |         0.8299 |            0.0312 |               -0.1158 |
| scenarioA_lat99     |   0.9408 |         0.9197 |                0.927  |     0.9734 |         0.8186 |            0.0326 |               -0.1222 |
| scenarioB_amf_seg01 |   0.9424 |         0.9442 |                0.8469 |     0.9415 |         0.8307 |           -0.0009 |               -0.1117 |
| scenarioK_csi       | nan      |         1      |                1      |     1      |         1      |          nan      |              nan      |
| scenarioN_target_n  |   0.9739 |         0.9752 |                0.9413 |     0.9749 |         0.9053 |            0.001  |               -0.0686 |
| scenarioR_resource  |   0.456  |         0.4527 |                0.4559 |     0.9714 |         0.4559 |            0.5154 |               -0.0001 |
| scenarioU_unrelated |   0.9408 |         0.9408 |                0.9556 |     0.9734 |         0.9417 |            0.0326 |                0.0009 |

## Algorithm: `XGBoost`

| scenario            |   BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   AutoFeatPlus |   Δ AutoFeat−BASE |   Δ AutoFeatPlus−BASE |
|:--------------------|-------:|---------------:|----------------------:|-----------:|---------------:|------------------:|----------------------:|
| scenario1           | 0.9662 |         0.948  |                0.9492 |     0.9662 |         0.9454 |            0      |               -0.0208 |
| scenario2c          | 0.5395 |         0.9455 |                0.9472 |     0.9417 |         0.9518 |            0.4022 |                0.4123 |
| scenarioA_lat95     | 0.9551 |         0.9607 |                0.9604 |     0.9796 |         0.9048 |            0.0245 |               -0.0503 |
| scenarioA_lat99     | 0.9508 |         0.9534 |                0.9552 |     0.9765 |         0.8845 |            0.0257 |               -0.0663 |
| scenarioB_amf_seg01 | 0.9412 |         0.9346 |                0.7941 |     0.9162 |         0.7742 |           -0.025  |               -0.167  |
| scenarioK_csi       | 0      |         0.9896 |                0.9948 |     1      |         0.9948 |            1      |                0.9948 |
| scenarioN_target_n  | 0.9786 |         0.9791 |                0.9212 |     0.9794 |         0.9158 |            0.0008 |               -0.0628 |
| scenarioR_resource  | 0.5395 |         0.5389 |                0.5373 |     0.9724 |         0.5373 |            0.4329 |               -0.0022 |
| scenarioU_unrelated | 0.9508 |         0.9508 |                0.9536 |     0.9765 |         0.9508 |            0.0257 |                0      |


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
