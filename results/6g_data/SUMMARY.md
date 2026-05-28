# Augmentation benchmark — cross-scenario summary

_62 rows from 8 scenarios, 2 algorithms._


## Algorithm: `RandomForest`

| scenario            |     BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   Δ AutoFeat−BASE |
|:--------------------|---------:|---------------:|----------------------:|-----------:|------------------:|
| scenario1           |   0.9649 |         0.9461 |                0.9462 |     0.9649 |            0      |
| scenario2c          |   0.456  |         0.9858 |                0.9865 |     0.9842 |            0.5282 |
| scenarioA_lat95     |   0.9457 |         0.9325 |                0.9302 |     0.9769 |            0.0312 |
| scenarioA_lat99     |   0.9408 |         0.9278 |                0.9257 |     0.9734 |            0.0326 |
| scenarioB_amf_seg01 |   0.9424 |         0.9329 |                0.7418 |     0.8885 |           -0.0539 |
| scenarioK_csi       | nan      |         1      |                1      |     1      |          nan      |
| scenarioK_kul       | nan      |         1      |                1      |     1      |          nan      |
| scenarioN_target_n  |   0.9739 |         0.9933 |                0.9937 |     0.9743 |            0.0004 |

## Algorithm: `XGBoost`

| scenario            |   BASE |   Join_All_BFS |   Join_All_BFS_Filter |   AutoFeat |   Δ AutoFeat−BASE |
|:--------------------|-------:|---------------:|----------------------:|-----------:|------------------:|
| scenario1           | 0.9662 |         0.948  |                0.9492 |     0.9662 |            0      |
| scenario2c          | 0.5395 |         0.9917 |                0.9793 |     0.9915 |            0.452  |
| scenarioA_lat95     | 0.9551 |         0.9631 |                0.9606 |     0.9796 |            0.0245 |
| scenarioA_lat99     | 0.9508 |         0.9551 |                0.9518 |     0.9765 |            0.0257 |
| scenarioB_amf_seg01 | 0.9412 |         0.9337 |                0.6856 |     0.8745 |           -0.0667 |
| scenarioK_csi       | 0      |         0.9896 |                0.9948 |     1      |            1      |
| scenarioK_kul       | 0      |         1      |                1      |     1      |            1      |
| scenarioN_target_n  | 0.9786 |         0.9955 |                0.8879 |     0.9784 |           -0.0002 |


## Scenario context

| Scenario | Purpose | Expected | Target |
|---|---|---|---|
| `scenario1` | Cross-service workload augmentation | refuse | `lat99` |
| `scenario2c` | Feature recovery (self-join via time) | showcase | `lat99` |
| `scenarioA_lat95` | Cross-app temporal (target lat95) | partial | `lat95` |
| `scenarioA_lat99` | Cross-app temporal (target lat99) | partial | `lat99` |
| `scenarioB_amf_seg01` | Within-app via amf segments (join on n) | partial | `lat99` |
| `scenarioK_csi` | KUL MaMIMO CSI (csi_as_features layout) | showcase | `target_x` |
| `scenarioK_kul` | KUL MaMIMO indoor localisation | showcase | `target_x` |
| `scenarioN_target_n` | Inverse problem (target = workload n) | refuse | `n` |
