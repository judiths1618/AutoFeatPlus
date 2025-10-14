from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Result:
    TFD_PATH = "TFD_PATH"
    TFD = "AutoFeat"
    TFD_REL = "AutoFeat_Rel"
    TFD_RED = "AutoFeat_Red"
    TFD_Pearson = "AutoFeat-Pearson-MRMR"
    TFD_Pearson_JMI = "AutoFeat-Pearson-JMI"
    TFD_JMI = "AutoFeat-Spearman-JMI"
    ARDA = "ARDA"
    JOIN_ALL_BFS = "Join_All_BFS"
    JOIN_ALL_BFS_F = "Join_All_BFS_Filter"
    JOIN_ALL_BFS_W = "Join_All_BFS_Wrapper"
    JOIN_ALL_DFS = "Join_All_DFS"
    JOIN_ALL_DFS_F = "Join_All_DFS_Filter"
    JOIN_ALL_DFS_W = "Join_All_DFS_Wrapper"
    BASE = "BASE"

    algorithm: str
    data_path: str = ""
    approach: str = ""
    data_label: str = ""
    join_time: float = 0.0
    total_time: float = 0.0
    feature_selection_time: float = 0.0
    depth: int = 0
    accuracy: float = 0.0
    train_time: float = 0.0
    feature_importance: Dict[str, float] = field(default_factory=dict)
    join_path_features: List[str] = field(default_factory=list)
    cutoff_threshold: float = 0.0
    redundancy_threshold: float = 0.0
    rank: int = 0
    top_k: int = 0

    def __post_init__(self):
        self.total_time += self.join_time + self.train_time + self.feature_selection_time
