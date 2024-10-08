from .sceneflow_dataset import SceneFlowDatset
from .usvinland_dataset import USVInlandDataset
from .kitti_dataset import KITTIDataset
from .mb_eth_dataset import MyDataset
from .spring_dataset import SpringDataset

__datasets__ = {
    "sceneflow": SceneFlowDatset,
    "usvinland": USVInlandDataset,
    "kitti": KITTIDataset,
    "middlebury": MyDataset,
    "eth3d": MyDataset,
    "spring": SpringDataset
}
