import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from pathlib import Path

DEFAULT_PATH_DATA = Path("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/")
EPS = 1.0e-7

class NbodyDataset(Dataset):
    def __init__(
        self,
        cutoff: float = 0.1,
        boxsize: float = 1000.0,
        stage: str = 'train',
        path_to_data: Path = DEFAULT_PATH_DATA,
        params_to_condition: List[str] = ["Omega_m", "sigma_8"],
        feature_standarization: Optional[Dict[str, np.array]] = None,
    ):
        """ Create a dataset for N-body simulations

        Args:
            cutoff (float, optional): cut-off radius beyond which nodes are not connected, in units of the boxsize. 
            Defaults to 0.1.
            boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
            path_to_data (Path, optional): path to where data is stored. Defaults to DEFAULT_PATH_DATA.
            params_to_condition (List[str], optional): parameters to condition on. Defaults to ["Omega_m", "sigma_8"].
            feature_standarization (Optional[Dict[str, np.array]], optional): Dictionary with parameters
            that define how to standarize the data. If not given, it'd compute mean and variance from the dataset. Defaults to None.
        """
        super().__init__()
        self.cutoff = cutoff 
        features = np.load(path_to_data / f'{stage}_halos.npy')
        self.conditioning = pd.read_csv(path_to_data / f'{stage}_cosmology.csv')[
            params_to_condition
        ].values
        self.params_to_condition = params_to_condition
        self.positions = features[..., :3] / boxsize  # Normalize positions
        self.features = features[..., 3:]
        # Transform mass by log
        self.features[..., -1] = np.log10(self.features[..., -1])
        # Standarize
        if feature_standarization is not None:
            self.features = (
                self.features - feature_standarization["mean"]
            ) / feature_standarization["std"]
        else:
            features_mean = self.features.mean(axis=(0, 1))
            features_std = self.features.std(axis=(0, 1))
            self.features = (self.features - features_mean + EPS) / (features_std + EPS)

    def __len__(self) -> int:
        """ Number of examples in the dataset

        Returns:
            int: number of examples 
        """
        return len(self.features)

    def get_standarization_dict(self,)->Dict[str, np.array]:
        return {
            'mean': self.features.mean(axis=(0, 1)),
            'std': self.features.std(axis=(0, 1)),
        }

    def __getitem__(self, idx: int) -> Data:
        """ Get an example from the dataset

        Args:
            idx (int): index of the example 

        Returns:
            Data: example 
        """
        pos = torch.tensor(self.positions[idx], dtype=torch.float32)
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        return Data(
            features=features,
            pos=pos,
            y=self.conditioning[idx],
        ):