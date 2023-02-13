import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path

DEFAULT_PATH_DATA = Path("/n/holyscratch01/iaifi_lab/ccuesta/data_for_sid/")


class NbodyDataset:
    def __init__(
        self,
        stage: str = "train",
        path_to_data: Path = DEFAULT_PATH_DATA,
        params_to_condition: List[str] = ["Omega_m", "sigma_8"],
        include_pos_in_features: bool = False,
        boxsize: float = 1000.0,
        n_features: int = 7,
        n_particles: Optional[int] = None,
    ):
        """Create a dataset for N-body simulations

        Args:
            stage (str, optional): stage of the dataset. Defaults to "train".
            path_to_data (Path, optional): path to where data is stored. Defaults to DEFAULT_PATH_DATA.
            params_to_condition (List[str], optional): parameters to condition on. Defaults to ["Omega_m", "sigma_8"].
            include_pos_in_features (bool, optional): whether to include positions in the features. Defaults to False.
            boxsize (float, optional): size of the simulation box. Defaults to 1000.0.
            n_features (int, optional): number of features. Defaults to 7.
            n_particles (Optional[int], optional): number of particles, it will sample the most massive objects. 
            Defaults to None.
        """
        features = np.load(path_to_data / f"{stage}_halos.npy")
        if n_particles:
            features = features[:, :n_particles, :]
        self.conditioning = pd.read_csv(path_to_data / f"{stage}_cosmology.csv")[
            params_to_condition
        ].values
        self.params_to_condition = params_to_condition
        self.positions = features[..., :3] / boxsize  # Normalize positions
        if include_pos_in_features:
            start_idx = 0
        else:
            start_idx = 3
        end_idx = start_idx + n_features
        self.features = features[..., start_idx:end_idx]
        if n_features == 7:
            # Transform mass by log
            self.features[..., -1] = np.log10(self.features[..., -1])
        self.mask = np.ones((self.features.shape[0], self.features.shape[1]))

    def __len__(self) -> int:
        """Number of examples in the dataset

        Returns:
            int: number of examples
        """
        return len(self.features)

    def get_standarization_dict(
        self,
    ) -> Dict[str, np.array]:
        """Get feature standarization

        Returns:
            Dict[str, np.array]: dictionary with mean and std of the dataset 
        """
        return {
            "mean": self.features.mean(axis=(0, 1)),
            "std": self.features.std(axis=(0, 1)),
        }

    def __getitem__(self, idx: int) -> Tuple[np.array, np.array, np.array]:
        """Get an example from the dataset

        Args:
            idx (int): index of the example

        Returns:
            Data: example
        """
        return self.positions[idx], self.features[idx], self.conditioning[idx]
