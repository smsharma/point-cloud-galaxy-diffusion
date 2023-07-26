import jax.numpy as np

class PeriodicNormal:
    def __init__(
       self, 
       unit_cell: np.array,
       coord_std: np.array,
       box_size: float,
       loc: np.array,
       scale: float,
       n_pos_features: int = 3,
    ):
        """ A multivariate normal distribution with periodic boundary conditions on its mean.

        Args:
            unit_cell (np.array): unit cell for applying periodic boundary conditions 
            coord_std (np.array): standard deviation of the coordinates (assumed coordinates 
                                    are standardized)
            box_size (float): size of the box 
            loc (np.array): mean of the distribution 
            scale (float): scale of the distribution 
        """
        self.unit_cell = unit_cell
        self.coord_std = coord_std
        self.box_size = box_size
        self.loc = loc
        self.scale = scale
        self.n_pos_features = n_pos_features

    def log_prob(self, x: np.array)->np.array:
        """ Compute the log probability of a set of points under the distribution.

        Args:
            x (np.array): a particle configuration

        Returns:
            np.array: the log probability of that particle configuration 
        """
        dr = (x - self.loc) * self.coord_std
        dr = dr.at[...,:self.n_pos_features].set(
            apply_pbc(dr[...,:self.n_pos_features], self.box_size*self.unit_cell,)
        )
        dr = dr/ self.coord_std
        log_unnormalized = -0.5 * 1./self.scale ** 2 * dr**2
        log_normalization =  0.5 * np.log(2.*np.pi) + np.log(self.scale)
        return log_unnormalized - log_normalization

    def mean(self)->np.array:
        """ Get the mean of the distribution

        Returns:
            np.array: mean 
        """
        return self.loc * np.ones_like(self.scale)

def wrap_positions_to_periodic_box(positions: np.array, cell_matrix: np.array)->np.array:
    """
    Apply periodic boundary conditions to a set of positions.

    Args:
        positions (np.array): An array of shape (N, 3) containing the particle positions.
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        numpy.ndarray: An array of shape (N, 3) containing the wrapped particle positions.
    """
    inv_cell_matrix = np.linalg.inv(cell_matrix)
    fractional_positions = np.matmul(positions, inv_cell_matrix)
    fractional_positions = np.mod(fractional_positions, 1.0)
    return np.matmul(fractional_positions, cell_matrix)

def apply_pbc(dr: np.array, cell: np.array) -> np.array:
    """Apply periodic boundary conditions to a displacement vector, dr, given a cell.

    Args:
        dr (np.array): An array of shape (N,3) containing the displacement vector
        cell_matrix (np.array): A 3x3 matrix describing the box dimensions and orientation.

    Returns:
        np.array: displacement vector with periodic boundary conditions applied
    """
    return dr - np.round(dr.dot(np.linalg.inv(cell))).dot(cell)

