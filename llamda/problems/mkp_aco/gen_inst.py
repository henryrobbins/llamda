# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/problems/mkp_aco/gen_inst.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import numpy as np
import numpy.typing as npt


def gen_instance(
    n: int, m: int
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Generate *well-stated* MKP instances
    Args:
        n: # of knapsacks
        m: # of constraints, a.k.a., the problem dimensionality
    """
    prize = np.random.rand(n)
    weight_matrix = np.random.rand(n, m)
    constraints = np.random.uniform(low=weight_matrix.max(0), high=weight_matrix.sum(0))
    # after norm, constraints are all 1
    weight_matrix = weight_matrix / constraints.reshape(1, *constraints.shape)
    return prize, weight_matrix  # (n, ), (n, m)


def generate_dataset(filepath: str, n: int, m: int, batch_size: int = 64) -> None:
    prizes = []
    weights = []
    for _ in range(batch_size):
        prize, weight = gen_instance(n, m)
        prizes.append(prize)
        weights.append(weight)
    prizes = np.stack(prizes)
    weights = np.stack(weights)
    np.savez(filepath, prizes=prizes, weights=weights)


def generate_datasets(basepath: str | None = None) -> None:
    import os

    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    m = 5
    for mood, seed, problem_sizes in [
        ("train", 1234, (100,)),
        ("val", 3456, (100, 300, 500)),
        ("test", 4567, (100, 200, 300, 500, 1000)),
    ]:
        np.random.seed(seed)
        batch_size = 5 if mood == "train" or mood == "val" else 64
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.npz")
            generate_dataset(filepath, n, m, batch_size=batch_size)


if __name__ == "__main__":
    generate_datasets()
