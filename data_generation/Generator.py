# Adapted from "f-sensitivity-through-evar" by Matej Havelka, used under the MIT License.
# Source: https://github.com/MatejHav/f-sensitivity-through-evar
# Copyright (c) 2024 Matej Havelka. See LICENSE at the repository root for full terms.

import threading
import numpy as np
import pandas as pd
from tqdm import tqdm

from data_generation import DataObject


class Generator:
    """Generate causal data (U, X, T, Y) with a hidden confounder from user-supplied generators."""

    def __init__(self, generators: dict, noise_generators: dict, sizes: dict):
        assert all(c in generators and c in noise_generators and c in sizes
                   for c in ["U", "X", "T", "Y"]), "All inputs require U, X, T and Y"
        self.size = sizes
        self.generators = generators
        self.noise_generators = noise_generators

    def generate(self, num_rows: int, n_jobs: int, path: str, verbose: int = 0):
        """Generate num_rows samples over n_jobs threads, save to `path` (overwritten), return
        (DataObject, path)."""
        assert num_rows >= n_jobs, "Number of samples must be >= number of jobs."
        data = []

        def _generator_helper(k):
            bar = tqdm(range(k)) if verbose > 0 else range(k)
            for _ in bar:
                U = self.generators["U"](self.noise_generators["U"]())
                X = self.generators["X"](U, self.noise_generators["X"]())
                T = self.generators["T"](U, X, self.noise_generators["T"]())
                Y = self.generators["Y"](U, X, T, self.noise_generators["Y"]())
                data.append([*U, *X, *T, *Y])

        threads = [threading.Thread(target=_generator_helper, args=[num_rows // n_jobs])
                   for _ in range(n_jobs)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        df = pd.DataFrame(data, columns=[*[f"U{i}" for i in range(self.size["U"])],
                                         *[f"X{i}" for i in range(self.size["X"])],
                                         "T", "Y"])
        df.to_csv(path, index=False, columns=df.columns)
        return DataObject(df), path
