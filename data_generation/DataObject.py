# Adapted (simplified) from "f-sensitivity-through-evar" by Matej Havelka, used under the MIT
# License. Source: https://github.com/MatejHav/f-sensitivity-through-evar
# Copyright (c) 2024 Matej Havelka. See LICENSE at the repository root for full terms.

import pandas as pd


class DataObject:
    """Thin wrapper holding the generated DataFrame; experiments read `.data`."""

    def __init__(self, data: pd.DataFrame):
        self.data = data
