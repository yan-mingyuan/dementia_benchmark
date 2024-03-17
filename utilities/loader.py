from config import *

import os
import pandas as pd


def get_data_wave(wave):
    file_path = os.path.join(DIR_RW, f'w{wave}_subsets.dta')
    wave_data = pd.read_stata(file_path)

    # Columns to drop
    # cols_related = []
    cols_related = [
        'walkra', 'dressa', 'batha', 'eata', 'beda', 'adlwa',
        'phonea', 'moneya', 'medsa', 'shopa', 'mealsa', 'iadla',
        'tr20',
    ]
    cols_to_drop = SELF_DEM_COLS + PROXY_DEM_COLS + cols_related
    wave_data.drop(cols_to_drop, axis=1, inplace=True)

    features, labels = wave_data.drop(['demcls'], axis=1), wave_data['demcls']
    return features, labels
