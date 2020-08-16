"""
Convert IMDB META data(matlab .mat) to csv.
IMDB-WIKI – 500k+ face images with age and gender labels
- https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
- https://stackoverflow.com/questions/13965740/converting-matlabs-datenum-format-to-python
"""

from datetime import datetime, timedelta
import numpy as np
import scipy
from scipy.io import loadmat
import pandas as pd

class mat_to_csv:

    def __init__(self):
        super().__init__()

    def matlab_datenum2dt(self, matlab_datenum):
        return datetime.fromordinal(int(matlab_datenum) - 366) +\
                timedelta(days=int(matlab_datenum%1))

    def convert(self, path_mat, key):

        path_save = 'wiki/wiki.pkl'
        mat = loadmat(path_mat)

        print(mat['__header__'])
        print(mat['__version__'])

        # Extract values
        dt = mat[key][0, 0]

        # Check for columns
        print('columns:\n', dt.dtype.names)

        # Extract values with simple format
        keys_s = ('gender', 'dob', 'photo_taken',
                'face_score', 'second_face_score')
        values = {k: dt[k].squeeze() for k in keys_s}

        # Extract values with nested format
        keys_n = ('full_path', 'name')
        for k in keys_n:
            values[k] = np.array([x if not x else x[0] for x in dt[k][0]])

        # Convert face location to DataFrame
        # img(face_location(2):face_location(4),face_location(1):face_location(3),:))
        values['face_location'] =\
            [tuple(x[0].tolist()) for x in dt['face_location'].squeeze()]

        # Check all values extracted have same length
        set_nrows = {len(v) for _, v in values.items()}
        assert len(set_nrows) == 1

        df_values = pd.DataFrame(values)

        df_values.rename(columns = {'name': 'full_name'}, inplace=True)

        # # Convert matlab datenum to datetime
        # df_values['dob'] = df_values['dob'].apply(self.matlab_datenum2dt)

        # # Calc ages when photo taken
        # df_values['photo_taken_age'] = \
        #     df_values.apply(lambda x: x['photo_taken'] - x['dob'].year, axis=1)

        # Concat all together and save
        # Do not use csv format to work around tuple to be string
        # df_values.to_pickle(path_save)

        return df_values
