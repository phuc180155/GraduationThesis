import numpy as np
import pandas as pd
import os, sys
sys.path.append("/mnt/disk1/doan/phucnp/Graduation_Thesis/other_model_repo/vest-python")

from vest.models.bivariate import BivariateVEST, multivariate_feature_extraction

from vest.config.aggregation_functions \
    import (SUMMARY_OPERATIONS_ALL,
            SUMMARY_OPERATIONS_FAST,
            SUMMARY_OPERATIONS_SMALL)

series = np.random.random((20, 3))
series = pd.DataFrame(series, columns=['x', 'y', 'z'])

model = BivariateVEST()

features = \
    model.extract_features(series,
                           pairwise_transformations=False,
                           summary_operators=SUMMARY_OPERATIONS_SMALL)

pd.DataFrame(features.items())

multivariate_feature_extraction(series,
                                apply_transform_operators=False,
                                summary_operators=SUMMARY_OPERATIONS_SMALL)
