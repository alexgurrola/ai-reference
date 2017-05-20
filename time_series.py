import numpy as np
import pandas as pd

rng = pd.date_range('1/1/2011', periods=72, freq='H')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.head())
