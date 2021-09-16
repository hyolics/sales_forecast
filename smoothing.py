'''對銷貨進行smoothing'''
import pandas as pd
import numpy as np


def triangular_smooth(time_ser, shift_day):
    time_ser = time_ser.resample('D').sum()

    df3 = pd.DataFrame(index=time_ser.index)
    smooth_range = np.arange(-shift_day, shift_day+1, 1)
    for day in smooth_range:
        df3[day] = time_ser.shift(day)
    df3 = df3.fillna(0)

    s = pd.Series(data=0, index=df3.index)
    w_sum = 0
    for col in df3.columns:
        w = shift_day + 1 - abs(col)
        s += w * df3[col]
        w_sum += w
    s = s / w_sum

    return s
