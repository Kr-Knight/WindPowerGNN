import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def matrix(p, y, num):
    error = np.abs(p - y)
    # max_bias = np.max(error)
    rmse = np.sqrt(mean_squared_error(p, y))
    mae = np.mean(error)
    acc = (1 - rmse)*100
    # var_base = np.var(error)
    print('Station{}-- MAE:{}, RMSE:{}, ACC:{}'.
          format(num, mae, rmse, acc))

    return acc
