import numpy as np
data = np.loadtxt('healthcare-dataset-stroke-data.csv', delimiter=',', dtype=object)
column_names = data[0]
data = data[1:]
