import numpy as np

# returns the input and output for the
# training and test set
# inputs
# -> data_file (.npy file with the data for the entire time period)
# -> timesep (number of intervals used to predict the next one)
# -> num_days_test (the size of the test set in number of days)
def load_data(data_file,timestep,num_days_test):
    # there are 48 time slots per day
    len_test = 48 * num_days_test
    
    print("loading OD matrix from",data_file)
    oddata_raw = np.load(data_file)[()]
    map_height = oddata_raw.shape[2]
    map_width = oddata_raw.shape[3]
    print("raw data has shape ",oddata_raw.shape)

    #normalizing the OD matrix
    odmax = np.max(oddata_raw)
    print("OD max is",odmax)
    oddata = oddata_raw * 2.0 / odmax - 1.0

    x = np.concatenate([oddata[i:i-timestep, np.newaxis, ...] for i in range(timestep)], axis=1)
    x_train, x_test = x[:-len_test], x[-len_test:]

    y = []
    y.append(oddata[timestep:,...])
    y = np.concatenate(y)
    y_train, y_test = y[:-len_test], y[-len_test:]

    print("Finishsed loading data. Returning sets with shapes:")
    print("X train:",x_train.shape)
    print("X test:",x_test.shape)
    print("y train:",y_train.shape)
    print("y test:",y_test.shape)

    return x_train, y_train, x_test, y_test, odmax, map_width, map_height

if __name__ == '__main__':
    fname = "/data/dissertation_deliverables/data/demand_model_data/od_matrix_100pc_20x5.npy"

    load_data(fname,5,60)