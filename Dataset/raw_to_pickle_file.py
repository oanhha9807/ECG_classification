import pandas as pd
import numpy as np
import wfdb
import ast

def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

# path = '/home/oem/oanh/DL_ECG_Classification/physionet.org/files/ptb-xl/1.0.1/'
path = '/data/oanh/ECG_data/physionet.org/files/ptb-xl/1.0.1/'
sampling_rate=100

# load and convert annotation data
Y = pd.read_csv(path + 'ptbxl_database.csv', index_col='ecg_id')
Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

# Load raw signal data
X = load_raw_data(Y, sampling_rate, path)

# Load scp_statements.csv for diagnostic aggregation
agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
agg_df = agg_df[agg_df.diagnostic == 1]

def aggregate_diagnostic(y_dic):
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            tmp.append(agg_df.loc[key].diagnostic_class)
    return list(set(tmp))

# Apply diagnostic superclass
Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

# Split data into train and test (https://physionet.org/content/ptb-xl/1.0.1/)
test_fold = 10
dev_fold = 9
# Train
X_train = X[np.where((Y.strat_fold != test_fold) & (Y.strat_fold != dev_fold))]
y_train = Y[(Y.strat_fold != test_fold) & (Y.strat_fold != dev_fold)].diagnostic_superclass
#Dev
X_dev = X[np.where(Y.strat_fold == dev_fold)]
y_dev = Y[(Y.strat_fold == dev_fold)].diagnostic_superclass
# Test
X_test = X[np.where(Y.strat_fold == test_fold)]
y_test = Y[Y.strat_fold == test_fold].diagnostic_superclass




import pickle
pickle_out = open("X_train_final_100.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train_final_100.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()

pickle_out = open("X_dev_final_100.pickle","wb")
pickle.dump(X_dev, pickle_out)
pickle_out.close()

pickle_out = open("y_dev_final_100.pickle","wb")
pickle.dump(y_dev, pickle_out)
pickle_out.close()

pickle_out = open("X_test_final_100.pickle","wb")
pickle.dump(X_test, pickle_out)
pickle_out.close()

pickle_out = open("y_test_final_100.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()


def labelstovector(X, y):
    '''
    Convert the labels in y into vectors:
    Multi-label problem:
    Norm: [0,0,0,0]
    MI: [1,0,0,0]
    STTC: [0,1,0,0]
    CD: [0,0,1,0]
    HYP: [0,0,0,1]
    Combination example:
    HYP and MI: [1,0,0,1]
    HYP and CD and STTC: [0,1,1,1]
    -----------------------------------------------------------
    Args: X (number of examples, signal length, number of leads)
          y (number of examples, )
    '''
    y_list = []
    X_list = []
    for label, ecg in zip(y, X):
        if len(label) != 0:  # ignore examples with label = []
            aux_vec = np.zeros(5)
            if 'MI' in label:
                aux_vec[0] = 1
            if 'STTC' in label:
                aux_vec[1] = 1
            if 'CD' in label:
                aux_vec[2] = 1
            if 'HYP' in label:
                aux_vec[3] = 1
            if 'NORM' in label:
                aux_vec[4] = 1

            # print(label)
            # print(aux_vec)
            y_list.append(aux_vec)
            # print(ecg.shape)
            X_list.append(ecg)

    return X_list, y_list

X_train_processed, y_train_processed = labelstovector(X_train, y_train)
X_dev_processed, y_dev_processed = labelstovector(X_dev, y_dev)
X_test_processed, y_test_processed = labelstovector(X_test, y_test)

import pickle
pickle_out = open("X_train_processed_final_100.pickle","wb")
pickle.dump(X_train_processed, pickle_out)
pickle_out.close()

pickle_out = open("y_train_processed_final_100.pickle","wb")
pickle.dump(y_train_processed, pickle_out)
pickle_out.close()

pickle_out = open("X_dev_processed_final_100.pickle","wb")
pickle.dump(X_dev_processed, pickle_out)
pickle_out.close()

pickle_out = open("y_dev_processed_final_100.pickle","wb")
pickle.dump(y_dev_processed, pickle_out)
pickle_out.close()

pickle_out = open("X_test_processed_final_100.pickle","wb")
pickle.dump(X_test_processed, pickle_out)
pickle_out.close()

pickle_out = open("y_test_processed_final_100.pickle","wb")
pickle.dump(y_test_processed, pickle_out)
pickle_out.close()

