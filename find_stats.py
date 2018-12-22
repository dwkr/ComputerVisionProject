import os
import json
import numpy as np

along_axes = (0,2,3)
statsfile = "stats.json"

def findStats(logging, train_data_loader, calculate):

    if not calculate:
        if os.path.exists(statsfile):
            with open(statsfile, 'r') as file:
                #logging.info('\nReading stats from file {} \n'.format(statsfile))
                return json.load(file)
        else:
            logging.info('\nFile {} not found.Calculating stats.\n'.format(statsfile))

    mean_X1, mean_X2, mean_X3, = 0, 0, 0
    var_X1, var_X2, var_X3 = 0, 0, 0
    for batch_idx, (X, Y) in enumerate(train_data_loader):
        X = [x.numpy() for x in X]
        mean_X1 += np.mean(X[0]/255,axis=along_axes)
        mean_X2 += np.mean(X[1]/255,axis=along_axes)
        mean_X3 += np.mean(X[2]/255,axis=along_axes)
        var_X1 += np.square(np.std(X[0]/255,axis=along_axes))
        var_X2 += np.square(np.std(X[1]/255,axis=along_axes))
        var_X3 += np.square(np.std(X[2]/255,axis=along_axes))
        
    divisor = len(train_data_loader)
    mean_X1 /= divisor
    mean_X2 /= divisor
    mean_X3 /= divisor
    var_X1 /= divisor
    var_X2 /= divisor
    var_X3 /= divisor
    
    stats = { "X1": {"mean": mean_X1.tolist(), "std": np.sqrt(var_X1).tolist()},
              "X2": {"mean": mean_X2.tolist(), "std": np.sqrt(var_X2).tolist()},
              "X3": {"mean": mean_X3.tolist(), "std": np.sqrt(var_X3).tolist()}
            }
            
    with open(statsfile, 'w') as file:
        json.dump(stats,file)
    logging.info('\nSaved calculated stats to file {} \n'.format(statsfile))
    
    return stats