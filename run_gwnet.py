from gen_data import *

sensor_ids, sensor_id_to_ind, adj_mx = load_adj()
dataloader = load_dataset()
scaler = dataloader['scaler']

print (adj_mx[0].shape)
