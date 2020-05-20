from gen_data import *
from gwnet import *

sensor_ids, sensor_id_to_ind, adj_mx = load_adj()
#dataloader = load_dataset()
#scaler = dataloader['scaler']

print ('============================================> ', adj_mx[0].shape)
model = gwnet(name = 'gwnet_v_1',
              num_nodes = 207,
              dropout = 0.3,
              supports = adj_mx,
              support_len = len(adj_mx))

#for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
#    print (x.shape, y.shape)
#    break
