import tensorflow as tf

class nconv:
    def __init__(self, name):
        self.name = name

    def __call__(self, x, A):
        out = tf.einsum('ncvl,vw->ncwl', x, A)
        return out

class linear:
    def __init__(self, name, c_in, c_out):
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
        self.vars = {}

        with tf.variable_scope(self.name) as scope:
            self.vars['weight'] = init_weight(name='weight', shape=[1,1,self.c_in,self.c_out])
            self.vars['bias'] = init_bias(name='bias', shape=[self.c_out])

    def __call__(self, x):
        out = tf.nn.conv2d(x, self.vars['weight'], strides=[1,1,1,1], padding='VALID') + self.vars['bias']
        return out

class conv2d:
    def __init__(self, name, shape, dilation):
        self.name = name
        self.shape = shape
        self.dilation = dilation
        self.vars = {}

        with tf.variable_scope(self.name) as scope:
            self.vars['weight'] = init_weight(name='weight', shape=shape)
            self.vars['bias'] = init_bias(name='bias', shape=shape[-1])

    def __call__(self, x):
        out = tf.nn.conv2d(x, self.vars['weight'], strides=[1,1,1,1], padding='VALID', dilation=self.dilation) + self.vars['bias']
        return out

class Layer_norm():
    def __init__(self, name, axes):
        self.name = name 
        self.axes = axes
        self.vars = {}

    def __call__(self, x):
        shape = tf.shape(x)
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['gamma'] = tf.get_variable('gamma', initializer=tf.ones(shape))
            self.vars['beta']  = tf.get_variable('beta', initializer=tf.zeros(shape))

        mu, sigma = tf.nn.moments(x, axes=self.axes, keep_dims=True)
        _x = (x - mu) / tf.sqrt(sigma + 1e-5) * self.vars['gamma'] + self.vars['beta']
        return _x

class gcn:
    def __init__(self, name, c_in, c_out, dropout, support_len=3, order=2):
        self.name = name
        self.c_in = c_in
        self.c_out = c_out
        self.dropout = dropout
        self.support_len = support_len
        self.order = order

        with tf.variable_scope(self.name) as scope:
            self.nconv = nconv(name='nconv')
            self.c_in = (self.order * self.support_len + 1) * self.c_in
            self.mlp = linear('linear', self.c_in, self.c_out)

    def __call__(self, x, support):
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = tf.concatenate(out, axis=1)
        h = self.mlp(h)
        h = tf.nn.dropout(h, self.dropout)
        return h

class gwnet:
    def __init__(self, name, num_nodes, dropout=0.3, support_len=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,
                 out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2,
                 blocks=4, layers=2):
        self.name = name
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = []
        self.gate_convs = []
        self.residual_convs = []
        self.skip_convs = []
        self.bn = []
        self.gconv = []

        self.start_conv = linear(name='start_conv', c_in=in_dim, c_out=residual_channels)
        self.supports = supports
        receptive_field = 1

        self.supports_len = 1
        if supports is not None:
            self.supports_len += len(self.supports)

        if gcn_bool and addaptadj:
            if aptinit is None:
                if supports is None:
                    self.supports = []

                self.nodevec1 = init_weight(name='E1', shape=[num_nodes, 10])
                self.nodevec1 = init_weight(name='E2', shape=[10, num_nodes])
                self.supports_len += 1
            else:
                print ('aptinit must be None')
                exit()

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                self.filter_convs.append(conv2d(name=f'filer_conv2d_{b}_{i}', 
                                                shape=[kernel_size,1,residual_channels,dilation_channels], 
                                                dilation=new_dilation))
                
                self.gate_convs.append(conv2d(name=f'gate_conv2d_{b}_{i}', 
                                              shape=[kernel_size,1,residual_channels,dilation_channels], 
                                              dilation=new_dilation))
                
                self.residual_convs.append(conv2d(name=f'residual_conv2d_{b}_{i}', 
                                                  shape=[1,1,dilation_channels,residual_channels],
                                                  dilation=1))
                
                self.skip_convs.append(conv2d(name=f'skip_conv2d_{b}_{i}', 
                                              shape=[1,1,dilation_channels,skip_channels], 
                                              dilation=1))

                self.bn.append(Layer_norm(name=f'batch_norm_{b}_{i}',
                                          axes=[3]))






    def computation_graph(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.n_slots, self.n_nodes, self.din]) 
        self.layers = []


















































