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
    def __init__(self, name, shape, axes):
        self.name = name
        self.shape = shape
        self.axes = axes
        self.vars = {}

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['gamma'] = tf.get_variable('gamma', initializer=tf.ones(self.shape))
            self.vars['beta']  = tf.get_variable('beta', initializer=tf.zeros(self.shape))

    def __call__(self, x):
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

        h = tf.concatenate(out, axis=3)
        h = self.mlp(h)
        h = tf.nn.dropout(h, self.dropout)
        return h

class gwnet:
    def __init__(self, name, num_nodes, dropout=0.3, support_len=None, gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2,
                 out_dim=12, residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512, kernel_size=2,
                 blocks=4, layers=2, num_slots=12, batch_size=64):
        self.name = name
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj
        self.num_slots = num_slots
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.out_dim = out_dim

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

        count = 1
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
                                          shape=[batch_size, num_slots-count, num_nodes, 1],
                                          axes=[0,1,2]))
                count += 1

                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(gcn(dilation_channels, residual_channels, dropout, support_len=self.support_len))

        self.end_conv_1 = conv2d(name='end_conv_1',
                                 shape=[1,1,skip_channels,end_channels], 
                                 dilation=1)

        self.end_conv_2 = conv2d(name='end_conv_2',
                                 shape=[1,1,end_channels, out_dim],
                                 dilation=1)

        self.receptive_field = receptive_field
        self.computation_graph()

    def computation_graph(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, self.num_slots, self.num_nodes, self.in_dim])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.out_dim, self.num_nodes, 1])

        in_len = self.num_slots
        if in_len < self.receptive_field:
            padding = tf.constant([[0,0],[self.receptive_field - in_len, 0],[0,0],[0,0]])
            x = tf.pad(self.x, padding)
        else:
            x = self.x

        x = self.start_conv(x)
        skip = 0

        new_supports = None

        if self.gcn_bool and self.addaptadj and self.supports is not None:
            adp = tf.nn.softmax(tf.nn.relu(tf.matmul(self.nodevec1,self.nodevec2)),axis=1)
            new_supports = self.supports + [adp]

        for i in range(self.blocks * self.layers):
            residual = x
            _filter = self.filter_convs[i](residual)
            _filter = tf.nn.tanh(_filter)
            gate = self.gate_convs[i](residual)
            gate = tf.nn.sigmoid(gate)
            x = tf.multiply(_filter, gate)

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:,-tf.shape(s)[1]:,:,:]
            except:
                skip = 0
            skip = s + skip

            if self.gcn_bool and self.supports is not None:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, self.supports)
            else:
                x = self.residual_convs[i](x)

            x = x + residual[:,-tf.shape(x)[1]:,:,:]
            x = self.bn[i](x)

        x = tf.nn.relu(skip)
        x = tf.nn.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)
        return x




















































