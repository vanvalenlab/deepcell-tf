class UnmixingLayer(Layer):
    def __init__(self,
                n_dyes=4,
                n_channels=4,
                use_bias=True,
                kernel_initializer='glorot_uniform',
                bias_initializer='zeros',
                kernel_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                bias_constraint=None,
                **kwargs):
        super(UnmixingLayer, self).__init__(**kwargs)
        self.n_dyes=4
        self.n_channels=4
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking=True
        self.input_spec = InputSpec(min_ndim=2)

    def build(self, input_shape):
     if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[channel_axis].value is None:
            raise ValueError('The channel dimension of the inputs to '
                             '`TensorProduct` should be defined. '
                             'Found `None`.')
        input_dim = int(input_shape[channel_axis])
        kernel_shape = (input_dim, self.output_dim)

        self.kernel = self.add_weight(
            'kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(
                'bias',
                shape=(self.output_dim,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True)
        else:
            self.bias = None

        # Set input spec.
        self.input_spec = InputSpec(min_ndim=2,
                                    axes={channel_axis: input_dim})
        self.built = True