from mxnet import gluon as g
import mxnet as mx
from utils import gauss_kernel
import os

def _conv3x3(channels, stride, in_channels):
    return g.nn.Conv2D(channels, kernel_size=3, strides=stride, padding=1, use_bias=False, in_channels=in_channels)

class BasicBlockV2(g.nn.HybridBlock):
    r"""BasicBlock V2 from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    This is used for ResNet V2 for 18, 34 layers.
    Parameters
    ----------
    channels : int
        Number of output channels.
    stride : int
        Stride size.
    downsample : bool, default False
        Whether to downsample the input.
    in_channels : int, default 0
        Number of input channels. Default is 0, to infer from the graph.
    """
    def __init__(self, channels, stride, downsample=False, in_channels=0, **kwargs):
        super(BasicBlockV2, self).__init__(**kwargs)
        self.bn1 = g.nn.InstanceNorm()
        self.conv1 = _conv3x3(channels, stride, in_channels)
        self.bn2 = g.nn.InstanceNorm()
        self.conv2 = _conv3x3(channels, 1, channels)
        if downsample:
            self.downsample = g.nn.Conv2D(channels, 1, stride, use_bias=False,
                                        in_channels=in_channels)
        else:
            self.downsample = None

    def hybrid_forward(self, F, x):
        residual = x
        x = self.bn1(x)
        x = F.Activation(x, act_type='relu')
        if self.downsample:
            residual = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = F.Activation(x, act_type='relu')
        x = self.conv2(x)
        return x + residual


class ResNetV2(g.nn.HybridBlock):
    r"""ResNet V2 model from
    `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    block : HybridBlock
        Class for the residual block. Options are BasicBlockV1, BottleneckV1.
    layers : list of int
        Numbers of layers in each block
    channels : list of int
        Numbers of channels in each block. Length should be one larger than layers list.
    classes : int, default 1000
        Number of classification classes.
    thumbnail : bool, default False
        Enable thumbnail.
    """
    def __init__(self, block, layers, channels, **kwargs):
        super(ResNetV2, self).__init__(**kwargs)
        assert len(layers) == len(channels) - 1
        with self.name_scope():
            self.features = g.nn.HybridSequential(prefix='generator')
            self.features.add(g.nn.Conv2D(channels=64, kernel_size=9, strides=1, padding=1, use_bias=True))
            self.features.add(g.nn.Activation('relu'))

            in_channels = channels[0]
            for i, num_layer in enumerate(layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(block, num_layer, channels[i+1],
                                                   stride, i+1, in_channels=in_channels))
                in_channels = channels[i+1]

            self.features.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=3, use_bias=True))
            self.features.add(g.nn.Activation('relu'))
            self.features.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=3, use_bias=True))
            self.features.add(g.nn.Activation('relu'))
            self.features.add(g.nn.Conv2D(channels=3, kernel_size=9, strides=1, padding=3, use_bias=True))
            self.features.add(g.nn.Activation('tanh'))

    def _make_layer(self, block, layers, channels, stride, stage_index, in_channels=0):
        layer = g.nn.HybridSequential(prefix='stage%d_'%stage_index)
        with layer.name_scope():
            layer.add(block(channels, stride, channels != in_channels, in_channels=in_channels,
                            prefix=''))
            for _ in range(layers-1):
                layer.add(block(channels, 1, False, in_channels=channels, prefix=''))
        return layer

    def hybrid_forward(self, F, x):
        x = self.features(x)
        return x

# Specification
resnet_spec = {11: ('basic_block', [1, 1, 1, 1], [64, 64, 64, 64])}

resnet_block_versions = [{'basic_block': BasicBlockV2}]

# Constructor
def get_resnet(num_layers, **kwargs):
    r"""ResNet V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    ResNet V2 model from `"Identity Mappings in Deep Residual Networks"
    <https://arxiv.org/abs/1603.05027>`_ paper.
    Parameters
    ----------
    version : int
        Version of ResNet. Options are 1, 2.
    num_layers : int
        Numbers of layers. Options are 18, 34, 50, 101, 152.
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    assert num_layers in resnet_spec, \
        "Invalid number of layers: %d. Options are %s"%(
            num_layers, str(resnet_spec.keys()))
    block_type, layers, channels = resnet_spec[num_layers]

    resnet_class = ResNetV2
    block_class = BasicBlockV2
    net = resnet_class(block_class, layers, channels, **kwargs)

    return net

def resnet11_v2(**kwargs):
    r"""ResNet-50 V1 model from `"Deep Residual Learning for Image Recognition"
    <http://arxiv.org/abs/1512.03385>`_ paper.
    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    ctx : Context, default CPU
        The context in which to load the pretrained weights.
    root : str, default '~/.mxnet/models'
        Location for keeping the model parameters.
    """
    return get_resnet(11, **kwargs)


class blur(g.nn.HybridBlock):
    def __init__(self):
        super(blur, self).__init__()

    def hybrid_forward(self, F, x, kernel_var):
        # self.kernel_var = mx.nd.transpose(mx.nd.array(self.kernel_var), (3, 2, 0, 1))
        x = F.Convolution(data=x, weight=kernel_var, num_filter=3, kernel=(21, 21), stride=(1, 1), pad=(1, 1), num_group=3,no_bias=True)
        return x

class resnet(g.nn.HybridBlock):
    def __init__(self):
        super(resnet, self).__init__()
        with self.name_scope():
            self.body1 = g.nn.HybridSequential('generator_1')
            self.body1.add(g.nn.Conv2D(channels=64, kernel_size=9, strides=1, padding=1, use_bias=True))
            self.body1.add(g.nn.Activation('relu'))

            self.body_res1 = g.nn.HybridSequential('generator_res1')
            self.body_res1.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res1.add(g.nn.InstanceNorm())
            self.body_res1.add(g.nn.Activation('relu'))
            self.body_res1.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res1.add(g.nn.InstanceNorm())
            self.body_res1.add(g.nn.Activation('relu'))

            self.body_res2 = g.nn.HybridSequential('generator_res2')
            self.body_res2.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res2.add(g.nn.InstanceNorm())
            self.body_res2.add(g.nn.Activation('relu'))
            self.body_res2.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res2.add(g.nn.InstanceNorm())
            self.body_res2.add(g.nn.Activation('relu'))

            self.body_res3 = g.nn.HybridSequential('generator_res3')
            self.body_res3.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res3.add(g.nn.InstanceNorm())
            self.body_res3.add(g.nn.Activation('relu'))
            self.body_res3.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res3.add(g.nn.InstanceNorm())
            self.body_res3.add(g.nn.Activation('relu'))

            self.body_res4 = g.nn.HybridSequential('generator_res4')
            self.body_res4.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res4.add(g.nn.InstanceNorm())
            self.body_res4.add(g.nn.Activation('relu'))
            self.body_res4.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=1, use_bias=True))
            self.body_res4.add(g.nn.InstanceNorm())
            self.body_res4.add(g.nn.Activation('relu'))

            self.body2 = g.nn.HybridSequential('generator_2')
            self.body2.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=3, use_bias=True))
            self.body2.add(g.nn.Activation('relu'))
            self.body2.add(g.nn.Conv2D(channels=64, kernel_size=3, strides=1, padding=3, use_bias=True))
            self.body2.add(g.nn.Activation('relu'))

            self.body3 = g.nn.HybridSequential('generator_final')
            self.body3.add(g.nn.Conv2D(channels=3, kernel_size=9, strides=1, padding=3, use_bias=True))
            self.body3.add(g.nn.Activation('tanh'))

    def forward(self, x):
        x = self.body1(x)
        residual1 = self.body_res1(x)
        x = residual1 + x
        residual2 = self.body_res2(x)
        x = residual2 + x
        residual3 = self.body_res3(x)
        x = residual3 + x
        residual4 = self.body_res4(x)
        x = residual4 + x
        x = self.body2(x)
        enhanced = self.body3(x) * 0.58 + 0.5
        return enhanced

class adversarial(g.nn.HybridBlock):
    def __init__(self):
        super(adversarial, self).__init__()
        with self.name_scope():
            self.body = g.nn.HybridSequential('discriminator')
            self.body.add(g.nn.Conv2D(channels=48, kernel_size=11, strides=4, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))

            self.body.add(g.nn.Conv2D(channels=128, kernel_size=5, strides=2, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))
            self.body.add(g.nn.InstanceNorm())

            self.body.add(g.nn.Conv2D(channels=192, kernel_size=3, strides=1, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))
            self.body.add(g.nn.InstanceNorm())

            self.body.add(g.nn.Conv2D(channels=192, kernel_size=3, strides=1, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))
            self.body.add(g.nn.InstanceNorm())

            self.body.add(g.nn.Conv2D(channels=128, kernel_size=3, strides=2, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))
            self.body.add(g.nn.InstanceNorm())

            self.body.add(g.nn.Flatten())
            self.body.add(g.nn.Dense(1024, use_bias=True))
            self.body.add(g.nn.LeakyReLU(0.2))

            self.body.add(g.nn.Dense(2, use_bias=True))

    def hybrid_forward(self, F, x):
        adv_out_logits = self.body(x)
        adv_out = F.softmax(adv_out_logits)
        return adv_out

# def adversarial(image_):
#     with tf.variable_scope("discriminator"):
#         conv1 = _conv_layer(image_, 48, 11, 4, batch_nn=False)
#         conv2 = _conv_layer(conv1, 128, 5, 2)
#         conv3 = _conv_layer(conv2, 192, 3, 1)
#         conv4 = _conv_layer(conv3, 192, 3, 1)
#         conv5 = _conv_layer(conv4, 128, 3, 2)
#
#         flat_size = 128 * 7 * 7
#         conv5_flat = tf.reshape(conv5, [-1, flat_size])
#
#         W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
#         bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))
#
#         fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)
#
#         W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
#         bias_out = tf.Variable(tf.constant(0.01, shape=[2]))
#
#         adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
#
#     return adv_out
#
#
# def _conv_init_vars(net, out_channels, filter_size, transpose=False):
#     _, rows, cols, in_channels = [i.value for i in net.get_shape()]
#
#     if not transpose:
#         weights_shape = [filter_size, filter_size, in_channels, out_channels]
#     else:
#         weights_shape = [filter_size, filter_size, out_channels, in_channels]
#
#     weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
#
#
# return weights_init