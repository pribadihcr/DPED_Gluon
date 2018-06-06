import mxnet as mx
from mxnet.gluon.model_zoo import vision
from mxnet import gluon as g
from model import resnet, resnet11_v2, adversarial, blur
import numpy as np
from scipy import misc
import imageio
import numpy as np
import sys
from mxnet import ndarray
from util_loss import TextureLoss, ColorLoss
from mxnet.gluon.model_zoo import vision
# import models
import utils
from load_dataset import load_test_data, load_batch
# defining size of the training image patches
ctx = mx.gpu(0)
ctx1 = mx.gpu(0)

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, \
dped_dir, vgg_dir, eval_step = utils.process_command_args(sys.argv)

np.random.seed(0)



print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
print("Training data was loaded\n")

TEST_SIZE = len(test_data)
num_test_batches = int(len(test_data)/batch_size)

def Conv(data, num_filter, kernel=(1, 1), stride=(1, 1), pad=(0, 0), name=None, suffix='', withRelu=False, withBn=False, bn_mom=0.9, workspace=256):
    conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,
                              name='%s%s_conv2d' % (name, suffix), workspace=workspace)
    if withBn:
        conv = mx.sym.BatchNorm(data=conv, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    if withRelu:
        conv = mx.sym.Activation(data=conv, act_type='relu', name='%s%s_relu' % (name, suffix))
    return conv

def Separable_Conv(data, num_in_channel, num_out_channel, kernel=(3, 3), stride=(1, 1), pad=(1, 1), name=None, suffix='', depth_mult=1, withBn=False, bn_mom=0.9, workspace=256):
    # original version of Separable Convolution
    # depthwise convolution
    #channels       = mx.sym.split(data=data, axis=1, num_outputs=num_in_channel) # for new version of mxnet > 0.8
    channels       = mx.sym.SliceChannel(data=data, axis=1, num_outputs=num_in_channel) # for old version of mxnet <= 0.8
    depthwise_outs = [mx.sym.Convolution(data=channels[i], num_filter=depth_mult, kernel=kernel,
                           stride=stride, pad=pad, name=name+'_depthwise_kernel_'+str(i), workspace=workspace)
                           for i in range(num_in_channel)]
    depthwise_out = mx.sym.Concat(*depthwise_outs)
    # pointwise convolution
    pointwise_out = Conv(data=depthwise_out, num_filter=num_out_channel, name=name+'_pointwise_kernel', withBn=False, bn_mom=0.9, workspace=256)
    if withBn:
        pointwise_out = mx.sym.BatchNorm(data=pointwise_out, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='%s%s_bn' % (name, suffix))
    return pointwise_out

def load_transform(paths):
    data =  np.zeros((len(paths), PATCH_SIZE))
    ii = 0
    for path in paths:
        I = np.asarray(imageio.imread(path))
        I = np.float16(np.reshape(I, [1, PATCH_SIZE])) / 255
        data[ii, :] = I
        ii += 1

    data = np.reshape(data, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])
    data_ = mx.nd.array(data)
    data_ = mx.nd.transpose(data_.astype(np.float32), (0, 3, 1, 2))
    return data_


rgb_mean = (mx.nd.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)).as_in_context(ctx)
rgb_std = (mx.nd.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)).as_in_context(ctx)


# def normalize_image(data):
#     return (data.astype('float32') / 255 - rgb_mean) / rgb_std

def color_normalize(src, mean, std=None):
    if mean is not None:
        src = src - mean
    if std is not None:
        src = src / std
    return src

def preprocess_vgginput(image):
    # normalized = mx.image.color_normalize(image,
    #                        mean=mx.nd.array([0.485, 0.456, 0.406]),
    #                        std=mx.nd.array([0.229, 0.224, 0.225]))
    # image1 = image / 255

    image = color_normalize(image,
                            mean=rgb_mean,
                            std=rgb_std)
    # data = g.utils.split_and_load(normalized, ctx_list=ctx1, batch_axis=0)
    # image = mx.nd.transpose(image, (0, 2, 3, 1))
    # image = [normalize_image(im) for im in image]
    # image = mx.nd.transpose(mx.nd.array(np.array(image)), (0, 3, 1, 2))
    return image

vgg19 = vision.vgg19(pretrained=True)
vgg19.collect_params().reset_ctx(ctx=ctx1)

x = mx.sym.var('data')
y = vgg19(x)
print('\n=== the symbolic program of net===')
interals = y.get_internals()
print(interals.list_outputs())

vgg19_relu5_4 = g.SymbolBlock([interals['vgg0_conv15_fwd_output']], x, params=vgg19.collect_params())

vgg19_relu5_4.hybridize()

# d_net = g.SymbolBlock([interals['discriminator0_d_dense0_fwd_output']], x, params=d_net_sigm.collect_params())

# vgg19_relu5_4.collect_params().reset_ctx(ctx=ctx)

enhanced = resnet()

enhanced.hybridize()

blur_op = blur()
blur_op.hybridize()

#dont forget about softmax
discrim_predictions_logits = adversarial()
discrim_predictions_logits.hybridize()

enhanced.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)
discrim_predictions_logits.collect_params().initialize(mx.init.Normal(0.02), ctx=ctx)

G_trainer = g.Trainer(enhanced.collect_params(), 'Adam', {'learning_rate': learning_rate})
D_trainer = g.Trainer(discrim_predictions_logits.collect_params(), 'Adam', {'learning_rate': learning_rate})

texture_cross_entropy = TextureLoss()#g.loss.SoftmaxCrossEntropyLoss()
color_cross_entropy = ColorLoss()
content_l2loss = g.loss.L2Loss()
tvx_l2loss = g.loss.L2Loss()
tvy_l2loss = g.loss.L2Loss()

x = mx.sym.var('data')

print('=== input data holder ===')
print(x)

y_enhanced = enhanced(x)

print('\n=== the symbolic program of net===')
print(y_enhanced)
interals = y_enhanced.get_internals()
print(interals.list_outputs())

y = discrim_predictions_logits(x)

print('\n=== the symbolic program of net===')
print(y)
interals = y.get_internals()
print(interals.list_outputs())

for i in range(num_train_iters):
    idx_train = np.random.randint(0, train_size, batch_size)
    phone_img_paths = np.array(train_data)[idx_train]
    dslr_img_paths = np.array(train_answ)[idx_train]

    phone_images = load_transform(phone_img_paths)
    dslr_images = load_transform(dslr_img_paths)

    m_phone_images = phone_images.as_in_context(ctx)
    m_dslr_images = dslr_images.as_in_context(ctx)
    with mx.autograd.record():
        enhanced_images = enhanced(m_phone_images)
        # can not find tf.image.rgb_to_grayscale like.
        # based tf operation https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py
        rgb_weights = mx.nd.array([0.2989, 0.5870, 0.1140]).as_in_context(ctx)
        enhanced_images_gray = mx.nd.transpose(enhanced_images, (0, 2, 3, 1))
        enhanced_images_gray = mx.nd.dot(enhanced_images_gray, rgb_weights).reshape(-1, PATCH_HEIGHT*PATCH_WIDTH)

        dslr_images_gray = mx.nd.transpose(dslr_images, (0, 2, 3, 1)).as_in_context(ctx)
        dslr_images_gray = mx.nd.dot(dslr_images_gray, rgb_weights).reshape(-1, PATCH_HEIGHT*PATCH_WIDTH)

        adv_ = mx.nd.zeros((batch_size, 1)).as_in_context(ctx)
        adversarial_ = enhanced_images_gray * (1 - adv_) + dslr_images_gray * adv_
        adversarial_ = adversarial_.reshape(-1, 1, PATCH_HEIGHT, PATCH_WIDTH)

        discrim_predictions = discrim_predictions_logits(adversarial_)

        #texture loss
        discrim_target = mx.nd.concat(adv_, 1 - adv_, dim=1)

        loss_discrim =  texture_cross_entropy(discrim_predictions, discrim_target)
        loss_texture = -1 * loss_discrim
        #content loss
        enhanced_vgg = vgg19_relu5_4(preprocess_vgginput(enhanced_images.as_in_context(ctx1)))
        dslr_vgg = vgg19_relu5_4(preprocess_vgginput(dslr_images.as_in_context(ctx1)))

        loss_content = 2 * content_l2loss(enhanced_vgg, dslr_vgg) / (6*6*512*batch_size)
        # loss color
        kernel_var = utils.gauss_kernel(21, 3, 3)
        kernel_var = mx.nd.transpose(mx.nd.array(kernel_var), (2, 3, 0, 1))
        # enhanced_images_blur = mx.symbol.Convolution(data=enhanced_images, weight=kernel_var, num_group=3)
        enhanced_images_blur = blur_op(enhanced_images, kernel_var.as_in_context(ctx))
        dlsr_images_blur = blur_op(dslr_images.as_in_context(ctx), kernel_var.as_in_context(ctx))

        loss_color = color_cross_entropy(dlsr_images_blur, enhanced_images_blur, batch_size)
        #total variation loss
        batch_shape = (batch_size, 3, PATCH_WIDTH, PATCH_HEIGHT)

        #TODO: need get size from shape. See tf version
        tv_y_size = 29700
        tv_x_size = 29700
        loss_tvx =  tvx_l2loss(enhanced_images[:,:,:,1:], enhanced_images[:,:,:,:batch_shape[2]-1])
        loss_tvy = tvy_l2loss(enhanced_images[:, :, 1:, :], enhanced_images[:, :, :batch_shape[2] - 1, :])
        loss_tv = 2 * (loss_tvx/tv_x_size + loss_tvy/tv_y_size) / batch_size

        loss_generator = (
        w_content * loss_content + w_texture * loss_texture +  w_color * loss_color + w_tv * loss_tv )
    loss_generator.backward()
    G_trainer.step(batch_size)

    with mx.autograd.record():
        enhanced_images = enhanced(m_phone_images)
        # can not find tf.image.rgb_to_grayscale like.
        # based tf operation https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/python/ops/image_ops_impl.py
        rgb_weights = mx.nd.array([0.2989, 0.5870, 0.1140]).as_in_context(ctx)
        enhanced_images_gray = mx.nd.transpose(enhanced_images, (0, 2, 3, 1))
        enhanced_images_gray = mx.nd.dot(enhanced_images_gray, rgb_weights).reshape(-1, PATCH_HEIGHT*PATCH_WIDTH)

        dslr_images_gray = mx.nd.transpose(dslr_images, (0, 2, 3, 1)).as_in_context(ctx)
        dslr_images_gray = mx.nd.dot(dslr_images_gray, rgb_weights).reshape(-1, PATCH_HEIGHT*PATCH_WIDTH)

        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])
        adv_ = mx.nd.array(swaps).as_in_context(ctx)
        adversarial_ = enhanced_images_gray * (1 - adv_) + dslr_images_gray * adv_
        adversarial_ = adversarial_.reshape(-1, 1, PATCH_HEIGHT, PATCH_WIDTH)

        discrim_predictions = discrim_predictions_logits(adversarial_)

        #texture loss
        discrim_target = mx.nd.concat(adv_, 1 - adv_, dim=1)
        loss_discrim = texture_cross_entropy(discrim_predictions, discrim_target)

    loss_discrim.backward()
    D_trainer.step(batch_size)

    if i % 100 == 0:
        im = misc.toimage(enhanced_images.asnumpy()[0], cmin=-1.0, cmax=1.0)
        im.save('./samples/enhanced_images.jpg')
        im = misc.toimage(dslr_images.asnumpy()[0], cmin=-1.0, cmax=1.0)
        im.save('./samples/dlsr.jpg')
        im = misc.toimage(phone_images.asnumpy()[0], cmin=-1.0, cmax=1.0)
        im.save('./samples/phone.jpg')

    print('[%d/%d] Loss_G: %.4f Loss_D: %.4f' % (i, num_train_iters,  mx.nd.mean(loss_generator).asscalar(), mx.nd.mean(loss_discrim).asscalar()))
        # # transform both dslr and enhanced images to grayscale
        # enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
        # dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image), [-1, PATCH_WIDTH * PATCH_HEIGHT])

    print(np.array(enhanced_images_gray.asnumpy()).shape)
    print(np.array(dslr_images_gray.asnumpy()).shape)
    print(np.array(adversarial_.asnumpy()).shape)
    print(np.array(discrim_predictions.asnumpy()).shape)
    print(np.array(discrim_target.asnumpy()).shape)

    print(np.array(enhanced_vgg.asnumpy()).shape)
    print(np.array(dslr_vgg.asnumpy()).shape)
    # dsfsd
    arg_names = set(y_enhanced.list_arguments())
    aux_names = set(y_enhanced.list_auxiliary_states())
    arg_dict = {}
    for name, param in enhanced.collect_params().items():
        if name in arg_names:
            arg_dict['arg:%s' % name] = param._reduce()
        else:
            assert name in aux_names
            arg_dict['aux:%s' % name] = param._reduce()

    ndarray.save('./models/dlsr.params', arg_dict)
    # model.collect_params().save('./models/CAE_mxnet.params')
    y_enhanced.save('./models/dlsr.json')


