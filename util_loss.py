from mxnet import gluon as g
import mxnet as mx
from mxnet import ndarray

def _reshape_like(F, x, y):
    return x.reshape(y.shape) if F is ndarray else F.reshape_like(x, y)
class TextureLoss(g.nn.HybridBlock):
    def __init__(self, axis=-1, batch_axis=0, **kwargs):
        super(TextureLoss, self).__init__()
        self._axis = axis

    def hybrid_forward(self, F, pred, label):
        pred = F.log(pred)
        label = _reshape_like(F, label, pred)
        loss = -F.sum(pred*label, axis=self._axis, keepdims=True)
        return loss

class ColorLoss(g.nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ColorLoss, self).__init__()

    def hybrid_forward(self, F, pred, label, bath_size):
        # loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)
        loss = F.sum((label - pred).__pow__(2))/(2*bath_size)
        return loss




