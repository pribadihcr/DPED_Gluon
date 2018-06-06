import mxnet as mx
import numpy as np

from PIL import Image
from scipy import misc
from collections import namedtuple
import matplotlib.pyplot as plt
import os
import time
from glob import glob
import scipy
import imageio


Batch = namedtuple('Batch', ['data'])

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

_t = {'inference' : Timer()}

import shutil

prefix = './models/dlsr'
shutil.copy(prefix+'.json', prefix+'-symbol.json')
shutil.copy(prefix+'.params', prefix+'-0000.params')
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, 0)
internals = sym.get_internals()
for sym_name in internals.list_outputs():
    print(sym_name)

# fea_symbol = internals[[sym_name for sym_name in internals.list_outputs() if sym_name=='dact5_output'][0]]
# print(fea_symbol)

# dsf
mod = mx.mod.Module(symbol=sym, context=mx.gpu(), label_names=None)
mod.bind(for_training=False, data_shapes=[('data', (1,3,720,1280))],
         label_shapes=mod._label_shapes)
mod.set_params(arg_params, aux_params, allow_missing=False)


files = ['./samples/input.jpg']
alls = 0
cnt = 0
tot=0
# test = image_iter()
# for i, batch in enumerate(test):
for file_ in files:
#     print(file_)
    st = time.time()
    # image =scipy.misc.imread(file_, mode='RGB').astype(np.float)
    # # image_ori = scipy.misc.imresize(image, [256, 256])
    # image = np.array(image_ori) / 127.5 - 1.0

    I = np.asarray(imageio.imread(file_))
    image = np.float16(I) / 255
    imager = np.expand_dims(image, axis=0)

    image = np.array(image).reshape(-1, 720, 1280, 3).transpose(0, 3, 1, 2)  # Batch | channels | H | W
    # ori = np.array(image, dtype='float32')
    while True:
        # start_time = time.time()
        _t['inference'].tic()
        mod.forward(Batch([mx.nd.array(image)]))
        # mod.forward(batch)
        # end_time = time.time() - start_time
        rec = mod.get_outputs()[0].asnumpy()
        _t['inference'].toc()
        # print("elapsed time:", end_time * 1000)
        print('elapsed time: {:.3f}s'.format(_t['inference'].average_time))
        print(np.array(rec).shape)
        rec = np.squeeze(rec, axis=0).transpose(1, 2, 0)

        im = scipy.misc.toimage(rec, cmin=-1.0, cmax=1.0)
        im.save('./samples/enhanced.jpg')






