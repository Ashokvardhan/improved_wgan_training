"""WGAN-GP ResNet for CIFAR-10"""
from __future__ import print_function

import os, sys
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
import tflib.plot
from tflib import fid
import glob

import numpy as np
import tensorflow as tf
import tensorflow.contrib.keras as K
import sklearn.datasets

import time
import functools
import locale
locale.setlocale(locale.LC_ALL, '')
from IPython import embed

from util import argprun

# python run_multiple.py -gpu 1 -iters 100000 -penalty_weight "0.1" -log_dir exp4 -penalty_mode grad -one_sided

# specify one_sided, data_dir and log_dir and penalty_weight (everything else should be default, no conditioning)


class TFModule(object):
    def __call__(self, *x, **kwargs):
        return self.forward(*x, **kwargs)


class ResamplingConv(TFModule):
    def __init__(self, in_planes, out_planes, kernel=3, padding=None, resample=None, bias=True):
        super(ResamplingConv, self).__init__()
        assert(kernel in (0, 1, 3, 5, 7))
        padding = (kernel - 1) // 2 if padding is None else padding
        if kernel == 0:
            self.conv = None
        else:
            self.conv = K.layers.Conv2D(out_planes,
                       kernel_size=kernel, padding="same", use_bias=bias,
                       data_format="channels_first")
        self.resample = resample
        self.resampler = None
        if resample == "up":
            self.resampler = K.layers.UpSampling2D(size=2, data_format="channels_first")
        elif resample == "down":
            self.resampler = K.layers.AveragePooling2D(pool_size=2, data_format="channels_first")

    def forward(self, x):
        if self.resample == "up":
            x = self.resampler(x)
        if self.conv is not None:
            x = self.conv(x)
        if self.resample == "down":
            x = self.resampler(x)
        return x

    @property
    def trainable_weights(self):
        ret = []
        if self.conv is not None:
            ret += self.conv.trainable_weights
        if self.resampler is not None:
            ret += self.resampler.trainable_weights
        return ret

    @property
    def updates(self):
        ret = []
        if self.conv is not None:
            ret += self.conv.updates
        if self.resampler is not None:
            ret += self.resampler.updates
        return ret


class ResBlock(TFModule):
    def __init__(self, inplanes, planes, kernel=3, resample=None, bias=True, batnorm=True):
        super(ResBlock, self).__init__()
        self.conv1 = ResamplingConv(None, planes, kernel,
                                    resample=resample if resample != "down" else None,
                                    bias=bias)
        self.bn1 = K.layers.BatchNormalization(axis=1) if batnorm else None
        self.relu = K.layers.Activation("relu")
        self.conv2 = ResamplingConv(None, planes, kernel,
                                    resample=resample if resample != "up" else None,
                                    bias=bias)
        self.bn2 = K.layers.BatchNormalization(axis=1) if batnorm else None
        self.resample = resample
        self.shortcut = ResamplingConv(None, planes, 0 if (inplanes == planes and resample is None) else 1,
                                       resample=resample,
                                       bias=bias)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out) if self.bn1 is not None else out
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out) if self.bn2 is not None else out

        residual = self.shortcut(residual)

        out += residual
        out = self.relu(out)

        return out

    @property
    def trainable_weights(self):
        ret = self.conv1.trainable_weights + (self.bn1.trainable_weights if self.bn1 is not None else []) \
              + self.conv2.trainable_weights + (self.bn2.trainable_weights if self.bn2 is not None else [])\
              + self.shortcut.trainable_weights
        return ret

    @property
    def updates(self):
        ret = (self.bn1.updates if self.bn1 is not None else []) + (self.bn2.updates if self.bn2 is not None else []) \
              + self.conv1.updates + self.conv2.updates + self.shortcut.updates
        return ret


class Gen(TFModule):
    def __init__(self, dim_g, z_dim=128, batsize=None, outdim=None, normalize=True, **kw):
        super(Gen, self).__init__(**kw)
        self.z_dim = z_dim
        self.layers = [
            K.layers.Lambda(lambda x: tf.expand_dims(tf.expand_dims(x, 2), 3)),
            K.layers.Conv2DTranspose(dim_g, 4, data_format="channels_first")] \
            + ([K.layers.BatchNormalization(axis=1)] if normalize else []) \
            + [K.layers.Activation("relu"),
            ResBlock(dim_g, dim_g, 3, resample="up", batnorm=normalize),
            ResBlock(dim_g, dim_g, 3, resample="up", batnorm=normalize),
            ResBlock(dim_g, dim_g, 3, resample="up", batnorm=normalize),
            K.layers.Conv2D(3, 3, padding="same", data_format="channels_first"),
            K.layers.Activation("tanh"),
            K.layers.Lambda(lambda x: tf.reshape(x, (-1, outdim)))
        ]

    def forward(self, n_samples, labels, noise=None):
        x = noise
        if noise is None:
            x = tf.random_normal([n_samples, self.z_dim])
        for layer in self.layers:
            x = layer(x)
        return x

    def get_model(self):
        output = self()

    @property
    def trainable_weights(self):
        ret = []
        for layer in self.layers:
            ret += layer.trainable_weights
        return ret

    @property
    def updates(self):
        ret = []
        for layer in self.layers:
            ret += layer.updates
        return ret


class Disc(TFModule):
    def __init__(self, dim_d, normalize=False, **kw):
        super(Disc, self).__init__(**kw)
        self.layers = [
            K.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 32, 32])),
            ResBlock(3, dim_d, 3, resample="down", batnorm=normalize),
            ResBlock(dim_d, dim_d, 3, resample="down", batnorm=normalize),
            ResBlock(dim_d, dim_d, 3, resample=None, batnorm=normalize),
            ResBlock(dim_d, dim_d, 3, resample=None, batnorm=normalize),
            K.layers.Lambda(lambda x: tf.reduce_mean(x, [2, 3])),
            K.layers.Dense(1),
            K.layers.Lambda(lambda x: tf.squeeze(x, 1))
        ]

    def forward(self, x, labels):
        for layer in self.layers:
            x = layer(x)
        return x, None

    def get_model(self, input=None):
        output = self(input)
        model = K.models.Model(inputs=input, outputs=output)
        return model

    @property
    def trainable_weights(self):
        ret = []
        for layer in self.layers:
            ret += layer.trainable_weights
        return ret

    @property
    def updates(self):
        ret = []
        for layer in self.layers:
            ret += layer.updates
        return ret


def run(mode="wgan-gp", dim_g=128, dim_d=128, critic_iters=5,
        n_gpus=1, normalization_g=True, normalization_d=False,
        batch_size=64, iters=100000, penalty_weight=10,
        one_sided=False, output_dim=3072, lr=2e-4, data_dir='/srv/denis/tfvision-datasets/cifar-10-batches-py',
        inception_frequency=1000, conditional=False, acgan=False, log_dir='default_log', run_fid=False,
        penalty_mode="grad"):   # grad or pagan or ot

    if mode == "wgan-gp" and penalty_mode == "grad":
        print("WGAN-LP" if one_sided else "WGAN-GP")

    # region start
    # Download CIFAR-10 (Python version) at
    # https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
    # extracted files here!
    lib.plot.logdir = log_dir
    print("log dir set to {}".format(lib.plot.logdir))
    # dump locals() for settings
    with open("{}/settings.dict".format(log_dir), "w") as f:
        loca = locals().copy()
        if penalty_mode != "grad":
            del loca["one_sided"]
        f.write(str(loca))
        print("saved settings: {}".format(loca))

    DATA_DIR = data_dir
    if len(DATA_DIR) == 0:
        raise Exception('Please specify path to data directory in gan_cifar_resnet.py!')

    N_GPUS = n_gpus
    if N_GPUS not in [1,2]:
        raise Exception('Only 1 or 2 GPUs supported!')

    BATCH_SIZE = batch_size # Critic batch size
    GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
    ITERS = iters # How many iterations to train for
    DIM_G = dim_g # Generator dimensionality
    DIM_D = dim_d # Critic dimensionality
    NORMALIZATION_G = normalization_g # Use batchnorm in generator?
    NORMALIZATION_D = normalization_d # Use batchnorm (or layernorm) in critic?
    OUTPUT_DIM = output_dim # Number of pixels in CIFAR10 (32*32*3)
    LR = lr # Initial learning rate
    DECAY = True # Whether to decay LR over learning
    N_CRITIC = critic_iters # Critic steps per generator steps
    INCEPTION_FREQUENCY = inception_frequency # How frequently to calculate Inception score

    CONDITIONAL = conditional # Whether to train a conditional or unconditional model
    ACGAN = acgan # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
    ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
    ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

    ONE_SIDED = one_sided
    LAMBDA = penalty_weight

    if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
        print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

    DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
    if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
        DEVICES = [DEVICES[0], DEVICES[0]]

    lib.print_model_settings(locals().copy())

    if run_fid:
        inception_path = "/tmp/inception"
        print("check for inception model..")
        inception_path = fid.check_or_download_inception(inception_path)  # download inception if necessary
        print("ok")

        print("create inception graph..")
        fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
        print("ok")

        # region cifar FID
        if not os.path.exists("cifar.fid.stats.npz"):  # compute fid stats for CIFAR
            train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)

            # loads all images into memory (this might require a lot of RAM!)
            print("load images..")
            images = []
            for imagebatch, _ in dev_gen():
                images.append(imagebatch)

            _use_train_for_stats = True
            if _use_train_for_stats:
                print("using train data for FID stats")
                for imagebatch, _ in train_gen():
                    images.append(imagebatch)

            allimages = np.concatenate(images, axis=0)
            # allimages = ((allimages+1.)*(255.99/2)).astype('int32')
            allimages = allimages.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
            # images = list(allimages)
            images = allimages
            print("%d images found and loaded: {}" % len(images), images.shape)

            # embed()

            print("calculate FID stats..")
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100, verbose=True)
                np.savez_compressed("cifar.fid.stats", mu=mu, sigma=sigma)
            print("finished")

            # embed()

        f = np.load("cifar.fid.stats.npz")
        mu_real, sigma_real = f['mu'][:], f['sigma'][:]

    else:
        print("  NOT RUNNING FID  !!! -- ()()()")

    # embed()

    import tflib.inception_score

    # endregion

    # region model

    def nonlinearity(x):
        return tf.nn.relu(x)

    def Normalize(name, inputs, labels=None):
        """This is messy, but basically it chooses between batchnorm, layernorm,
        their conditional variants, or nothing, depending on the value of `name` and
        the global hyperparam flags."""
        if not CONDITIONAL:
            labels = None
        if CONDITIONAL and ACGAN and ('Discriminator' in name):
            labels = None

        if ('Discriminator' in name) and NORMALIZATION_D:
            assert(False)
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
        elif ('Generator' in name) and NORMALIZATION_G:
            if labels is not None:
                assert(False)
                return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
            else:
                return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
        else:
            return inputs

    def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        return output

    def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0,3,1,2])
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
        """
        resample: None, 'down', or 'up'
        """
        if resample=='down':
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = ConvMeanPool
        elif resample=='up':
            conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_shortcut = UpsampleConv
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        elif resample==None:
            conv_shortcut = lib.ops.conv2d.Conv2D
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim==input_dim and resample==None:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = Normalize(name+'.N1', output, labels=labels)
        output = nonlinearity(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
        output = Normalize(name+'.N2', output, labels=labels)
        output = nonlinearity(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

        return shortcut + output

    def OptimizedResBlockDisc1(inputs):
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
        conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
        conv_shortcut = MeanPoolConv
        shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
        output = nonlinearity(output)
        output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
        return shortcut + output

    def Generator(n_samples, labels, noise=None):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])
        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*DIM_G, noise)
        output = tf.reshape(output, [-1, DIM_G, 4, 4])
        output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
        output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
        output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
        output = Normalize('Generator.OutputN', output)
        output = nonlinearity(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
        output = tf.tanh(output)
        return tf.reshape(output, [-1, OUTPUT_DIM])

    def Discriminator(inputs, labels):
        output = tf.reshape(inputs, [-1, 3, 32, 32])
        output = OptimizedResBlockDisc1(oudisc_update_opstput)
        output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
        output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
        output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
        output = nonlinearity(output)
        output = tf.reduce_mean(output, axis=[2,3])
        output_wgan = lib.ops.linear.Linear('Discriminator.Output', DIM_D, 1, output)
        output_wgan = tf.reshape(output_wgan, [-1])
        if CONDITIONAL and ACGAN:
            output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
            return output_wgan, output_acgan
        else:
            return output_wgan, None

    # endregion

    with tf.Session() as session:

        # discriminator = Disc(DIM_D).get_model(K.layers.Input(shape=(OUTPUT_DIM,)))
        # generator = Gen(DIM_G).get_model()

        K.backend.set_learning_phase(True)

        discriminator = Disc(DIM_D, normalize=normalization_d)
        generator = Gen(DIM_G, outdim=OUTPUT_DIM, normalize=normalization_g)
        #
        # discriminator = Discriminator
        # generator = Generator

        _iteration_gan = tf.placeholder(tf.int32, shape=None)
        all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

        all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        DEVICES_B = DEVICES[:len(DEVICES)/2]
        DEVICES_A = DEVICES[len(DEVICES)/2:]

        disc_costs = []     # total disc cost
        # for separate logging
        scorediff_acc = []
        gps_all_acc = []
        gps_pos_acc = []
        gps_neg_acc = []

        disc_acgan_costs = []
        disc_acgan_accs = []
        disc_acgan_fake_accs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    all_real_data_splits[len(DEVICES_A)+i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A)+i]
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i]
                ], axis=0)
                disc_all, disc_all_acgan = discriminator(real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:BATCH_SIZE/len(DEVICES_A)]
                disc_fake = disc_all[BATCH_SIZE/len(DEVICES_A):]
                disc_costs.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))
                scorediff_acc.append(tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real))

                if CONDITIONAL and ACGAN:
                    disc_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], labels=real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)])
                    ))
                    disc_acgan_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[:BATCH_SIZE/len(DEVICES_A)], dimension=1)),
                                real_and_fake_labels[:BATCH_SIZE/len(DEVICES_A)]
                            ),
                            tf.float32
                        )
                    ))
                    disc_acgan_fake_accs.append(tf.reduce_mean(
                        tf.cast(
                            tf.equal(
                                tf.to_int32(tf.argmax(disc_all_acgan[BATCH_SIZE/len(DEVICES_A):], dimension=1)),
                                real_and_fake_labels[BATCH_SIZE/len(DEVICES_A):]
                            ),
                            tf.float32
                        )
                    ))

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
                fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
                labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i],
                ], axis=0)
                if penalty_mode == "grad":
                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE/len(DEVICES_A),1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha*differences)
                    gradients = tf.gradients(discriminator(interpolates, labels)[0], [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    if ONE_SIDED is True:
                        gradient_penalty = LAMBDA *tf.reduce_mean(tf.clip_by_value(slopes - 1., 0, np.infty)**2)
                        gp_pos = gradient_penalty * 1.  #LAMBDA *tf.reduce_mean(tf.clip_by_value(slopes - 1., 0, np.infty)**2)
                        gp_neg = tf.constant(0.)
                    else:
                        gradient_penalty = LAMBDA *tf.reduce_mean((slopes-1.)**2)
                        gp_pos = LAMBDA *tf.reduce_mean(tf.clip_by_value(slopes - 1., 0, np.infty)**2)
                        gp_neg = LAMBDA *tf.reduce_mean(tf.clip_by_value(slopes - 1., -np.infty, 0)**2)
                elif penalty_mode == "pagan" or penalty_mode == "ot":
                    _EPS = 1e-6
                    _INTERP = False
                    print("SHAPES OF REAL AND FAKE DATA FOR PAGAN AND OT: ", real_data.get_shape(), fake_data.get_shape())
                    print("TYPES: ", type(real_data), type(fake_data))
                    if _INTERP:     # ONLY EXP_PAGANY_1 is _INTERP !!! ???
                        _alpha = tf.random_uniform(
                            shape=[BATCH_SIZE/len(DEVICES_A),1],
                            minval=0.,
                            maxval=1.
                        )
                        _alpha2 = tf.random_uniform(
                            shape=[BATCH_SIZE/len(DEVICES_A),1],
                            minval=0.,
                            maxval=1.
                        )
                        alpha = tf.minimum(_alpha, _alpha2)
                        alpha2 = tf.maximum(_alpha, _alpha2)
                        differences = fake_data - real_data
                        interp_real = real_data + (alpha * differences)         # points closer to real
                        interp_fake = real_data + (alpha2 * differences)        # points closer to fake
                    else:
                        interp_real = real_data
                        interp_fake = fake_data
                    _D_real, _ = discriminator(interp_real, labels)   # TODO: check make sure labels don't have influence
                    _D_fake, _ = discriminator(interp_fake, labels)
                    real_fake_dist = tf.norm(interp_real - interp_fake, ord=2, axis=1)     # data are vectors: (64, 3072) from get_shape()
                    print("SCORES AND DIST SHAPES: ", _D_real.get_shape(), _D_fake.get_shape(), real_fake_dist.get_shape())
                    print("TYPES: ", type(_D_real), type(_D_fake), type(real_fake_dist))

                    if penalty_mode == "pagan":
                        penalty_vecs = tf.clip_by_value(
                                            (tf.abs(_D_real - _D_fake)
                                                / tf.clip_by_value(real_fake_dist, _EPS, np.infty))
                                            - 1,
                                        0, np.infty) ** 2
                    elif penalty_mode == "ot":
                        penalty_vecs = tf.clip_by_value(
                                            _D_real - _D_fake - real_fake_dist,
                                        0, np.infty) ** 2
                    print("PENALTY VEC SHAPE: ", penalty_vecs.get_shape())
                    gradient_penalty = LAMBDA * tf.reduce_mean(penalty_vecs)
                    gp_pos = tf.constant(0.)
                    gp_neg = tf.constant(0.)
                else:
                    raise Exception("unknown penalty mode {}".format(penalty_mode))
                disc_costs.append(gradient_penalty)
                gps_all_acc.append(gradient_penalty)
                gps_pos_acc.append(gp_pos)
                gps_neg_acc.append(gp_neg)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)
        # for more logging
        scorediff = tf.add_n(scorediff_acc) / len(DEVICES_A)
        gps_all = tf.add_n(gps_all_acc) / len(DEVICES_B)
        gps_pos = tf.add_n(gps_pos_acc) / len(DEVICES_B)
        gps_neg = tf.add_n(gps_neg_acc) / len(DEVICES_B)

        if CONDITIONAL and ACGAN:
            disc_acgan = tf.add_n(disc_acgan_costs) / len(DEVICES_A)
            disc_acgan_acc = tf.add_n(disc_acgan_accs) / len(DEVICES_A)
            disc_acgan_fake_acc = tf.add_n(disc_acgan_fake_accs) / len(DEVICES_A)
            disc_cost = disc_wgan + (ACGAN_SCALE*disc_acgan)
        else:
            disc_acgan = tf.constant(0.)
            disc_acgan_acc = tf.constant(0.)
            disc_acgan_fake_acc = tf.constant(0.)
            disc_cost = disc_wgan

        if hasattr(discriminator, "trainable_weights"):
            disc_params = discriminator.trainable_weights       # keras mode
        else:
            disc_params = lib.params_with_name('Discriminator.')    # original

        disc_update_ops = []
        if hasattr(discriminator, "updates"):
            disc_update_ops = discriminator.updates

        if DECAY:
            decay = tf.maximum(0., 1.-(tf.cast(_iteration_gan, tf.float32)/ITERS))
        else:
            decay = 1.

        gen_costs = []
        gen_acgan_costs = []
        for device in DEVICES:
            with tf.device(device):
                n_samples = GEN_BS_MULTIPLE * BATCH_SIZE / len(DEVICES)
                fake_labels = tf.cast(tf.random_uniform([n_samples])*10, tf.int32)
                if CONDITIONAL and ACGAN:
                    disc_fake, disc_fake_acgan = Discriminator(Generator(n_samples,fake_labels), fake_labels)
                    gen_costs.append(-tf.reduce_mean(disc_fake))
                    gen_acgan_costs.append(tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=disc_fake_acgan, labels=fake_labels)
                    ))
                else:
                    gen_costs.append(-tf.reduce_mean(discriminator(generator(n_samples, fake_labels), fake_labels)[0]))
        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))
        if CONDITIONAL and ACGAN:
            gen_cost += (ACGAN_SCALE_G*(tf.add_n(gen_acgan_costs) / len(DEVICES)))

        if hasattr(generator, "trainable_weights"):
            gen_params = generator.trainable_weights            # keras mode
        else:
            gen_params = lib.params_with_name('Generator')      # original

        gen_update_ops = []
        if hasattr(generator, "updates"):
            gen_update_ops = generator.updates

        with tf.control_dependencies(gen_update_ops):
            gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
            gen_gv = gen_opt.compute_gradients(gen_cost, var_list=gen_params)
            gen_train_op = gen_opt.apply_gradients(gen_gv)

        with tf.control_dependencies(disc_update_ops):
            disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
            disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
            disc_train_op = disc_opt.apply_gradients(disc_gv)

        gen_train_ops = [gen_train_op]# + gen_update_ops
        disc_train_ops = [disc_train_op]# + disc_update_ops

        # K.backend.set_learning_phase(False)

        # Function for generating samples
        frame_i = [0]
        fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
        fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
        fixed_noise_samples = generator(100, fixed_labels, noise=fixed_noise)

        def generate_image(frame, true_dist):
            samples = session.run(fixed_noise_samples)
            samples = ((samples+1.)*(255./2)).astype('int32')
            lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), '{}/samples_{}.png'.format(log_dir, frame))

        # Function for calculating inception score
        fake_labels_100 = tf.cast(tf.random_uniform([100])*10, tf.int32)
        samples_100 = generator(100, fake_labels_100)

        def get_IS_and_FID(n):
            all_samples = []
            for i in xrange(n/100):
                all_samples.append(session.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = ((all_samples+1.)*(255.99/2)).astype('int32')
            all_samples = all_samples.reshape((-1, 3, 32, 32)).transpose(0,2,3,1)
            # print("getting IS and FID")
            # print(_iteration_gan)
            # embed()
            # with tf.Session() as _fid_session:
            #     _inception_score, _fid_score = fid.calc_IS_and_FID(all_samples, (mu_real, sigma_real), 100, verbose=True, session=_fid_session)
            # _inception, _inception_std = _inception_score
            _fid_score = 0
            _inception_score = lib.inception_score.get_inception_score(list(all_samples))
            print("calculated IS")
            # print(_inception_score, _inception_score_check)
            # embed()
            # assert(_inception_score[0] == _inception_score_check[0])
            # print("IS calculation same as old")
            if run_fid:
                mu_gen, sigma_gen = fid.calculate_activation_statistics(all_samples, session, 100, verbose=True)
                try:
                    _fid_score = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
                except Exception as e:
                    print(e)
                    _fid_score = 10e4
                print("calculated IS and FID")
                return _inception_score, _fid_score
            else:
                return _inception_score, 0

        train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)

        def inf_train_gen():
            while True:
                for images,_labels in train_gen():
                    yield images,_labels


        for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print ("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print ("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print ("\t{} ({})".format(v.name, shape_str))
            print ("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))

        session.run(tf.initialize_all_variables())

        gen = inf_train_gen()

        for iteration in range(ITERS):
            # TRAINING
            start_time = time.time()

            # embed()

            _gen_cost = 0
            if iteration > 0:
                _gen_cost, _ = session.run([gen_cost, gen_train_ops], feed_dict={_iteration_gan: iteration})

            for i in range(N_CRITIC):
                _data,_labels = gen.next()
                if CONDITIONAL and ACGAN:
                    assert(False)
                    _disc_cost, _disc_wgan, _disc_acgan, _disc_acgan_acc, _disc_acgan_fake_acc, _ \
                        = session.run([disc_cost, disc_wgan, disc_acgan, disc_acgan_acc, disc_acgan_fake_acc, disc_train_op],
                            feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration_gan:iteration})
                else:
                    _disc_cost, _scorediff, _gps_all, _gps_pos, _gps_neg, _ \
                        = session.run(
                    [disc_cost,  scorediff,  gps_all,  gps_pos,  gps_neg,  disc_train_ops],
                        feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration_gan:iteration})

            lib.plot.plot('cost', _disc_cost)
            # extra plot
            lib.plot.plot('core', _scorediff)
            lib.plot.plot('gp', _gps_all)
            lib.plot.plot('gp_pos', _gps_pos)
            lib.plot.plot('gp_neg', _gps_neg)
            lib.plot.plot('gen_cost', _gen_cost)
            if CONDITIONAL and ACGAN:
                lib.plot.plot('wgan', _disc_wgan)
                lib.plot.plot('acgan', _disc_acgan)
                lib.plot.plot('acc_real', _disc_acgan_acc)
                lib.plot.plot('acc_fake', _disc_acgan_fake_acc)
            lib.plot.plot('time', time.time() - start_time)

            if iteration % INCEPTION_FREQUENCY == INCEPTION_FREQUENCY-1:
                inception_score, fid_score = get_IS_and_FID(50000)
                lib.plot.plot('inception', inception_score[0])
                lib.plot.plot('inception_std', inception_score[1])      # std of inception over 10 splits
                lib.plot.plot('fid', fid_score)

            # Calculate dev loss and generate samples every 100 iters
            # VALIDATION
            if iteration % 100 == 99:
                dev_disc_costs = []
                dev_scorediff = []
                dev_gp_all = []
                dev_gp_pos = []
                dev_gp_neg = []
                dev_gen_costs = []
                for images,_labels in dev_gen():
                    _dev_disc_cost, _dev_scorediff, _dev_gps_all, _dev_gps_pos, _dev_gps_neg, _dev_gen_cost \
                        = session.run(
                    [disc_cost,          scorediff,      gps_all,      gps_pos,      gps_neg,      gen_cost],
                        feed_dict={all_real_data_int: images,all_real_labels:_labels})

                    dev_disc_costs.append(_dev_disc_cost)
                    dev_scorediff.append(_dev_scorediff)
                    dev_gp_all.append(_dev_gps_all)
                    dev_gp_pos.append(_dev_gps_pos)
                    dev_gp_neg.append(_dev_gps_neg)
                    dev_gen_costs.append(_dev_gen_cost)

                lib.plot.plot('dev_cost', np.mean(dev_disc_costs))
                lib.plot.plot('dev_core', np.mean(dev_scorediff))
                lib.plot.plot('dev_gp', np.mean(dev_gp_all))
                lib.plot.plot('dev_gp_pos', np.mean(dev_gp_pos))
                lib.plot.plot('dev_gp_neg', np.mean(dev_gp_neg))
                lib.plot.plot('dev_gen_cost', np.mean(dev_gen_costs))

            if iteration % 500 == 0:
                generate_image(iteration, _data)

            if (iteration < 500) or (iteration % 1000 == 999):
                lib.plot.flush()

            lib.plot.tick()


if __name__ == "__main__":
    argprun(run)
