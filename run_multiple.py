from util import argprun


def runrun(dim_g=128, dim_d=128, critic_iters=5,
        n_gpus=1, normalization_g=True, normalization_d=False,
        batch_size=64, iters=110000, penalty_weight="0.1 1 5 100 500",
        one_sided=False, output_dim=3072, lr=2e-4, data_dir='',
        inception_frequency=1000, conditional=False, acgan=False, log_dir='exp', gpu=0):

    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu)

    from tensorflow.python.client import device_lib
    print("VISIBLE DEVICES {}".format(str(device_lib.list_local_devices())))

    from gan_cifar_resnet import run

    logdir = log_dir
    logdir += ("_lp" if one_sided else "_gp")
    logdir += "_{}"

    expcount = 1

    pws = map(float, penalty_weight.split())

    for pw in pws:
        _logdir = logdir.format(expcount)
        if os.path.exists(_logdir):
            raise Exception("logdir {} exists".format(_logdir))
        else:
            os.makedirs(_logdir)
        run(dim_g=dim_g, dim_d=dim_d, critic_iters=critic_iters, n_gpus=n_gpus,
            normalization_d=normalization_d, normalization_g=normalization_g, batch_size=batch_size,
            iters=iters, penalty_weight=pw, one_sided=one_sided, output_dim=output_dim, lr=lr, data_dir=data_dir,
            inception_frequency=inception_frequency, conditional=conditional, acgan=acgan, log_dir=_logdir)
        expcount += 1


if __name__ == "__main__":
    argprun(runrun)