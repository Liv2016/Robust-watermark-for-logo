import tensorflow as tf
import os
import glob
import util
import models
import cv2
from os.path import join
import numpy as np
import time


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='watermark1', help="Experiment name")
    parser.add_argument('--secret_len', type=int, default=100, help="Watermark information length")
    parser.add_argument('--cover_h', type=int, default=400, help="Height of carrier image")
    parser.add_argument('--cover_w', type=int, default=400, help="Width of carrier image")
    parser.add_argument('--num_epochs', type=int, default=2, help="Epoch of train")
    parser.add_argument('--num_steps', type=int, default=140000, help="total steps")
    parser.add_argument('--batch_size', type=int, default=2, help="batch size")
    parser.add_argument('--lr', type=float, default=.0001, help="learning rate")
    parser.add_argument('--Delr', type=float, default=.0001, help="Denoiser learning rate")
    parser.add_argument('--is_train', type=bool, default=True)  #
    parser.add_argument('--dataset_path', type=str, default="F:\\myproject\\Dataset\\test",
                        help="dataset path of train")
    # parser.add_argument('--logs_path', type=str, default="./logs/")
    # parser.add_argument('--checkpoint_path', type=str, default='./checkpoints/')
    # parser.add_argument('--model_save_path', type=str, default='./models')

    parser.add_argument('--cover_lpips_ratio', type=float, default=3, help="ratio of lpips loss")
    parser.add_argument('--cover_lpips_step', type=int, default=15000)  #
    parser.add_argument('--cover_mse_ratio', type=float, default=3, help="ratio of mse loss")
    parser.add_argument('--cover_mse_step', type=int, default=15000)  #

    parser.add_argument('--wm_mse_ratio', type=float, default=3.0)  #
    parser.add_argument('--wm_mse_step', type=int, default=1)  #
    parser.add_argument('--wm_lpips_ratio', type=float, default=3.0)  #
    parser.add_argument('--wm_lpips_step', type=int, default=1)  #

    parser.add_argument('--use_second', type=bool, default=False)  #
    parser.add_argument('--gauss_stddev', type=float, default=.02)  #
    parser.add_argument('--is_in_warp', type=bool, default=False)  #
    parser.add_argument('--max_warp', type=float, default=.1)  #
    parser.add_argument('--max_bri', type=float, default=.3)  #
    parser.add_argument('--rnd_sat', type=float, default=1.0)  #
    parser.add_argument('--max_hue', type=float, default=.1)  #
    parser.add_argument('--cts_low', type=float, default=.5)  #
    parser.add_argument('--cts_high', type=float, default=1.5)  #

    parser.add_argument('--warp_step', type=int, default=10000)  #
    parser.add_argument('--bri_step', type=int, default=1000)  #
    parser.add_argument('--sat_step', type=int, default=1000)  #
    parser.add_argument('--hue_step', type=int, default=1000)  #
    parser.add_argument('--gaussian_step', type=int, default=1000)  #
    parser.add_argument('--cts_step', type=int, default=1000)  #

    parser.add_argument('--only_secret_N', help="The first N steps only optimize secret loss", type=int, default=2500)
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('--start_step', type=int, default=0)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--start_epoch_step', type=int, default=0)
    parser.add_argument('--mse_gain', type=float, default=10.0)
    parser.add_argument('--mse_gain_epoch', type=int, default=25)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    dataset_path = args.dataset_path
    file_names = glob.glob(dataset_path + '/*')
    # print("#######################")
    # print(file_names)
    file_len = len(file_names)
    total_step = int(file_len / args.batch_size)

    # place_holder
    global_index_tensor = tf.Variable(0, trainable=False, name='global_step')
    TFM_pl = tf.placeholder(shape=[None, 2, 8], dtype=tf.float32, name="warp_matrix")
    loss_ratio_pl = tf.placeholder(shape=[4], dtype=tf.float32, name="loss_ratio")

    # build graph
    image_batch = util.build_dataset(file_names, batch_size=args.batch_size, epoch=args.num_epochs + 1,
                                     H=args.cover_h, W=args.cover_w,
                                     secret_size=args.secret_len)
    Encoder = models.WatermarkEncoder(height=args.cover_h, width=args.cover_h)

    Deconder = models.WatermarkDecoder(height=args.cover_h, width=args.cover_w, base_num=32)

    Denoiser = models.WatermarkDenoiser()

    loss_total, loss_watermark, config_op, image_summary_op, cover_psnr, wm_psnr, loss_Denoiser = models.make_graph(
        Encoder, Deconder,
        image_batch,
        loss_ratio_pl,
        args,
        TFM_pl,
        global_index_tensor,
        Denoiser)

    variables = tf.trainable_variables()
    denosier_vars=[var for var in variables if 'denoiser' in var.name]
    Denoiser_optimizer = tf.train.AdamOptimizer(args.Delr).minimize(loss_Denoiser, var_list=denosier_vars)
    total_optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss_total, var_list=variables,
                                                               global_step=global_index_tensor)
    secret_loss_optimizer = tf.train.AdamOptimizer(args.lr).minimize(loss_watermark, var_list=variables,
                                                                     global_step=global_index_tensor)

    watermark_pl = tf.placeholder(shape=[None, args.cover_h, args.cover_w, 3], dtype=tf.float32, name="secret")
    cover_pl = tf.placeholder(shape=[None, args.cover_h, args.cover_w, 3], dtype=tf.float32, name="cover")
    stego = models.make_encode_graph(Encoder, watermark_pl, cover_pl)
    pre_watermark = models.make_decode_graph(Deconder, Denoiser, cover_pl)

    saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=100, keep_checkpoint_every_n_hours=5)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    base_path = './run/' + args.exp_name
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    with open(base_path + '/' + 'config.txt', "a") as file:
        file.write("#####################################" + "\n")
        str_time = time.strftime("%Y.%m.%d %H:%M:%S", time.localtime())
        file.write("run time: " + str_time + '\n')
        for arg in vars(args):
            para = "{: <25} {: <25}".format(str(arg) + ':', str(getattr(args, arg))) + '\n'
            file.write(para)
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        writer = tf.summary.FileWriter(join(base_path, 'logs'), sess.graph)

        if args.pretrained is not None:
            saver.restore(sess, args.pretrained)
        global_index = 0
        i = 0
        if args.start_step != 0:
            sess.run(tf.assign(global_index_tensor, args.start_step))
        if args.start_epoch != 0:
            i = args.start_epoch
        while i < args.num_epochs:
            i += 1
            j = 0
            while j < total_step:
                j += 1
                cover_mse_ratio = min(args.cover_mse_ratio * global_index / args.cover_mse_step, args.cover_mse_ratio)
                cover_lpips_ratio = min(args.cover_lpips_ratio * global_index / args.cover_lpips_step,
                                        args.cover_lpips_ratio)
                wm_mse_ratio = min(args.wm_mse_ratio * global_index / args.wm_mse_step,
                                   args.wm_mse_ratio)
                wm_lpips_ratio = min(args.wm_lpips_ratio * global_index / args.wm_lpips_step,
                                     args.wm_lpips_ratio)
                if i >= (args.num_epochs - args.mse_gain_epoch):
                    cover_mse_ratio *= args.mse_gain
                    wm_mse_ratio *= args.mse_gain
                shift_ratio = min(args.max_warp * global_index / args.warp_step, args.max_warp)
                shift_ratio = np.random.uniform() * shift_ratio
                TFM = util.get_transform_matrix(args.cover_h, np.floor(args.cover_h * shift_ratio),
                                                args.batch_size // 2)
                feed_dict = {TFM_pl: TFM,
                             loss_ratio_pl: [cover_mse_ratio, cover_lpips_ratio, wm_mse_ratio, wm_lpips_ratio]}

                sess.run(Denoiser_optimizer, feed_dict=feed_dict)
                if global_index < args.only_secret_N:
                    _, loss_np, global_index, config_np, loss_watermark_np, co_psnr_np, wm_psnr_np = sess.run(
                        [secret_loss_optimizer, loss_total, global_index_tensor, config_op, loss_watermark, cover_psnr,
                         wm_psnr],
                        feed_dict=feed_dict)
                else:
                    _, loss_np, global_index, config_np, loss_watermark_np, co_psnr_np, wm_psnr_np = sess.run(
                        [total_optimizer, loss_total, global_index_tensor, config_op, loss_watermark, cover_psnr,
                         wm_psnr],
                        feed_dict=feed_dict)

                if global_index % 100 == 0:
                    print("###############################################################")
                    print("Epoch: {}                    step: {}".format(i, j + 1))
                    print("total loss:{:.5f}            watermark_loss:{:.5f}".format(loss_np, loss_watermark_np))
                    print("cover-stego PSNR:{:.5f}      watermark PSNR:{:.5f}".format(co_psnr_np, wm_psnr_np))
                if global_index % 200 == 0:
                    writer.add_summary(config_np, global_index)
                    warp_scale = tf.Summary(
                        value=[tf.Summary.Value(tag='nosie_config/warp_shift_ratio', simple_value=shift_ratio)])
                    writer.add_summary(warp_scale, global_index)
                if global_index % 1000 == 0:
                    image_summary, global_index = sess.run([image_summary_op, global_index_tensor], feed_dict)
                    writer.add_summary(image_summary, global_index)
                if global_index % 20000 == 0:
                    saver.save(sess, join(base_path, 'checkpoints/') + args.exp_name + ".chkp",
                               global_step=global_index)

        tf.saved_model.simple_save(sess,
                                   join(base_path, 'model_save') + 'model' + time.strftime("%Y%m%d_%H%M%S",
                                                                                           time.localtime()),
                                   inputs={'watermark': watermark_pl, 'cover': cover_pl},
                                   outputs={'stego': stego, 'pre_watermark': pre_watermark})
        coord.request_stop()
        coord.join(threads)
    writer.close()


if __name__ == '__main__':
    main()
