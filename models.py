import numpy as np
import tensorflow as tf
from stn import spatial_transformer_network as stn_transformer
# from tensorflow.python.keras.models import *
# from tensorflow.python.keras.layers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import util
import lpips.lpips_tf as lpips_tf

class Encoder(Layer):
    def __init__(self, height, width):
        super(Encoder, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        # watermark, cover = inputs
        # watermark = watermark - .5
        # cover = cover - .5
        # inputs = concatenate([watermark, cover], axis=-1)
        # batch x 400 x 400 x 6
        conv1 = self.conv1(inputs)
        # batch x 400 x 400 x 32
        conv2 = self.conv2(conv1)
        # batch x 400 x 400 x 32
        conv3 = self.conv3(conv2)
        # batch x 400 x 400 x 64
        conv4 = self.conv4(conv3)
        # batch x 400 x 400 x 128
        conv5 = self.conv5(conv4)
        # batch x 400 x 400 x 256
        up6 = self.up6(UpSampling2D(size=(2, 2))(conv5))
        # batch x 800 x 800 x 128
        merge6 = concatenate([conv4, up6], axis=3)
        conv6 = self.conv6(merge6)
        up7 = self.up7(UpSampling2D(size=(2, 2))(conv6))
        merge7 = concatenate([conv3, up7], axis=3)
        conv7 = self.conv7(merge7)
        up8 = self.up8(UpSampling2D(size=(2, 2))(conv7))
        merge8 = concatenate([conv2, up8], axis=3)
        conv8 = self.conv8(merge8)
        up9 = self.up9(UpSampling2D(size=(2, 2))(conv8))
        merge9 = concatenate([conv1, up9, inputs], axis=3)
        conv9 = self.conv9(merge9)
        residual = self.residual(conv9)
        return residual

class Encoder2(Layer):
    def __init__(self, height, width):
        super(Encoder2, self).__init__()
        self.secret_dense = Dense(7500, activation='relu', kernel_initializer='he_normal')
        self.conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(32, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv3 = Conv2D(64, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv4 = Conv2D(128, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.conv5 = Conv2D(256, 3, activation='relu', strides=2, padding='same', kernel_initializer='he_normal')
        self.up6 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv6 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up7 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv7 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up8 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv8 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up9 = Conv2D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.residual = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5

        secret = self.secret_dense(secret)
        secret = Reshape((50, 50, 3))(secret)
        # secret = batch x 50 x 50 x 3
        secret_enlarged = UpSampling2D(size=(8, 8))(secret)
        # print(secret_enlarged.shape)
        # batch x 400 x 400 x 3
        inputs = concatenate([secret_enlarged, image], axis=-1)
        # print(inputs.shape)
        # batch x 400 x 400 x 6
        conv1 = self.conv1(inputs)
        # print(conv1.shape)
        # batch x 400 x 400 x 32
        conv2 = self.conv2(conv1)
        # print(conv2.shape)
        # batch x 200 x 200 x 32
        conv3 = self.conv3(conv2)
        # print(conv3.shape)
        # batch x 100 x 100 x 64
        conv4 = self.conv4(conv3)
        # print(conv4.shape)
        # batch x 50 x 50 x 128
        conv5 = self.conv5(conv4)
        # print(conv5.shape)
        # batch x 25 x 25 x 256
        up6 = self.up6(UpSampling2D(size=(2, 2))(conv5))
        # print(up6.shape)
        # batch x 50 x 50 x 128
        merge6 = concatenate([conv4, up6], axis=3)
        # print(merge6.shape)
        # batch x 50 x 50 x 256
        conv6 = self.conv6(merge6)
        # print(conv6.shape)
        # batch x 50 x 50 x 128
        up7 = self.up7(UpSampling2D(size=(2, 2))(conv6))
        # print(up7.shape)
        # batch x 100 x 100 x 64
        merge7 = concatenate([conv3, up7], axis=3)
        # print(merge7.shape)
        # batch x 100 x 100 x 128
        conv7 = self.conv7(merge7)
        # print(conv7.shape)
        # batch x 100 x 100 x 64
        up8 = self.up8(UpSampling2D(size=(2, 2))(conv7))
        # print(up8.shape)
        # batch x 200 x 200 x 32
        merge8 = concatenate([conv2, up8], axis=3)
        # print(merge8.shape)
        # batch x 200 x 200 x 64
        conv8 = self.conv8(merge8)
        # print(conv8.shape)
        # batch x 200 x 200 x 32
        up9 = self.up9(UpSampling2D(size=(2, 2))(conv8))
        # print(up9.shape)
        # batch x 400 x 400 x 32
        merge9 = concatenate([conv1, up9, inputs], axis=3)
        # print(merge9.shape)
        # batch x 400 x 400 x 70
        conv9 = self.conv9(merge9)
        # print(conv9.shape)
        # batch x 400 x 400 x 32
        residual = self.residual(conv9)
        # print(residual.shape)
        # batch x 400 x 400 x 3
        return residual


class Uplusplus(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(Uplusplus, self).__init__()
        self.hidden = Conv2D(base_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down00 = Conv2D(base_num * 2, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down10 = Conv2D(base_num * 4, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down20 = Conv2D(base_num * 8, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down30 = Conv2D(base_num * 16, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half

        # up
        self.up10 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up20 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up11 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up30 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up21 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up12 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up40 = Conv2D(base_num * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up31 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up22 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up13 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')

    def call(self, image):
        # v begin
        x00 = self.hidden(image)
        # print("x00:", x00.shape)
        # batch x 512 x 512 x 32

        x10 = self.down00(x00)
        # print("x10:", x10.shape)
        # batch x 256 x 256 x 64

        up01 = UpSampling2D(size=(2, 2))(x10)
        merge_00_01 = concatenate([x00, up01], axis=-1)
        x01 = self.up10(merge_00_01)
        # print("x01", x01.shape)
        # batch x 512 x 512 x 32
        # v end

        # v begin
        x20 = self.down10(x10)
        # print("x20:", x20.shape)
        # batch x 128 x 128 x 128
        up11 = UpSampling2D(size=(2, 2))(x20)
        merge_10_11 = concatenate([x10, up11], axis=-1)
        x11 = self.up20(merge_10_11)
        # print("x11:", x11.shape)
        # batch x 256 x 256 x 64

        up02 = UpSampling2D(size=(2, 2))(x11)
        merge_00_01_02 = concatenate([x00, x01, up02], axis=-1)
        x02 = self.up11(merge_00_01_02)
        # print("x02:", x02.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x30 = self.down20(x20)
        # print("x30:", x30.shape)
        # batch x 64 x 64 x 256
        up21 = UpSampling2D(size=(2, 2))(x30)
        merge_20_21 = concatenate([x20, up21], axis=-1)
        x21 = self.up30(merge_20_21)
        # print("x21:", x21.shape)
        # batch x 128 x 128 x 128

        up12 = UpSampling2D(size=(2, 2))(x21)
        merge_10_11_12 = concatenate([x10, x11, up12], axis=-1)
        x12 = self.up21(merge_10_11_12)
        # print("x12:", x12.shape)
        # batch x 256 x 256 x 64

        up03 = UpSampling2D(size=(2, 2))(x12)
        merge_00_01_02_03 = concatenate([x00, x01, x02, up03], axis=-1)
        x03 = self.up12(merge_00_01_02_03)
        # print("x03:", x03.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x40 = self.down30(x30)
        # print("x40:", x40.shape)
        # batch x 32 x 32 x 512

        up31 = UpSampling2D(size=(2, 2))(x40)
        merge_30_31 = concatenate([x30, up31], axis=-1)
        x31 = self.up40(merge_30_31)
        # print("x31:", x31.shape)
        # batch x 64 x 64 x 256

        up22 = UpSampling2D(size=(2, 2))(x31)
        merge_20_21_22 = concatenate([x20, x21, up22], axis=-1)
        x22 = self.up31(merge_20_21_22)
        # print("x22:", x22.shape)
        # batch x 128 x 128 x 128

        up13 = UpSampling2D(size=(2, 2))(x22)
        merge_10_11_12_13 = concatenate([x10, x11, x12, up13], axis=-1)
        x13 = self.up22(merge_10_11_12_13)
        # print("x13:", x13.shape)
        # batch x 256 x 256 x 64

        up04 = UpSampling2D(size=(2, 2))(x13)
        merge_00_01_02_03_04 = concatenate([x00, x01, x02, x03, up04], axis=-1)
        x04 = self.up13(merge_00_01_02_03_04)
        # print("x04:", x04.shape)
        # batch x 512 x 512 x 32
        # v end
        return x04


class WatermarkEncoder2(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(WatermarkEncoder2, self).__init__()
        self.fc = Dense(64 * 64 * 3, activation='relu', kernel_initializer='he_normal')
        self.base_filter_num = base_num
        self.Encoder = Uplusplus(height=height, width=width, base_num=base_num)
        self.RGB_recover = Conv2D(3, 1, activation='tanh', padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        secret, image = inputs
        secret = secret - .5
        image = image - .5
        secret = self.fc(secret)
        secret = Reshape((64, 64, 3))(secret)
        # secret = batch x 50 x 50 x 3
        secret_up = UpSampling2D(size=(8, 8))(secret)
        inputs = concatenate([secret_up, image], axis=-1)
        x04 = self.Encoder(inputs)
        output = self.RGB_recover(x04)
        return output


class WatermarkEncoder(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(WatermarkEncoder, self).__init__()
        self.multiple = 1
        if height % 8 == 0:
            self.multiple = 8
        elif height % 4 == 0:
            self.multiple = 4
        elif height % 2 == 0:
            self.multiple = 2
        self.pre_h = height // self.multiple
        self.pre_w = width // self.multiple
        self.fc = Dense(self.pre_h * self.pre_w * 3, activation='relu', kernel_initializer='he_normal')

        self.base_filter_num = base_num
        self.hidden = Conv2D(base_num, 3, activation='relu', padding='same', kernel_initializer='he_normal')

        self.down00 = Conv2D(base_num * 2, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down10 = Conv2D(base_num * 4, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down20 = Conv2D(base_num * 8, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half
        self.down30 = Conv2D(base_num * 16, 3, activation='relu', strides=2, padding='same',
                             kernel_initializer='he_normal')  # half

        # up
        self.up10 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up20 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up11 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up30 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up21 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up12 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up40 = Conv2D(base_num * 8, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up31 = Conv2D(base_num * 4, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up22 = Conv2D(base_num * 2, 2, activation='relu', padding='same', kernel_initializer='he_normal')
        self.up13 = Conv2D(base_num, 2, activation='relu', padding='same', kernel_initializer='he_normal')

        self.RGB_recover = Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal')

    def call(self, inputs):
        watermark, cover = inputs
        watermark = watermark - .5
        cover = cover - .5
        inputs = concatenate([watermark, cover], axis=-1)
        # print("input:", inputs.shape)
        # batch x 512 x 512 x 6

        # v begin
        x00 = self.hidden(inputs)
        # print("x00:", x00.shape)
        # batch x 512 x 512 x 32

        x10 = self.down00(x00)
        # print("x10:", x10.shape)
        # batch x 256 x 256 x 64

        up01 = UpSampling2D(size=(2, 2))(x10)
        merge_00_01 = concatenate([x00, up01], axis=-1)
        x01 = self.up10(merge_00_01)
        # print("x01", x01.shape)
        # batch x 512 x 512 x 32
        # v end

        # v begin
        x20 = self.down10(x10)
        # print("x20:", x20.shape)
        # batch x 128 x 128 x 128
        up11 = UpSampling2D(size=(2, 2))(x20)
        merge_10_11 = concatenate([x10, up11], axis=-1)
        x11 = self.up20(merge_10_11)
        # print("x11:", x11.shape)
        # batch x 256 x 256 x 64

        up02 = UpSampling2D(size=(2, 2))(x11)
        merge_00_01_02 = concatenate([x00, x01, up02], axis=-1)
        x02 = self.up11(merge_00_01_02)
        # print("x02:", x02.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x30 = self.down20(x20)
        # print("x30:", x30.shape)
        # batch x 64 x 64 x 256
        up21 = UpSampling2D(size=(2, 2))(x30)
        merge_20_21 = concatenate([x20, up21], axis=-1)
        x21 = self.up30(merge_20_21)
        # print("x21:", x21.shape)
        # batch x 128 x 128 x 128

        up12 = UpSampling2D(size=(2, 2))(x21)
        merge_10_11_12 = concatenate([x10, x11, up12], axis=-1)
        x12 = self.up21(merge_10_11_12)
        # print("x12:", x12.shape)
        # batch x 256 x 256 x 64

        up03 = UpSampling2D(size=(2, 2))(x12)
        merge_00_01_02_03 = concatenate([x00, x01, x02, up03], axis=-1)
        x03 = self.up12(merge_00_01_02_03)
        # print("x03:", x03.shape)
        # batch x 512 x 512 x 32

        # v end
        # v begin
        x40 = self.down30(x30)
        # print("x40:", x40.shape)
        # batch x 32 x 32 x 512

        up31 = UpSampling2D(size=(2, 2))(x40)
        merge_30_31 = concatenate([x30, up31], axis=-1)
        x31 = self.up40(merge_30_31)
        # print("x31:", x31.shape)
        # batch x 64 x 64 x 256

        up22 = UpSampling2D(size=(2, 2))(x31)
        merge_20_21_22 = concatenate([x20, x21, up22], axis=-1)
        x22 = self.up31(merge_20_21_22)
        # print("x22:", x22.shape)
        # batch x 128 x 128 x 128

        up13 = UpSampling2D(size=(2, 2))(x22)
        merge_10_11_12_13 = concatenate([x10, x11, x12, up13], axis=-1)
        x13 = self.up22(merge_10_11_12_13)
        # print("x13:", x13.shape)
        # batch x 256 x 256 x 64

        up04 = UpSampling2D(size=(2, 2))(x13)
        merge_00_01_02_03_04 = concatenate([x00, x01, x02, x03, up04], axis=-1)
        x04 = self.up13(merge_00_01_02_03_04)
        # print("x04:", x04.shape)
        # batch x 512 x 512 x 32
        # v end
        output = self.RGB_recover(x04)
        return output


class WatermarkDecoder(Layer):
    def __init__(self, height, width, base_num: int = 32):
        super(WatermarkDecoder, self).__init__()
        self.height = height
        self.width = width
        self.stn_params = Sequential([
            Conv2D(32, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=2, activation='relu', padding='same'),
            Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),
            Flatten(),
            Dense(128, activation='relu')
        ])
        initial = np.array([[1., 0, 0], [0, 1., 0]])
        initial = initial.astype('float32').flatten()

        self.W_fc1 = tf.Variable(tf.zeros([128, 6]), name='W_fc1')
        self.b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        # self.autoencoder = Sequential([
        #     Conv2D(128, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(64, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(32, (3, 3), strides=2, activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(32, (2,2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(32, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(64, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(128, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal'),
        # ])
        self.autoencoder = Encoder(height=height, width=width)
    def call(self, image):
        image = image - .5
        stn_params = self.stn_params(image)
        x = tf.matmul(stn_params, self.W_fc1) + self.b_fc1
        transformed_image = stn_transformer(image, x, [self.height, self.width, 3])
        de_image = self.autoencoder(transformed_image)
        return de_image


class WatermarkDenoiser(Layer):
    def __init__(self, is_train: bool = True):
        super(WatermarkDenoiser, self).__init__()
        self.denoiser = Sequential([
            Conv2D(64, (3, 3), strides=1, activation='relu', padding='same'),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            BatchNormalization(trainable=is_train),
            ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            # Conv2D(64, (3, 3), strides=1, padding='same', use_bias=False),
            # BatchNormalization(trainable=is_train),
            # ReLU(),
            Conv2D(3, (3, 3), strides=1, padding='same', use_bias=False),
        ])
        # self.denoiser2 = Sequential([
        #     Conv2D(64, (3, 3), strides=1, activation='relu', padding='same'),
        #     Conv2D(128, (3, 3), strides=2, padding='same', kernel_initializer='he_normal'),
        #     Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'),
        #     Conv2D(64, (3, 3), strides=2, padding='same', kernel_initializer='he_normal'),
        #     Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal'),
        #     Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer='he_normal'),
        #     Conv2D(32, (3, 3), strides=2, padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(32, (2,2), padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(32, (2, 2), padding='same', kernel_initializer='he_normal'),
        #
        #     Conv2D(64, (2, 2), padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(64, (2, 2), padding='same', kernel_initializer='he_normal'),
        #
        #     Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal'),
        #
        #     UpSampling2D(size=(2, 2)),
        #     Conv2D(128, (2, 2), padding='same', kernel_initializer='he_normal'),
        #     Conv2D(3, 1, activation=None, padding='same', kernel_initializer='he_normal'),
        # ])

    def call(self, image_n):
        denoise = self.denoiser(image_n)
        res = image_n - denoise
        return res, denoise


def image_to_summary(image, name, family='Visual_Result'):
    image = tf.clip_by_value(image, 0, 1)
    image = tf.cast(image * 255, dtype=tf.uint8)
    summary = tf.summary.image(name, image, max_outputs=1, family=family)
    return summary


def calculate_acc(secret_pre, secret_true):
    secret_pre_r = tf.round(secret_pre)
    secret_pre_r = tf.cast(secret_pre_r, dtype=tf.int8)
    secret_label = tf.cast(secret_true, dtype=tf.int8)
    return tf.contrib.metrics.accuracy(secret_pre_r, secret_label)


def get_secret_acc(secret_true, secret_pred):
    with tf.variable_scope("acc"):
        secret_pred = tf.round(tf.sigmoid(secret_pred))
        correct_pred = tf.count_nonzero(secret_pred - secret_true, axis=1)
        bit_acc = 1 - tf.reduce_sum(correct_pred) / tf.size(secret_pred, out_type=tf.int64)
        return bit_acc


def noise_attack(watermarked_image, TFM, args, global_step):
    # use_second = tf.constant(value=args.use_second, dtype=tf.bool)
    linear_fn = lambda linear_total_step: tf.minimum(tf.to_float(global_step) / linear_total_step, 1.)

    # random blur
    # k = util.random_blur_kernel(N_blur=7)
    # attacked_image = tf.nn.conv2d(watermarked_image, k, [1, 1, 1, 1], padding='SAME')
    
    # no blur 3.3
    # attacked_image = util.random_blur(watermarked_image, kenel_size=7)

    # gaussian nosie
    # gaussian_noise = tf.random_normal(shape=tf.shape(attacked_image), mean=0.0, stddev=args.gauss_stddev, dtype=tf.float32)
    # attacked_image = attacked_image + gaussian_noise
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)

    rnd_stddev = tf.random.uniform([]) * linear_fn(args.gaussian_step) * args.gauss_stddev
    attacked_image = util.rnd_gaussain_noise(watermarked_image, gauss_stddev=rnd_stddev)

    # contrast & brightness shift
    # contrast_params = [.5, 1.5]
    # rnd_bri = .3
    # rnd_hue = .1
    #
    # contrast_scale = tf.random_uniform(shape=[tf.shape(attacked_image)[0]], minval=contrast_params[0],
    #                                    maxval=contrast_params[1])
    # contrast_scale = tf.reshape(contrast_scale, shape=[tf.shape(attacked_image)[0], 1, 1, 1])
    # rnd_brightness = util.get_rnd_brightness_tf(rnd_bri, rnd_hue, args.batch_size)
    #
    # attacked_image = attacked_image * contrast_scale
    # attacked_image = attacked_image + rnd_brightness
    # attacked_image = tf.clip_by_value(attacked_image, 0, 1)
    cts_low = 1. - (1. - args.cts_low) * linear_fn(args.cts_step)
    cts_high = 1. + (args.cts_high - 1.) * linear_fn(args.cts_step)
    rnd_bri = linear_fn(args.bri_step) * args.max_bri
    rnd_hue = linear_fn(args.hue_step) * args.max_hue

    if not args.use_second:
        attacked_image = util.rnd_bri_cts(attacked_image, args.batch_size // 2, contrast_low=cts_low,
                                          contrast_high=cts_high,
                                          rnd_bri=rnd_bri, rnd_hue=rnd_hue)
    else:
        attacked_image = tf.image.random_contrast(attacked_image, lower=cts_low, upper=cts_high)
        attacked_image = tf.image.random_hue(attacked_image, max_delta=rnd_hue)
        attacked_image = tf.image.random_brightness(attacked_image, max_delta=rnd_bri)

    # random rat

    rnd_sat = tf.random.uniform([]) * linear_fn(args.sat_step) * args.rnd_sat
    if not args.use_second:
        # attacked_image_lum = tf.expand_dims(tf.reduce_sum(attacked_image * tf.constant([.3, .6, .1]), axis=3), 3)
        # attacked_image = (1 - rnd_sat) * attacked_image + rnd_sat * attacked_image_lum
        attacked_image = util.random_saturation(attacked_image, rnd_sat=rnd_sat)
    else:
        # my
        attacked_image = tf.image.adjust_saturation(attacked_image, saturation_factor=rnd_sat)
        # attacked_image = tf.image.random_saturation(attacked_image, lower=.0, upper=1.0)

    # jpeg compression
    if not args.use_second:
        rnd_factor = tf.random.uniform([]) + 0.1
        attacked_image = util.jpeg_compress_decompress(attacked_image, factor=rnd_factor)
    else:
        attacked_image = util.batch_jepg_attack(attacked_image, low=50, high=100)

    # warp
    if args.is_in_warp:
        attacked_image = util.random_warp(attacked_image, TFM, H=args.cover_h, max_factor=args.max_warp)
    noise_config = [tf.summary.scalar('nosie_config/rnd_bri', rnd_bri, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_sat', rnd_sat, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_hue', rnd_hue, family='noise_config'),
                    tf.summary.scalar('nosie_config/rnd_noise', rnd_stddev, family='noise_config'),
                    tf.summary.scalar('nosie_config/contrast_low', cts_low, family='noise_config'),
                    tf.summary.scalar('nosie_config/contrast_high', cts_high, family='noise_config'),
                    tf.summary.scalar('nosie_config/jpeg_attack_strength', rnd_factor, family='noise_config')]
    return attacked_image, noise_config


def make_graph(Encoder, Decoder, image_batch, loss_ratio_pl, args, TFM, global_step, Denoiser):
    watermark = image_batch[: (args.batch_size // 2)]
    cover = image_batch[(args.batch_size // 2):]
    stego = Encoder([watermark, cover])
    attack_stego, noise_config = noise_attack(stego, TFM=TFM, args=args, global_step=global_step)

    clean_stego, de_noise = Denoiser(attack_stego)
    # loss_Denoiser = tf.losses.mean_squared_error(clean_stego, cover)
    loss_Denoiser = tf.losses.mean_squared_error(clean_stego, stego)

    pre_watermark = Decoder(clean_stego)

    # loss
    wm_psnr = tf.reduce_mean(tf.image.psnr(watermark, pre_watermark, max_val=1.0))
    cover_psnr = tf.reduce_mean(tf.image.psnr(cover, stego, max_val=1.0))

    wm_mse = tf.losses.mean_squared_error(watermark, pre_watermark)
    wm_lpips = tf.reduce_mean(lpips_tf.lpips(watermark, pre_watermark))
    loss_watermark = wm_mse + wm_lpips

    cover_lpips = tf.reduce_mean(lpips_tf.lpips(stego, cover))
    cover_mse = tf.losses.mean_squared_error(stego, cover)

    loss_total = cover_mse * loss_ratio_pl[0] + cover_lpips * loss_ratio_pl[1] + wm_mse * loss_ratio_pl[2] # + wm_lpips * loss_ratio_pl[3]
    config_op = tf.summary.merge([
        tf.summary.scalar('watermark_psnr', wm_psnr, family='loss_config'),
        tf.summary.scalar('cover_psnr', cover_psnr, family='loss_config'),
        tf.summary.scalar('total_loss', loss_total, family='loss_config'),
        tf.summary.scalar('cover_mse', cover_mse, family='loss_config'),
        tf.summary.scalar('cover_Lpip', cover_lpips, family='loss_config'),
        tf.summary.scalar('wm_mse', wm_mse, family='loss_config'),
        tf.summary.scalar('wm_lpips', wm_lpips, family='loss_config'),
        tf.summary.scalar('loss_Denoiser', loss_Denoiser, family='loss_config'),
    ] + noise_config)

    image_summary_op = tf.summary.merge([
        image_to_summary(cover, 'cover_image', family='Visual_Result'),
        image_to_summary(stego, 'stego', family='Visual_Result'),
        image_to_summary(attack_stego, 'attacked_image', family='Visual_Result'),
        image_to_summary(watermark, 'watermark', family='Visual_Result'),
        image_to_summary(pre_watermark, 'pre_watermark', family='Visual_Result'),
        image_to_summary(clean_stego, 'clean_stego', family='Visual_Result'),
        image_to_summary(de_noise, 'De_noise', family='Visual_Result'),
    ])
    return loss_total, loss_watermark, config_op, image_summary_op, cover_psnr, wm_psnr, loss_Denoiser


def make_encode_graph(Encoder, watermark_batch, cover_batch):
    stego = Encoder((watermark_batch, cover_batch))
    watered_image = tf.clip_by_value(stego, 0, 1)
    return watered_image


def make_decode_graph(Decoder, Denoiser, watered_image):
    watered_image, _ = Denoiser(watered_image)

    pre_watermark = Decoder(watered_image)
    pre_watermark = tf.clip_by_value(pre_watermark, 0, 1)
    return pre_watermark


def mymodel_test():
    H = 400
    W = 400
    myEncoder = WatermarkEncoder(width=H, height=W)

    secret = np.random.binomial(1, .5, 100)
    secret = tf.cast(secret, dtype=tf.float32)
    secret = tf.reshape(secret, [1, 100])
    img = tf.random_normal([1, H, W, 3])
    res = myEncoder([img, img])

    myDecoder = WatermarkDecoder(H, W)
    de_secret = myDecoder(res)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        wm, stego = sess.run([res, de_secret])
        print(wm.shape)
        print(stego.shape)







