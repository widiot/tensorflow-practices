import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import inference

# 优化方法参数
LEARNING_RATE_BASE = 0.05  # 基础学习率
LEARNING_RATE_DECAY = 0.99.  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 训练参数
BATCH_SIZE = 100  # 一个训练batch中的图片数
TRAINING_STEPS = 30000  # 训练轮数

# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'lenet5.ckpt'


def train(mnist):
    # 实现模型
    x = tf.placeholder(
        tf.float32, [
            BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE,
            inference.NUM_CHANNELS
        ],
        name='x-input')  # 输入层
    y_ = tf.placeholder(
        tf.float32, [None, inference.OUTPUT_NODE], name='y-input')  # 标签
    regularizer = tf.contrib.layers.l2_regularizer(
        REGULARIZATION_RATE)  # 定义L2正则化损失函数
    y = inference.inference(x, True, regularizer)  # 输出层

    # 存储训练轮数，设置为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 设置滑动平均方法
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)  # 定义滑动平均类
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())  # 在所有可训练的变量上使用滑动平均

    # 设置指数衰减法
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 最小化损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))  # 计算每张图片的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算当前batch中所有图片的交叉熵平均值
    loss = cross_entropy_mean + tf.add_n(
        tf.get_collection('losses'))  # 总损失等于交叉熵损失和正则化损失的和
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)  # 优化损失函数

    # 同时反向传播和滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化持久化类
    saver = tf.train.Saver()

    # 开始训练
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 迭代训练
        for i in range(TRAINING_STEPS):
            # 产生该轮batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs = np.reshape(
                xs, (BATCH_SIZE, inference.IMAGE_SIZE, inference.IMAGE_SIZE,
                     inference.NUM_CHANNELS))  # 将MNIST数据格式转为四维矩阵
            _, loss_value, step = sess.run(
                [train_op, loss, global_step], feed_dict={
                    x: xs,
                    y_: ys
                })

            # 每1000轮保存一次模型
            if i % 1000 == 0:
                # 输出训练情况
                print('After %d training steps, loss is %g.' % (step,
                                                                loss_value))

                # 保存当前模型
                saver.save(
                    sess,
                    os.path.join(MODEL_SAVE_PATH, MODEL_NAME),
                    global_step=global_step)


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets('../data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
