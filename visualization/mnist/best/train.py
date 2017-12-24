import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference

# 优化方法参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 训练参数
BATCH_SIZE = 100  # 一个训练batch中的图片数
TRAINING_STEPS = 10000  # 训练轮数

# 模型保存的路径和文件名
MODEL_SAVE_PATH = 'model/'
MODEL_NAME = 'mnist.ckpt'


def train(mnist):
    # 将处理输入数据的计算都放在input命名空间下
    with tf.name_scope('input'):
        x = tf.placeholder(
            tf.float32, [None, inference.INPUT_NODE], name='x-input')  # 输入层
        y_ = tf.placeholder(
            tf.float32, [None, inference.OUTPUT_NODE], name='y-input')  # 标签
    regularizer = tf.contrib.layers.l2_regularizer(
        REGULARIZATION_RATE)  # 定义L2正则化损失函数
    y = inference.inference(x, regularizer)  # 输出层

    # 存储训练轮数，设置为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 设置滑动平均方法，都放在moving_average命名空间下
    with tf.name_scope('moving_average'):
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)  # 定义滑动平均类
        variable_averages_op = variable_averages.apply(
            tf.trainable_variables())  # 在所有可训练的变量上使用滑动平均值

    # 最小化损失函数，都放在loss_function命名空间下
    with tf.name_scope('loss_function'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=y, labels=tf.argmax(y_, 1))  # 计算每张图片的交叉熵
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy)  # 计算当前batch中所有图片的交叉熵平均值
        loss = cross_entropy_mean + tf.add_n(
            tf.get_collection('losses'))  # 总损失等于交叉熵损失和正则化损失的和

    # 设置指数衰减法，都放在train_step命名空间下
    with tf.name_scope('train_step'):
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE, global_step,
            mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY)
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
            loss, global_step=global_step)  # 优化损失函数

    # 同时反向传播和滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 开始训练
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 迭代训练
        for i in range(TRAINING_STEPS):
            # 产生该轮batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
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

        # 将当前的计算图输出到TensorBoard日志文件
        writer = tf.summary.FileWriter('log/', tf.get_default_graph())
        writer.close()


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets('../../../mnist/data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
