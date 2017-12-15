import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 神经网络结构参数
INPUT_NODE = 784  # 输入层节点数。等于MNIST图片的像素
LAYER_NODE = 500  # 隐藏层节点数。只用一个隐藏层，含500个节点
OUTPUT_NODE = 10  # 输出层节点数。等于0~9对应的10个数字

# 优化方法参数
LEARNING_RATE_BASE = 0.8  # 基础学习率
LEARNING_RATE_DECAY = 0.99  # 学习率的衰减率
REGULARIZATION_RATE = 0.0001  # 正则化项在损失函数中的系数
MOVING_AVERAGE_DECAY = 0.99  # 滑动平均衰减率

# 训练参数
BATCH_SIZE = 100  # 一个训练batch中的图片数
TRAINING_STEPS = 30000  # 训练轮数


# 利用给定神经网络的输入和参数，返回前向传播结果
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 如果没有提供滑动平均类，直接使用参数当前的取值
    if avg_class == None:
        # 计算隐藏层的前向传播结果
        layer = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)

        # 返回输出层的前向传播结果
        return tf.matmul(layer, weights2) + biases2
    else:
        # 计算变量的滑动平均值，再计算前向传播结果
        layer = tf.nn.relu(
            tf.matmul(input_tensor, avg_class.average(weights1)) +
            avg_class.average(biases1))
        return tf.matmul(
            layer, avg_class.average(weights2)) + avg_class.average(biases2)


# 训练模型的过程
def train(mnist):
    # 实现模型
    x = tf.placeholder(tf.float32, [None, INPUT_NODE])  # 输入层
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE])  # 标签
    weights1 = tf.Variable(
        tf.truncated_normal([INPUT_NODE, LAYER_NODE], stddev=0.1))  # 隐藏层权重
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER_NODE]))  # 隐藏层偏置
    weights2 = tf.Variable(
        tf.truncated_normal([LAYER_NODE, OUTPUT_NODE], stddev=0.1))  # 输出层权重
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))  # 输出层偏置
    y = inference(x, None, weights1, biases1, weights2, biases2)  # 输出层

    # 存储训练轮数，设置为不可训练
    global_step = tf.Variable(0, trainable=False)

    # 设置滑动平均方法
    variable_averages = tf.train.ExponentialMovingAverage(
        MOVING_AVERAGE_DECAY, global_step)  # 定义滑动平均类
    variable_averages_op = variable_averages.apply(
        tf.trainable_variables())  # 在所有可训练的变量上使用滑动平均值
    average_y = inference(x, variable_averages, weights1, biases1, weights2,
                          biases2)  # 计算使用了滑动平均的前向传播结果

    # 设置正则化方法
    regularizer = tf.contrib.layers.l2_regularizer(
        REGULARIZATION_RATE)  # 定义L2正则化损失函数
    regularization = regularizer(weights1) + regularizer(
        weights2)  # 计算模型的正则化损失

    # 设置指数衰减法
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE, global_step, mnist.train.num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY)

    # 最小化损失函数
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=y, labels=tf.argmax(y_, 1))  # 计算每张图片的交叉熵
    cross_entropy_mean = tf.reduce_mean(cross_entropy)  # 计算当前batch中所有图片的交叉熵平均值
    loss = cross_entropy_mean + regularization  # 总损失等于交叉熵损失和正则化损失的和
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(
        loss, global_step=global_step)  # 优化损失函数

    # 同时反向传播和滑动平均
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(average_y, 1),
                                  tf.argmax(y_, 1))  # 检验使用滑动平均模型的前向传播的是否正确
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 计算正确率

    # 开始训练
    with tf.Session() as sess:
        # 初始化所有变量
        tf.global_variables_initializer().run()

        # 验证数据及测试数据
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}

        # 迭代训练
        for i in range(TRAINING_STEPS):
            # 每1000轮输出在验证数据集上的正确率
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print('After %d training steps, validateion accuracy is %g ' %
                      (i, validate_acc))

            # 产生该轮batch
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 训练结束在测试集上计算正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print('After %d training steps, test accuracy is %g ' %
              (TRAINING_STEPS, test_acc))


# 主程序入口
def main(argv=None):
    mnist = input_data.read_data_sets('data/', one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()
