import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

SUMMARY_DIR = 'log/'
BATCH_SIZE = 100
TRAIN_STEPS = 10000


# 生成变量监控信息，并定义生成监控信息日志的操作
def variable_summaries(name, var):
    # 将生成监控信息的操作放在同一个命名空间下
    with tf.name_scope('summaries'):
        # 记录张量中元素的取值分布
        tf.summary.histogram(name, var)

        # 计算变量的平均值
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        # 计算变量的标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)


# 生成一层全连接层神经网络
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在同一个命名空间下
    with tf.name_scope(layer_name):
        # 声明权重
        with tf.name_scope('weights'):
            weights = tf.Variable(
                tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(layer_name + '/weights', weights)

        # 声明偏置
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(layer_name + '/biases', biases)

        with tf.name_scope('Wx_plus_b'):
            # 记录输出节点在经过激活函数之前的分布
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)

        # 记录输出节点经过激活函数之后的分布
        activations = act(preactivate, name='activation')
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


def main(_):
    mnist = input_data.read_data_sets('../../mnist/data/', one_hot=True)

    # 定义输入
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 记录将输入向量还原成的像素矩阵
    with tf.name_scope('input_reshape'):
        image_reshaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_reshaped_input)

    hidden1 = nn_layer(x, 784, 500, 'layer1')
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity)

    # 计算并记录交叉熵
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
        tf.summary.scalar('cross entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 计算模型在当前给定数据上的正确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 执行所有日志生成操作
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        # 初始化写日志的writer
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph)
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 运行训练步骤以及所有的日志操作
            summary, _ = sess.run(
                [merged, train_step], feed_dict={
                    x: xs,
                    y_: ys
                })

            # 写入所有日志
            summary_writer.add_summary(summary, i)

            if i % 1000 == 0:
                print(i)

    summary_writer.close()


if __name__ == '__main__':
    tf.app.run()
