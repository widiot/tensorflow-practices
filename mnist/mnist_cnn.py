import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

# 导入数据集
mnist = input_data.read_data_sets("data/", one_hot=True)


# 权重初始化：使用标准差为0.1的正态分布
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置初始化：使用常量0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积：使用1步长，0边距的模板，保证输出和输入是同一个大小
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding="SAME")


# 池化：用简单传统的2x2大小的模板
def max_pool_2x2(x):
    return tf.nn.max_pool(
        x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 第一层卷积
x = tf.placeholder(tf.float32, [None, 784])  # 使用占位符，用于指定任意数量的图片
W_conv1 = weight_variable([5, 5, 1,
                           32])  # 第1,2维是patch的大小，第3维是输入的通道数目，第4维是输出的通道数目
b_conv1 = bias_variable([32])  # 每一个输出通道都有一个对应的偏置量

x_image = tf.reshape(
    x, [-1, 28, 28, 1])  # 把x变成一个4d向量，第2、第3维对应图片的宽、高，第4维代表图片的颜色通道数

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(
    h_conv1)  # 把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling

# 第二层卷积
W_conv2 = weight_variable([5, 5, 32, 64])  # 每个5x5的patch会得到64个特征
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 密集连接层
W_fc1 = weight_variable([7 * 7 * 64,
                         1024])  # 图片尺寸减小到7x7,加入一个有1024个神经元的全连接层，用于处理整个图片
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])  # 把池化层输出的张量reshape成一些向量
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder("float")  # 用placeholder来代表一个神经元的输出在dropout中保持不变的概率
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 输出层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)  # 添加一个softmax层

# 最小化交叉熵
y_ = tf.placeholder("float", [None, 10])  # 正确值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(
    cross_entropy)  # 用ADAM优化器来做梯度最速下降

# 初始化变量
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session()
sess.run(init)

# 训练和评估模型
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

for step in range(20000):
    batch = mnist.train.next_batch(50)
    if step % 100 == 0:  # 每100次迭代输出一次日志
        train_accuracy = sess.run(
            accuracy, feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1.0
            })
        print("步骤 %d，训练准确度 %g" % (step, train_accuracy))
    sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

print("测试准确度 %g" % (sess.run(
    accuracy,
    feed_dict={
        x: mnist.test.images,
        y_: mnist.test.labels,
        keep_prob: 1.0
    })))
