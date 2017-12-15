import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf

# 导入数据集
mnist = input_data.read_data_sets("data/", one_hot=True)

# 实现softmax回归模型
x = tf.placeholder(tf.float32, [None, 784])  # 使用占位符，用于指定任意数量的图片
W = tf.Variable(tf.zeros([784, 10]))  # 初始化权重值
b = tf.Variable(tf.zeros([10]))  # 初始化偏置量
y = tf.nn.softmax(tf.matmul(x, W) + b)  # 实现softmax回归模型函数

# 最小化交叉熵
y_ = tf.placeholder("float", [None, 10])  # 正确值
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 计算交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(
    cross_entropy)  # 用梯度下降算法以0.01的学习速率最小化交叉熵

# 初始化变量
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session()
sess.run(init)

# 训练模型
for step in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# 评估模型
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) # 将布尔值转为浮点数
print(
    sess.run(
        accuracy, feed_dict={
            x: mnist.test.images,
            y_: mnist.test.labels
        }))
