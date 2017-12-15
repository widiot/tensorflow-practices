import tensorflow as tf
import numpy as np

# 创建100个点
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3

# 构造一个softmax线性模型
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))
y = x_data * Weights + biases

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))