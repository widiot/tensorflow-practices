import tensorflow as tf

# 创建一个先进先出队列，指定队列最多保存两个元素，类型为整数
q = tf.FIFOQueue(2, 'int32')

# 初始化队列中的元素
init = q.enqueue_many(([0, 10], ))

# 执行出队操作
x = q.dequeue()

# 将元素+1后入队
y = x + 1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    # 运行初始化队列的操作
    init.run()

    # 执行出队+1入队的操作
    for _ in range(5):
        v, _ = sess.run([x, q_inc])
        print(v)
