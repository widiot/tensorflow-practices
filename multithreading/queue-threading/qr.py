import tensorflow as tf

# 声明一个FIFO队列，最多100个实数元素
q = tf.FIFOQueue(100, 'float')

# 定义队列的入队操作
enqueue_op = q.enqueue([tf.random_normal([1])])

# 使用QueueRunner来创建多个线程运行队列的入队操作
qr = tf.train.QueueRunner(q, [enqueue_op] * 5)

# 将定义过的QueueRunner加入计算图上指定的集合
tf.train.add_queue_runner(qr)

# 定义出队操作
dequeue_op = q.dequeue()

with tf.Session() as sess:
    # 使用Coordinator来协同启动的线程
    coord = tf.train.Coordinator()

    # 使用QueueRunner时需要明确调用tf.train.start_queue_runners来启动所有进程
    threads = tf.train.start_queue_runners(coord=coord)

    # 获取队列中的取值
    for _ in range(3):
        print(sess.run(dequeue_op)[0])

    # 停止所有线程
    coord.request_stop()
    coord.join(threads)
