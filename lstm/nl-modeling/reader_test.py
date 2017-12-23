import reader
import tensorflow as tf

# 数据路径
DATA_PATH = 'simple-examples/data/'

# 读取原始数据
train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

# 将数据组织成batch大小为4，截断长度为5的数据组，要放在开启多线程之前
batch = reader.ptb_producer(train_data, 4, 5)

with tf.Session() as sess:
    tf.global_variables_initializer().run()

    # 开启多线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 读取前两个batch，其中包括每个时刻的输入和对应的答案，ptb_producer()会自动迭代
    for i in range(2):
        x, y = sess.run(batch)
        print('x:', x)
        print('y:', y)

    # 关闭多线程
    coord.request_stop()
    coord.join(threads)
