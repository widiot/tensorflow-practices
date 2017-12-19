import tensorflow as tf

# 获取文件列表
files = tf.train.match_filenames_once('data/data.tfrecords-*')

# 创建文件输入队列
filename_queue = tf.train.string_input_producer(files, shuffle=False)

# 读取并解析Example
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'i': tf.FixedLenFeature([], tf.int64),
        'j': tf.FixedLenFeature([], tf.int64)
    })

# i代表特征向量，j代表标签
example, label = features['i'], features['j']

# 一个batch中的样例数
batch_size = 3

# 文件队列中最多可以存储的样例个数
capacity = 1000 + 3 * batch_size

# 组合样例
example_batch, label_batch = tf.train.batch(
    [example, label], batch_size=batch_size, capacity=capacity)

with tf.Session() as sess:
    # 使用match_filenames_once需要用local_variables_initializer初始化一些变量
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])

    # 用Coordinator协同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 获取并打印组合之后的样例。真实问题中一般作为神经网路的输入
    for i in range(2):
        cur_example_batch, cur_label_batch = sess.run(
            [example_batch, label_batch])
        print(cur_example_batch, cur_label_batch)

    coord.request_stop()
    coord.join(threads)