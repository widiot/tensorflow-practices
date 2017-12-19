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

with tf.Session() as sess:
    # 使用match_filenames_once需要用local_variables_initializer初始化一些变量
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])

    # 打印文件名
    print(sess.run(files))

    # 用Coordinator协同线程，并启动线程
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 获取数据
    for i in range(6):
        print(sess.run([features['i'], features['j']]))
    coord.request_stop()
    coord.join(threads)