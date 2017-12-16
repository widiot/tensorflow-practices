import tensorflow as tf

# 创建一个reader来读取TFRecord文件中的Example
reader = tf.TFRecordReader()

# 创建一个队列来维护输入文件列表
filename_queue = tf.train.string_input_producer(['record/output.tfrecords'])

# 从文件中读出一个Example
_, serialized_example = reader.read(filename_queue)

# 用FixedLenFeature将读入的Example解析成tensor
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# 将字符串解析成图像对应的像素数组
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

# 每次运行读取一个Example。当所有样例读取完之后，在此样例中程序会重头读取
for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(label)
