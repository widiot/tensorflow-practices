import tensorflow as tf

# 创建文件列表
files = tf.train.match_filenames_once('data/data.tfrecords-*')

# 创建输入文件队列
filename_queue = tf.train.string_input_producer(files, shuffle=Flase)

# 解析数据。假设image是图像数据，label是标签，height、width、channels给出了图片的维度
reader = tf.TFRecordReader()
_, serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'channels': tf.FixedLenFeature([], tf.int64)
    })
image, label = features['image'], features['label']
height, width = features['height'], features['width']
channels = features['channels']

# 从原始图像中解析出像素矩阵，并还原图像
decoded_image = tf.decode_raw(image, tf.uint8)
decoded_image.set_shape([height, width, channels])

# 定义神经网络输入层图片的大小
image_size = 299

# preprocess_for_train函数是对图片进行预处理的函数
distorted_image = preprocess_for_train(decoded_image, image_size, image_size,
                                       None)

# 组合成batch
min_after_dequeue = 10000
batch_size = 100
capacity = min_after_dequeue + 3 * batch_size
image_batch, label_batch = tf.train.shuffle_batch(
    [distorted_image, label],
    batch_size=batch_size,
    capacity=capacity,
    min_after_dequeue=min_after_dequeue)

# 定义神经网络的结构及优化过程
logit = inference(image_batch)
loss = calc_loss(logit, label_batch)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(
        [tf.global_variables_initializer(),
         tf.local_variables_initializer()])
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # 神经网络训练过程
    for i in range(TRAINING_ROUNDS):
        sess.run(train_step)

    coord.request_stop()
    coord.join()
