import tensorflow as tf


# 创建TFRecord文件的帮助函数
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# 模拟海量数据情况下将数据写入不同的文件
num_shards = 2  # 总共写入多少个文件
instances_per_shard = 2  # 每个文件有多少数据

for i in range(num_shards):
    # 按0000n-of-0000m的后缀区分文件。n代表当前文件编号，m代表文件总数
    filename = ('data/data.tfrecords-%.5d-of-%.5d' % (i, num_shards))
    writer = tf.python_io.TFRecordWriter(filename)

    # 将数据封装成Example结构并写入TFRecord文件
    for j in range(instances_per_shard):
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'i': _int64_feature(i),
                'j': _int64_feature(j)
            }))
        writer.write(example.SerializeToString())
    writer.close()
