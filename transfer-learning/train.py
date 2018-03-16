import glob
import os.path
import random
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

# 数据参数
MODEL_DIR = 'model/'  # inception-v3模型的文件夹
MODEL_FILE = 'tensorflow_inception_graph.pb'  # inception-v3模型文件名
CACHE_DIR = 'data/tmp/bottleneck'  # 图像的特征向量保存地址
INPUT_DATA = 'data/flower_photos'  # 图片数据文件夹
VALIDATION_PERCENTAGE = 10  # 验证数据的百分比
TEST_PERCENTAGE = 10  # 测试数据的百分比

# inception-v3模型参数
BOTTLENECK_TENSOR_SIZE = 2048  # inception-v3模型瓶颈层的节点个数
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'  # inception-v3模型中代表瓶颈层结果的张量名称
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'  # 图像输入张量对应的名称

# 神经网络的训练参数
LEARNING_RATE = 0.01
STEPS = 1000
BATCH = 100
CHECKPOINT_EVERY = 100
NUM_CHECKPOINTS = 5


# 从数据文件夹中读取所有的图片列表并按训练、验证、测试分开
def create_image_lists(validation_percentage, test_percentage):
    result = {}  # 保存所有图像。key为类别名称。value也是字典，存储了所有的图片名称
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]  # 获取所有子目录
    is_root_dir = True  # 第一个目录为当前目录，需要忽略

    # 分别对每个子目录进行操作
    for sub_dir in sub_dirs:
        if is_root_dir:
            is_root_dir = False
            continue

        # 获取当前目录下的所有有效图片
        extensions = {'jpg', 'jpeg', 'JPG', 'JPEG'}
        file_list = []  # 存储所有图像
        dir_name = os.path.basename(sub_dir)  # 获取路径的最后一个目录名字
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list:
            continue

        # 将当前类别的图片随机分为训练数据集、测试数据集、验证数据集
        label_name = dir_name.lower()  # 通过目录名获取类别的名称
        training_images = []
        testing_images = []
        validation_images = []
        for file_name in file_list:
            base_name = os.path.basename(file_name)  # 获取该图片的名称
            chance = np.random.randint(100)  # 随机产生100个数代表百分比
            if chance < validation_percentage:
                validation_images.append(base_name)
            elif chance < (validation_percentage + test_percentage):
                testing_images.append(base_name)
            else:
                training_images.append(base_name)

        # 将当前类别的数据集放入结果字典
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images
        }

    # 返回整理好的所有数据
    return result


# 通过类别名称、所属数据集、图片编号获取一张图片的地址
def get_image_path(image_lists, image_dir, label_name, index, category):
    label_lists = image_lists[label_name]  # 获取给定类别中的所有图片
    category_list = label_lists[category]  # 根据所属数据集的名称获取该集合中的全部图片
    mod_index = index % len(category_list)  # 规范图片的索引
    base_name = category_list[mod_index]  # 获取图片的文件名
    sub_dir = label_lists['dir']  # 获取当前类别的目录名
    full_path = os.path.join(image_dir, sub_dir, base_name)  # 图片的绝对路径
    return full_path


# 通过类别名称、所属数据集、图片编号获取特征向量值的地址
def get_bottleneck_path(image_lists, label_name, index, category):
    return get_image_path(image_lists, CACHE_DIR, label_name, index,
                          category) + '.txt'


# 使用inception-v3处理图片获取特征向量
def run_bottleneck_on_image(sess, image_data, image_data_tensor,
                            bottleneck_tensor):
    bottleneck_values = sess.run(bottleneck_tensor,
                                 {image_data_tensor: image_data})
    bottleneck_values = np.squeeze(bottleneck_values)  # 将四维数组压缩成一维数组
    return bottleneck_values


# 获取一张图片经过inception-v3模型处理后的特征向量
def get_or_create_bottleneck(sess, image_lists, label_name, index, category,
                             jpeg_data_tensor, bottleneck_tensor):
    # 获取一张图片对应的特征向量文件的路径
    label_lists = image_lists[label_name]
    sub_dir = label_lists['dir']
    sub_dir_path = os.path.join(CACHE_DIR, sub_dir)
    if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)
    bottleneck_path = get_bottleneck_path(image_lists, label_name, index,
                                          category)

    # 如果该特征向量文件不存在，则通过inception-v3模型计算并保存
    if not os.path.exists(bottleneck_path):
        image_path = get_image_path(image_lists, INPUT_DATA, label_name, index,
                                    category)  # 获取图片原始路径
        image_data = gfile.FastGFile(image_path, 'rb').read()  # 获取图片内容
        bottleneck_values = run_bottleneck_on_image(
            sess, image_data, jpeg_data_tensor,
            bottleneck_tensor)  # 通过inception-v3计算特征向量

        # 将特征向量存入文件
        bottleneck_string = ','.join(str(x) for x in bottleneck_values)
        with open(bottleneck_path, 'w') as bottleneck_file:
            bottleneck_file.write(bottleneck_string)
    else:
        # 否则直接从文件中获取图片的特征向量
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]

    # 返回得到的特征向量
    return bottleneck_values


# 随机获取一个batch图片作为训练数据
def get_random_cached_bottlenecks(sess, n_classes, image_lists, how_many,
                                  category, jpeg_data_tensor,
                                  bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    for _ in range(how_many):
        # 随机一个类别和图片编号加入当前的训练数据
        label_index = random.randrange(n_classes)
        label_name = list(image_lists.keys())[label_index]
        image_index = random.randrange(65535)
        bottleneck = get_or_create_bottleneck(
            sess, image_lists, label_name, image_index, category,
            jpeg_data_tensor, bottleneck_tensor)
        ground_truth = np.zeros(n_classes, dtype=np.float32)
        ground_truth[label_index] = 1.0
        bottlenecks.append(bottleneck)
        ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


# 获取全部的测试数据
def get_test_bottlenecks(sess, image_lists, n_classes, jpeg_data_tensor,
                         bottleneck_tensor):
    bottlenecks = []
    ground_truths = []
    label_name_list = list(image_lists.keys())
    # 枚举所有的类别和每个类别中的测试图片
    for label_index, label_name in enumerate(label_name_list):
        category = 'testing'
        for index, unused_base_name in enumerate(
                image_lists[label_name][category]):
            bottleneck = get_or_create_bottleneck(
                sess, image_lists, label_name, index, category,
                jpeg_data_tensor, bottleneck_tensor)
            ground_truth = np.zeros(n_classes, dtype=np.float32)
            ground_truth[label_index] = 1.0
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
    return bottlenecks, ground_truths


def main(_):
    # 读取所有的图片
    image_lists = create_image_lists(VALIDATION_PERCENTAGE, TEST_PERCENTAGE)
    n_classes = len(image_lists.keys())

    # 读取训练好的inception-v3模型
    with gfile.FastGFile(os.path.join(MODEL_DIR, MODEL_FILE), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # 加载inception-v3模型，并返回数据输入张量和瓶颈层输出张量
    bottleneck_tensor, jpeg_data_tensor = tf.import_graph_def(
        graph_def,
        return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME])

    # 定义新的神经网络输入
    bottleneck_input = tf.placeholder(
        tf.float32, [None, BOTTLENECK_TENSOR_SIZE],
        name='BottleneckInputPlaceholder')

    # 定义新的标准答案输入
    ground_truth_input = tf.placeholder(
        tf.float32, [None, n_classes], name='GroundTruthInput')

    # 定义一层全连接层解决新的图片分类问题
    with tf.name_scope('final_training_ops'):
        weights = tf.Variable(
            tf.truncated_normal(
                [BOTTLENECK_TENSOR_SIZE, n_classes], stddev=0.1))
        biases = tf.Variable(tf.zeros([n_classes]))
        logits = tf.matmul(bottleneck_input, weights) + biases
        final_tensor = tf.nn.softmax(logits)

    # 定义交叉熵损失函数
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=ground_truth_input)
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        cross_entropy_mean)

    # 计算正确率
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(
            tf.argmax(final_tensor, 1), tf.argmax(ground_truth_input, 1))
        evaluation_step = tf.reduce_mean(
            tf.cast(correct_prediction, tf.float32))

    # 训练过程
    with tf.Session() as sess:
        init = tf.global_variables_initializer().run()

        # 模型和摘要的保存目录
        import time
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(
            os.path.join(os.path.curdir, 'runs', timestamp))
        print('\nWriting to {}\n'.format(out_dir))
        # 损失值和正确率的摘要
        loss_summary = tf.summary.scalar('loss', cross_entropy_mean)
        acc_summary = tf.summary.scalar('accuracy', evaluation_step)
        # 训练摘要
        train_summary_op = tf.summary.merge([loss_summary, acc_summary])
        train_summary_dir = os.path.join(out_dir, 'summaries', 'train')
        train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                     sess.graph)
        # 开发摘要
        dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # 保存检查点
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(
                tf.global_variables(), max_to_keep=NUM_CHECKPOINTS)

        for i in range(STEPS):
            # 每次获取一个batch的训练数据
            train_bottlenecks, train_ground_truth = get_random_cached_bottlenecks(
                sess, n_classes, image_lists, BATCH, 'training',
                jpeg_data_tensor, bottleneck_tensor)
            _, train_summaries = sess.run(
                [train_step, train_summary_op],
                feed_dict={
                    bottleneck_input: train_bottlenecks,
                    ground_truth_input: train_ground_truth
                })

            # 保存每步的摘要
            train_summary_writer.add_summary(train_summaries, i)

            # 在验证集上测试正确率
            if i % 100 == 0 or i + 1 == STEPS:
                validation_bottlenecks, validation_ground_truth = get_random_cached_bottlenecks(
                    sess, n_classes, image_lists, BATCH, 'validation',
                    jpeg_data_tensor, bottleneck_tensor)
                validation_accuracy, dev_summaries = sess.run(
                    [evaluation_step, dev_summary_op],
                    feed_dict={
                        bottleneck_input: validation_bottlenecks,
                        ground_truth_input: validation_ground_truth
                    })
                print(
                    'Step %d : Validation accuracy on random sampled %d examples = %.1f%%'
                    % (i, BATCH, validation_accuracy * 100))

            # 每隔checkpoint_every保存一次模型和测试摘要
            if i % CHECKPOINT_EVERY == 0:
                dev_summary_writer.add_summary(dev_summaries, i)
                path = saver.save(sess, checkpoint_prefix, global_step=i)
                print('Saved model checkpoint to {}\n'.format(path))

        # 最后在测试集上测试正确率
        test_bottlenecks, test_ground_truth = get_test_bottlenecks(
            sess, image_lists, n_classes, jpeg_data_tensor, bottleneck_tensor)
        test_accuracy = sess.run(
            evaluation_step,
            feed_dict={
                bottleneck_input: test_bottlenecks,
                ground_truth_input: test_ground_truth
            })
        print('Final test accuracy = %.1f%%' % (test_accuracy * 100))


if __name__ == '__main__':
    tf.app.run()
