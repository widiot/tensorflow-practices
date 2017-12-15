import time
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import inference
import train

# 每10秒加载一次最新的模型，并在测试数据上测试最新模型的正确率
EVAL_INTERVAL_SECS = 10
NUM_VALIDATION = 5000


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        # 定义输入输出的格式
        x = tf.placeholder(
            tf.float32, [
                NUM_VALIDATION, inference.IMAGE_SIZE, inference.IMAGE_SIZE,
                inference.NUM_CHANNELS
            ],
            name='x-input')
        y_ = tf.placeholder(
            tf.float32, [NUM_VALIDATION, inference.OUTPUT_NODE],
            name='y-input')
        y = inference.inference(x, False, None)

        # 验证集
        xs = np.reshape(mnist.validation.images, [
            NUM_VALIDATION, inference.IMAGE_SIZE, inference.IMAGE_SIZE,
            inference.NUM_CHANNELS
        ])
        validate_feed = {x: xs, y_: mnist.validation.labels}

        # 评估模型
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 通过变量重命名方式加载模型，获取滑动平均值
        variable_averages = tf.train.ExponentialMovingAverage(
            train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        # 每隔10秒检测正确率
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    # 加载模型
                    saver.restore(sess, ckpt.model_checkpoint_path)

                    # 通过文件名字获取该模型保存的轮数
                    global_step = ckpt.model_checkpoint_path.split('/')[
                        -1].split('-')[-1]

                    # 验证并输出结果
                    accuracy_score = sess.run(
                        accuracy, feed_dict=validate_feed)
                    print(
                        'After %s training steps, validattion accuracy = %g' %
                        (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('../data/', one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    tf.app.run()
