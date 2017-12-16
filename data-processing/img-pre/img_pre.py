import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# 通过调整亮度、对比度、饱和度、色相的顺序随机调整图像的色彩
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32 / 255.0)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
    elif color_ordering == 1:
        image = tf.image.random_brightness(image, max_delta=32 / 255.0)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    elif color_ordering == 2:
        # 一共可以有24种排列情况
        pass
    return tf.clip_by_value(image, 0.0, 1.0)


# 对图像进行预处理
def preprocess_for_train(image, height, width, bbox):
    # 默认整个图像是需要关注的部分
    if bbox is None:
        bbox = tf.constant(
            [0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    # 转换图像张量的类型
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随机截取图像，减少需要关注的物体大小对图像识别算法的影响
    bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(
        tf.shape(image), bounding_boxes=bbox, min_object_covered=0.1)
    distorted_image = tf.slice(image, bbox_begin, bbox_size)

    # 将随机截取的图像调整为输入层的大小
    distorted_image = tf.image.resize_images(
        distorted_image, size=[height, width], method=np.random.randint(4))

    # 随机左右翻转图像
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # 使用一种随机顺序调整图像色彩
    distorted_image = distort_color(distorted_image, np.random.randint(2))
    return distorted_image


image_raw_data = tf.gfile.FastGFile('images/1.jpg', 'rb').read()
with tf.Session() as sess:
    img_data = tf.image.decode_jpeg(image_raw_data)
    boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    for i in range(6):
        result = preprocess_for_train(img_data, 300, 300, boxes)
        plt.imshow(result.eval())
        plt.show()