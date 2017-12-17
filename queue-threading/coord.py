import tensorflow as tf
import numpy as np
import threading
import time


# 线程中运行的程序，每隔1秒判断是否停止并打印ID
def MyLoop(coord, worker_id):
    while not coord.should_stop():
        # 随机停止所有线程
        if np.random.rand() < 0.1:
            print('Stoping from id: %d\n' % (worker_id))
            coord.request_stop()
        else:
            print('Working on id: %d\n' % (worker_id))
        time.sleep(1)


# 创建Coordinator
coord = tf.train.Coordinator()

# 声明创建5个线程
threads = [threading.Thread(target=MyLoop, args=(coord, i)) for i in range(5)]

# 启动所有线程
for t in threads:
    t.start()

# 等待所有线程退出
coord.join()