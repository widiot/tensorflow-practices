import reader
import numpy as np
import tensorflow as tf

# 数据参数
DATA_PATH = 'simple-examples/data/'  # 数据存放路径
VOCAB_SIZE = 10000  # 单词数量

# 神经网络参数
HIDDEN_SIZE = 200  # LSTM隐藏层规模
NUM_LAYERS = 2  # LSTM结构层数
LEARNING_RATE = 1.0  # 学习速率
KEEP_PROB = 0.5  # 节点不被dropout的概率
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的参数

# 训练参数
TRAIN_BATCH_SIZE = 20  # 训练数据batch大小
TRAIN_NUM_STEP = 35  # 训练数据截断长度

# 测试参数
EVAL_BATCH_SIZE = 1  # 测试数据batch大小
EVAL_NUM_STEP = 1  # 测试数据截断
NUM_EPOCH = 2  # 使用训练数据的轮数


# 通过PTBModel描述模型，方便维护循环神经网络中的状态
class PTBModel():
    def __init__(self, is_training, batch_size, num_steps):
        # 记录batch和截断长度
        self.batch_size = batch_size
        self.num_steps = num_steps

        # 定义输入层
        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义预期输出
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        # 定义LSTM为使用dropout的两层网络
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(HIDDEN_SIZE)
        if is_training:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(
                lstm_cell, output_keep_prob=KEEP_PROB)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * NUM_LAYERS)

        # 初始化state
        self.initial_state = cell.zero_state(batch_size, tf.float32)

        # 将单词ID转为单词向量。每个单词都是HIDDEN_SIZE维
        embedding = tf.get_variable('embedding', [VOCAB_SIZE, HIDDEN_SIZE])

        # 将原本batch_size*num_steps的输入层转化为batch_size*num_steps*HIDDEN_SIZE
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # 只在训练时使用dropout
        if is_training:
            inputs = tf.nn.dropout(inputs, KEEP_PROB)

        # 定义输出列表
        outputs = []
        state = self.initial_state
        with tf.variable_scope('RNN'):
            for time_step in range(num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                cell_output, state = cell(inputs[:, time_step, :],
                                          state)  # 将当前时刻的数据和状态传入LSTM
                outputs.append(cell_output)  # 将当前输出加入输出列表

        # 将输出列表展开成[batch,hidden_size*num_steps]
        # 再reshape成[batch*num_steps,hidden_size]
        output = tf.reshape(tf.concat(outputs, 1), [-1, HIDDEN_SIZE])

        # 将输出传入全连接层，每个时刻的输出都是长度为VOCAB_SIZE的数组
        weight = tf.get_variable('weight', [HIDDEN_SIZE, VOCAB_SIZE])
        bias = tf.get_variable('bias', [VOCAB_SIZE])
        logits = tf.matmul(output, weight) + bias

        # 定义交叉熵损失函数，sequence_loss_by_example计算一个序列的交叉熵的和
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [logits],  # 预测结果
            [tf.reshape(self.targets, [-1])
             ],  # 预期结果。将[batch_size,num_steps]压缩成一维
            [tf.ones([batch_size * num_steps], dtype=tf.float32)
             ]  # 损失的权重。这里所有的权重都为1，表示不同batch和不同时刻的重要程度都一样
        )

        # 计算得到每个batch的平均损失
        self.cost = tf.reduce_sum(loss) / batch_size
        self.final_state = state

        # 只在训练时反向传播
        if not is_training:
            return
        trainable_variables = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(
            tf.gradients(self.cost, trainable_variables),
            MAX_GRAD_NORM)  # 控制梯度大小。避免梯度膨胀

        # 定义优化方法
        optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE)

        # 定义训练步骤
        self.train_op = optimizer.apply_gradients(
            zip(grads, trainable_variables))


# 使用给定的model在data上运行train_op并返回在全部数据上的perplexity
def run_epoch(session, model, data_queue, train_op, output_log, epoch_size):
    # 计算perplexity的辅助变量
    total_costs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    # 使用当前数据训练或测试模型
    for step in range(epoch_size):
        # 生成输入和答案
        feed_dict = {}
        x, y = session.run(data_queue)
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y

        # 将状态转为字典
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        # 获取损失值和下一个状态
        cost, state, _ = session.run(
            [model.cost, model.final_state, train_op], feed_dict=feed_dict
        )  # 在当前batch上运行train_op并计算损失值。交叉熵损失函数计算的是下一个单词为给定单词的概率
        total_costs += cost
        iters += model.num_steps

        # 训练时输出日志
        if output_log and step % 100 == 0:
            print('After %d steps,perplexity is %.3f' %
                  (step, np.exp(total_costs / iters)))

    return np.exp(total_costs / iters)


def main(_):
    # 原始数据
    train_data, valid_data, test_data, _ = reader.ptb_raw_data(DATA_PATH)

    # 计算一个epoch需要训练的次数
    train_data_len = len(train_data)  # 数据集的大小
    train_batch_len = train_data_len // TRAIN_BATCH_SIZE  # batch的个数
    train_epoch_size = (train_batch_len - 1) // TRAIN_NUM_STEP  # 该epoch的训练次数

    valid_data_len = len(valid_data)
    valid_batch_len = valid_data_len // EVAL_BATCH_SIZE
    valid_epoch_size = (valid_batch_len - 1) // EVAL_NUM_STEP

    test_data_len = len(test_data)
    test_batch_len = test_data_len // EVAL_BATCH_SIZE
    test_epoch_size = (test_batch_len - 1) // EVAL_NUM_STEP

    # 定义初始化函数
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的模型
    with tf.variable_scope(
            'language_model', reuse=None, initializer=initializer):
        train_model = PTBModel(True, TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)

    # 定义评估用的模型
    with tf.variable_scope(
            'language_model', reuse=True, initializer=initializer):
        eval_model = PTBModel(False, EVAL_BATCH_SIZE, EVAL_NUM_STEP)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        # 生成数据队列，必须放在开启多线程之前
        train_queue = reader.ptb_producer(train_data, train_model.batch_size,
                                          train_model.num_steps)
        valid_queue = reader.ptb_producer(valid_data, eval_model.batch_size,
                                          eval_model.num_steps)
        test_queue = reader.ptb_producer(test_data, eval_model.batch_size,
                                         eval_model.num_steps)

        # 开启多线程从而支持ptb_producer()使用tf.train.range_input_producer()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 使用训练数据训练模型
        for i in range(NUM_EPOCH):
            print('In iteration: %d' % (i + 1))
            run_epoch(sess, train_model, train_queue, train_model.train_op,
                      True, train_epoch_size)  # 训练模型
            valid_perplexity = run_epoch(sess, eval_model, valid_queue,
                                         tf.no_op(), False,
                                         valid_epoch_size)  # 使用验证数据评估模型
            print('Epoch: %d Validation Perplexity: %.3f' % (i + 1,
                                                             valid_perplexity))

        # 使用测试数据测试模型
        test_perplexity = run_epoch(sess, eval_model, test_queue,
                                    tf.no_op(), False, test_epoch_size)
        print('Test Perplexity: %.3f' % test_perplexity)

        # 停止所有线程
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    tf.app.run()
