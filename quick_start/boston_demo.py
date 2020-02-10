from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy
import math
import sys


BATCH_SIZE = 20

train_reader = paddle.batch(paddle.reader.shuffle(
        paddle.dataset.uci_housing.train(), buf_size=500),
        batch_size=BATCH_SIZE)

test_reader = paddle.batch(paddle.reader.shuffle(
        paddle.dataset.uci_housing.test(), buf_size=500),
        batch_size=BATCH_SIZE)


# 配置训练程序
x = fluid.layers.data(name='x', shape=[13], dtype='float32')  # 定义输入的形状和数据类型
y = fluid.layers.data(name='y', shape=[1], dtype='float32')  # 定义输出的形状和数据类型
y_predict = fluid.layers.fc(input=x, size=1, act=None)  # 连接输入和输出的全连接层

main_program = fluid.default_main_program()  # 获取默认/全局主函数
startup_program = fluid.default_startup_program()  # 获取默认/全局启动程序

cost = fluid.layers.square_error_cost(input=y_predict, label=y)  # 利用标签数据和输出的预测数据估计方差
avg_loss = fluid.layers.mean(cost)  # 对方差求均值，得到平均损失

# Optimizer Function 配置
sgd_optimizer = fluid.optimizer.SGD(learning_rate=0.001)
sgd_optimizer.minimize(avg_loss)


# 克隆main_program得到test_program
# 有些operator在训练和测试之间的操作是不同的，例如batch_norm，使用参数for_test来区分该程序是用来训练还是用来测试
# 该api不会删除任何操作符,请在backward和optimization之前使用
test_program = main_program.clone(for_test=True)


# 定义运算场所
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()  # 指明executor的执行场所

# executor可以接受传入的program，并根据feed map(输入映射表)和fetch list(结果获取表)向program中添加数据输入算子和结果获取算子。使用close()关闭该executor，调用run(...)执行program。
exe = fluid.Executor(place)

# 创建训练过程
num_epochs = 100


def train_test(executor, program, reader, feeder, fetch_list):
    accumulated = 1 * [0]
    count = 0
    for data_test in reader():
        outs = executor.run(program=program,
                            feed=feeder.feed(data_test),
                            fetch_list=fetch_list)
        accumulated = [x_c[0] + x_c[1][0] for x_c in zip(accumulated, outs)]  # 累加测试过程中的损失值
        count += 1  # 累加测试集中的样本数量
    return [x_d / count for x_d in accumulated]  # 计算平均损失


# 训练主循环
# %matplotlib inline
params_dirname = "fit_a_line.inference.model"
feeder = fluid.DataFeeder(place=place, feed_list=[x, y])
exe.run(startup_program)
train_prompt = "train cost"
test_prompt = "test cost"
from paddle.utils.plot import Ploter
plot_prompt = Ploter(train_prompt, test_prompt)
step = 0

exe_test = fluid.Executor(place)


for pass_id in range(num_epochs):
    for data_train in train_reader():
        avg_loss_value, = exe.run(main_program,
                                  feed=feeder.feed(data_train),
                                  fetch_list=[avg_loss])
        if step % 10 == 0: # 每10个批次记录并输出一下训练损失
            # plot_prompt.append(train_prompt, step, avg_loss_value[0])
            # plot_prompt.plot()
            print("%s, Step %d, Cost %f" %
                      (train_prompt, step, avg_loss_value[0]))
        if step % 100 == 0:  # 每100批次记录并输出一下测试损失
            test_metics = train_test(executor=exe_test,
                                     program=test_program,
                                     reader=test_reader,
                                     fetch_list=[avg_loss.name],
                                     feeder=feeder)
            # plot_prompt.append(test_prompt, step, test_metics[0])
            # plot_prompt.plot()
            print("%s, Step %d, Cost %f" %
                      (test_prompt, step, test_metics[0]))
            if test_metics[0] < 10.0: # 如果准确率达到要求，则停止训练
                break

        step += 1

        if math.isnan(float(avg_loss_value[0])):
            sys.exit("got NaN loss, training failed.")

        #保存训练参数到之前给定的路径中
        if params_dirname is not None:
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)


