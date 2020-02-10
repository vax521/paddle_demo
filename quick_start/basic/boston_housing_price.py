# step 1 准备数据

import paddle.fluid as fluid
import paddle
import numpy as np
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

BUF_SIZE = 500
BATCH_SIZE = 20

# 训练数据读取器，每次从缓存随机读取批次大小数据
train_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.train(),
                            buf_size=BUF_SIZE),
                            batch_size=BATCH_SIZE)
# 测试数据读取器
test_reader = paddle.batch(paddle.reader.shuffle(paddle.dataset.uci_housing.test(),buf_size=BUF_SIZE),
                           batch_size=BATCH_SIZE)

# 查看数据
print(next(paddle.dataset.uci_housing.train()()))


# STEP2 网络配置
x = fluid.layers.data(name='x', shape=[13], dtype='float32')
y = fluid.layers.data(name='y', shape=[1], dtype='float32')
y_predict = fluid.layers.fc(input=x,size=1,act=None)
# 定义损失函数
cost = fluid.layers.square_error_cost(input=y_predict,label=y)
avg_cost = fluid.layers.mean(cost)
# 定义优化函数
optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
opts = optimizer.minimize(avg_cost)
test_program = fluid.default_main_program().clone(for_test=True)


# STEP3 模型训练
# 创建Excutor
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_main_program())
# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[x,y],)

# 绘制损失值变化趋势
iter=0
iters=[]
train_costs=[]


def draw_train_process(iters,train_costs):
    title = "training cost"
    plt.title(title)
    plt.xlabel("iter", fontsize=24)
    plt.ylabel("cost", fontsize=24)
    plt.plot(iters, train_costs, coler='red',label='train cost')
    plt.show()
# 保存训练模型
EPOCH_NUM=50
model_save_dir = "./model/fit_a_line.inference.model"
for pass_id in range(EPOCH_NUM):
    train_cost = 0
    for batch_id,data in enumerate(train_reader()):
        train_cost = exe.run(program=fluid.default_main_program(),
                             feed=feeder.feed(data),
                             fetch_list=[avg_cost])
        if batch_id % 40 == 0:
            print("Pass:%d,Cost:%.5f"%(pass_id,train_cost))
        iter = iter+BATCH_SIZE
        iters.append(iter)
        train_costs.append(train_cost[0][0])
    test_cost = 0
    for batch_id,data in enumerate(test_reader()):
        test_cost = exe.run(program=test_program,
                            feed=feeder.feed(data),
                            fetch_list=[avg_cost])
    print('Test:%d,Cost:%.5f' % (pass_id, test_cost[0]))
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

print("save model to %s" % model_save_dir)
fluid.io.save_inference_model(model_save_dir,
                              ['x'],
                              [y_predict],
                              exe)
draw_train_process(iters,train_costs)