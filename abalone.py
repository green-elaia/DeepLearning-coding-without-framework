import numpy as np
import csv
import time


np.random.seed(1234)   # seed 고정
def randomize(): np.random.seed(time.time())   # 현재시각을 seed로

RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001


def abalone_exec(epoch_count=10, mb_size=10, report=1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)


def load_abalone_dataset():
    with open('./data/abalone.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        # next()는 iterator의 __next__() 호출시 next item을 검색함. 첫줄이 header row라서 건너뛰기 위함.
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros(shape=(len(rows), input_cnt+output_cnt))

    for n, row in enumerate(rows):
        if row[0] == 'I': data[n, 0] = 1
        if row[0] == 'M': data[n, 1] = 1
        if row[0] == 'F': data[n, 2] = 1
        data[n, 3:] = row[1:]


def init_model():
    """
    initialize a model weight and bias
    """
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros(output_cnt)


def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch+1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {0}: loss={1:5.3f}, accuracy={2:5.3f}/{3:5.3f}'.\
                  format(epoch+1, np.mean(losses), np.mean(accs), acc))

    final_acc = run_test(test_x, test_y)
    print('\nFinal Test: final accuracy = {:5.3f}'.format(final_acc))


def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] *0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count


def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size*nth:mb_size*(nth+1)]]
    return train_data[:,:-output_cnt], train_data[:,-output_cnt:]


def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]


def run_train(x, y):
    output, aux_nn = forward_neuralnet(x)
    loss, aux_pp = forward_postproc(output, y)
    accuracy = eval_accuracy(output, y)

    G_loss =  1.0
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy


def run_test(x, y):
    output, _ = forward_neuralnet(x)
    accuracy = eval_accuracy(output, y)
    return accuracy


def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias
    return output, x


def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()

    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis=0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b


def forward_postproc(output, y):
    diff = output - y
    square = np.square(diff)
    loss = np.mean(square)
    return loss, diff


def backprop_postproc(G_loss, diff):
    shape = diff.shape

    g_loss_square = np.ones(shape) / np.prod(shape)
    g_square_diff = 2 * diff
    g_diff_output = 1

    G_square = g_loss_square * G_loss
    G_diff = g_square_diff * G_square
    G_output = g_diff_output * G_diff

    return G_output


def eval_accuracy(output, y):
    mdiff = np.mean(np.abs((output - y) / y))
    return 1 - mdiff


if __name__ == "__main__":
    """
    * 출력결과 loss는 지속적으로 줄어드는 반면 accuracy는 그대로인 것을 볼 때 
      전복의 외형으로 고리수를 예측한다는 문제 자체가 한계로 작용했을 수 있다.
    * 학습률과 배치 사이즈는 학습에 큰 영향을 주는 하이퍼파라미터이다.
      그래서 여러 경우로 학습해서 비교해보는 것이 필요하다.
    """
    print("Epoch: 10, batch size: 10, Learning rate: {}".format(LEARNING_RATE))
    abalone_exec()
    print("="*100)
    LEARNING_RATE = 0.01
    print("Epoch: 40, batch size: 40, Learning rate: {}".format(LEARNING_RATE))
    abalone_exec(40, 40, 4)
