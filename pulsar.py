import numpy as np
import csv
import time


np.random.seed(1234)   # seed 고정
def randomize(): np.random.seed(time.time())   # 현재시각을 seed로

RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001



def pulsar_exec(epoch_count=10, mb_size=10, report=1, adjust_ratio=False):
    """
    :param adjust_ratio: False일 경우 원래 데이터셋 사용, True일 경우 균형잡힌 데이터셋 사용
    """
    load_pulsar_dataset(adjust_ratio)
    init_model()
    train_and_test(epoch_count, mb_size, report)


def load_pulsar_dataset(adjust_ratio):
    pulsars, stars = [], []
    with open('./data/pulsar_stars.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        for row in csvreader:
            if row[8] == '1':
                pulsars.append(row)
            else:
                stars.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 8, 1

    star_cnt, pulsar_cnt = len(stars), len(pulsars)

    if adjust_ratio:
        data = np.zeros([2*star_cnt, 9])
        data[:star_cnt, :] = np.asarray(stars, dtype='float32')
        for n in range(star_cnt):
            data[star_cnt+n] = np.asarray(pulsars[n % pulsar_cnt], dtype='float32')
    else:
        data = np.zeros([star_cnt+pulsar_cnt, 9])
        data[:star_cnt, :] = np.asarray(stars, dtype='float32')
        data[star_cnt:, :] = np.asarray(pulsars, dtype='float32')


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
            acc_str = ', '.join(['%5.3f']*4) % tuple(acc)
            print('Epoch {0}: loss={1:5.3f}, result={2}'.\
                  format(epoch+1, np.mean(losses), acc_str))

    final_acc = run_test(test_x, test_y)
    acc_str = ', '.join(['%5.3f'] * 4) % tuple(final_acc)
    print('\nFinal Test: final result = {}'.format(acc_str))


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
    entropy = sigmoid_cross_entropy_with_logits(y, output)
    loss = np.mean(entropy)
    return loss, [y, output, entropy]


def backprop_postproc(G_loss, aux):
    y, output, entropy = aux

    # np.prod() 쓰는 이유는 entropy matrix가 n by m 일 때 n*m으로 나눠줘야 하기 때문
    # 지금은 entropy shape이 input_cnt*1 임
    g_loss_entropy = 1.0 / np.prod(entropy.shape)
    g_entropy_output = sigmoid_cross_entropy_with_logits_derv(y, output)

    G_entropy = g_loss_entropy * G_loss
    G_output = g_entropy_output * G_entropy
    return G_output


def eval_accuracy(output, y):
    est_yes = np.greater(output, 0)
    ans_yes = np.greater(y, 0.5)
    est_no = np.logical_not(est_yes)
    ans_no = np.logical_not(ans_yes)

    tp = np.sum(np.logical_and(est_yes, ans_yes))
    fp = np.sum(np.logical_and(est_yes, ans_no))
    fn = np.sum(np.logical_and(est_no, ans_yes))
    tn = np.sum(np.logical_and(est_no, ans_no))

    accuracy = safe_div(tp+tn, tp+fp+tn+fn)
    precision = safe_div(tp, tp+fp)
    recall = safe_div(tp, tp+fn)
    f1 = 2 * safe_div(precision*recall, precision+recall)

    return [accuracy, precision, recall, f1]


def safe_div(p, q):
    p, q = float(p), float(q)
    if np.abs(q) < 1.0e-20:
        return np.sign(p)
    return p / q


def relu(x):
    return np.maximum(x, 0)


def sigmoid(x):
    return np.exp(-relu(-x)) / (1.0 + np.exp(-np.abs(x)))


def sigmoid_derv(x):
    return sigmoid(x) * (1 - sigmoid(x))


def sigmoid_cross_entropy_with_logits(z, x):
    return relu(x) - x * z + np.log(1 + np.exp(-np.abs(x)))


def sigmoid_cross_entropy_with_logits_derv(z, x):
    return -z + sigmoid(x)



if __name__ == "__main__":
    """
    * 원래 데이터셋은 펄서인 경우가 10%를 차지하고 있다. 이런 경우 제대로 학습되지 않은 신경망도
    무조건 일반별이라고 답하면 정확도는 90%가 될 것이다.
    그렇기 때문에 데이터셋을 균형있게 구성할 필요가 있다. 상대적으로 적은 펄서 데이터를 중복
    사용하되 노이즈를 주입해주어 데이터셋을 구성하면 균형잡힌 데이터셋을 만들 수 있다.
    * 균형잡힌 데이터셋으로 학습을 진행했을 때 재현율, f1이 높아지면서 모델의 성능이 향상되었음을
    확인할 수 있다.
    """
    pulsar_exec(epoch_count=20, mb_size=20,adjust_ratio=True)
