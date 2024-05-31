max_size = int((1.5 - 0.5) / 0.1 + 1)
duration_range = [0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
n = 40
"""
动态时间窗：预测目标投票+时间权重策略

Input: 预测值数组
Output：False | {'target': 预测值, 'time': 秒 * 10}
策略：
    # 当数组长度小于等于最大长度时：当出现3个连续相同的值 t，则return t，否则 return False
    # 当数组长度达到最大长度时且无连续预测值是：则采用时间权重投票，返回权重最大的预测目标。

单人ITR计算策略：
    n: 40
    p: online_acc
    t: 总时间/num_trial + 0.5
    num_trial: 80
"""


def consistent(res):
    ans = {}
    if len(res) >= 3:
        last_three = res[-3:]
        if all(x == last_three[0] for x in last_three):
            ans['target'] = res[-1]
            ans['time'] = len(res) + 4
            return ans
        else:
            return False
    else:
        return False


def get_max_value_key(dic):
    return max(dic.items(), key=lambda x: x[1])[0]


def time_weighted_res(res):
    index_sum = {}  # 预测值权重字典
    ans = {}
    for i, value in enumerate(res):
        if value in index_sum:
            index_sum[value] += i + 5
        else:
            index_sum[value] = i + 5

    result = get_max_value_key(index_sum)
    ans['target'] = result
    ans['time'] = len(res) + 4
    # print('DICT', index_sum)
    return ans


def online_predict(res):
    # res: 一维预测数组
    if len(res) < max_size:
        return consistent(res)
    elif len(res) == max_size:
        if consistent(res) is False:
            return time_weighted_res(res)
        else:
            return consistent(res)
    else:
        return False


def test():
    # arr = [4, 21, 36, 21, 4, 21, 4, 6, 21, 21, 28]  # {'target': 21, 'time': 15}
    arr = [6, 28, 6, 6, 28, 4]
    # test = [1, 2, 2, 3, 3, 2, 3, 2] # False
    # test = [1, 2, 2, 2] # {'target': 2, 'time': 8}
    # test = [1, 2] # False
    res = online_predict(arr)
    print(res)


if __name__ == "__main__":
    test()


