"""读取用户信号，算法解析，需要 ws"""
import math
from fastapi import APIRouter, WebSocket, status, WebSocketDisconnect, WebSocketException
from pydantic import BaseModel, Field
import numpy as np
from scipy.io import loadmat
import os
from pathname import AllPath
import random
import torch
from constants import BenchmarkDataset, AlgorithmBox
from Algorithm.Model.flexEEG import FlexEEGNet
from Algorithm.Model.DTCN import DTCNet
from Algorithm.online import online_predict
import asyncio
from dotenv import load_dotenv

# 原始信号文件夹
ssvep_dir = AllPath['_SSVEP']
log_dir = AllPath['_LOG']

ssvepApp = APIRouter()


class SubjectSignal(BaseModel):
    dataset: str
    idx: int = Field(default=1, title="受试者序号", description="受试者序号为整数", ge=1)
    algorithm: str
    target: str  # 单目标识别
    log_title: str  # log_title --> 年月日时分_letter_content.mat + 年月日时分_letter_content文件夹【内部是单个字母的详细mat日志】
    trail_title: str


# 脑电数据地址；模型地址；
load_dotenv()
mat_file_path = os.environ.get('MAT_FILE_PATH')
model_dir = os.environ.get('MODEL_DIR')


def model_predict(subject, duration, num_bands, test_data, fs):
    model = FlexEEGNet(
        num_channel=11,
        num_classes=40,
        signal_length=int(fs * duration),
        n_delay=num_bands,
        filters=32,
    )
    model_name = 'FlexEEG_' + str(duration) + 's_S' + str(subject) + '.pth'
    model_path = os.path.join(model_dir, model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        outputs = model(torch.Tensor(test_data))
    target = outputs.argmax(axis=1)

    predict = np.array(target)[0]
    return predict


def model_DTCN_predict(subject, duration, test_data, fs):
    model = DTCNet(
        num_channel=11,
        num_classes=40,
        signal_length=int(fs * duration),
    )

    model_name = 'DTCNet_b4_' + str(duration) + 's_S' + str(subject) + '.pth'
    model_path = os.path.join(model_dir, model_name)
    print('模型', model_name)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    with torch.no_grad():
        outputs = model(torch.Tensor(test_data))
    target = outputs.argmax(axis=1)

    predict = np.array(target)[0]
    return predict


def toTensor(data, count, fs, gaze_shifting_time, subject, num_delay, trail_time):
    trail_data = np.stack((data, data), axis=0)
    duration = round((count - gaze_shifting_time / trail_time) * trail_time, 1)
    test_data = trail_data[:, :, :, int(gaze_shifting_time * fs):int(gaze_shifting_time * fs + duration * fs)]
    res = model_predict(subject=subject, duration=duration, num_bands=num_delay, test_data=test_data, fs=fs)
    return res


def window_predict(data, duration, fs, gaze_shifting_time, subject, num_delay):
    trail_data = np.stack((data, data), axis=0)
    test_data = trail_data[:, :, :, int(gaze_shifting_time * fs):int(gaze_shifting_time * fs + duration * fs)]
    res = model_predict(subject=subject, duration=duration, num_bands=num_delay, test_data=test_data, fs=fs)
    return res


def window_DTCN_predict(data, duration, fs, gaze_shifting_time, subject):
    trail_data = np.stack((data, data), axis=0)
    test_data = trail_data[:, :, int(gaze_shifting_time * fs):int(gaze_shifting_time * fs + duration * fs)]
    print("大小", trail_data.shape, test_data.shape)
    res = model_DTCN_predict(subject=subject, duration=duration, test_data=test_data, fs=fs)
    return res


# 依赖项要求模型加载完成
@ssvepApp.websocket("/model")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # all_time = 0
    try:
        while True:
            message = await websocket.receive_json()
            mission = message.get("Mission")
            dataset = message.get("Dataset")
            algorithm = message.get("Algorithm")
            subject = message.get("Subject")
            windowSize = message.get("WindowSize")
            print('---> 咋回事', mission, dataset, algorithm, subject, windowSize)
            # 1. 判断请求info合法性
            if dataset != 'benchmark':
                raise WebSocketException(
                    code=status.WS_1003_UNSUPPORTED_DATA,
                    reason='目前系统只支持Benchmark范式，带后续版本更新'
                )
            if algorithm not in AlgorithmBox:
                raise WebSocketException(
                    code=status.WS_1003_UNSUPPORTED_DATA,
                    reason='该算法不存在！'
                )
            if int(subject) < 0 or int(subject) > BenchmarkDataset['_SUBJECTS']:
                raise WebSocketException(
                    code=status.WS_1003_UNSUPPORTED_DATA,
                    reason='该受试者不存在！'
                )
            # 2. 读信号
            subject_file = mat_file_path + 'S' + str(subject) + '.mat'
            try:
                data = loadmat(subject_file)
            except FileNotFoundError:
                raise WebSocketException(
                    code=status.WS_1003_UNSUPPORTED_DATA,
                    reason='File not found,请检查路径或导入数据文件'
                )

            # 原始脑电数据
            data = loadmat(subject_file)
            rawdata = data['eeg']

            # 初始化计数器和数据列表
            num_delay = 3  # 延迟时间点
            trail_time = 0.04
            max_time_length = 1.5
            valid_model_box = [
                0.5, 0.6, 0.7, 0.8, 0.9,
                1.0, 1.1, 1.2, 1.3, 1.4, 1.5
            ]  # 动态时间窗模型库
            fs = BenchmarkDataset['_FS']
            pre_stimulus_time = BenchmarkDataset['_TRIGGER_TIME']  # Benchmark 刺激trigger为固定的0.5s。复杂场景需考虑trigger信号的捕获。
            gaze_shifting_time = BenchmarkDataset['_GAZE_SHIFT']
            target_box = BenchmarkDataset['_LOW_TARGETS']
            Test_Block = BenchmarkDataset['_TEST_BLOCKS']
            # 随机选取一个块的数据
            random_block_idx = Test_Block[random.randint(0, len(Test_Block) - 1)]
            count = 1  # 计数器
            data_list = []  # 单目标预测队列

            target_index = target_box.index(mission)  # 目标下标
            block_index = random_block_idx
            stop_flag = False
            # 每40ms发送一次数据包
            if algorithm == "FlexEEGNet":
                print('静态窗算法', algorithm)
                while count <= math.ceil((windowSize + gaze_shifting_time) / trail_time):
                    channel_data = []
                    for delay in range(num_delay):
                        channel_band = rawdata[:, int(pre_stimulus_time * fs + delay):
                                                  int(pre_stimulus_time * fs + trail_time * fs * count + delay),
                                       target_index, block_index]

                        channel_data.append(channel_band)

                    if count == math.ceil((windowSize + gaze_shifting_time) / trail_time):
                        res = window_predict(channel_data, float(windowSize), fs, gaze_shifting_time, subject, num_delay)
                        print('PredictFin_' + target_box[int(res)] + '_' + str(windowSize*10))
                        await websocket.send_text('PredictFin_' + target_box[int(res)] + '_' + str(windowSize*10))
                        break
                    count += 1
                    await asyncio.sleep(trail_time)
            elif algorithm == "DTCNet":
                print('静态窗算法', algorithm)
                while count <= math.ceil((windowSize + gaze_shifting_time) / trail_time):
                    channel_data = rawdata[:, int(pre_stimulus_time * fs):
                                              int(pre_stimulus_time * fs + trail_time * fs * count),
                                   target_index, block_index]

                    if count == math.ceil((windowSize + gaze_shifting_time) / trail_time):
                        res = window_DTCN_predict(channel_data, float(windowSize), fs, gaze_shifting_time, subject)
                        print('PredictFin_' + target_box[int(res)] + '_' + str(windowSize * 10))
                        await websocket.send_text('PredictFin_' + target_box[int(res)] + '_' + str(windowSize * 10))
                        break
                    count += 1
                    await asyncio.sleep(trail_time)
            elif algorithm == "DW-FlexEEGNet":
                print('动态窗算法', algorithm)
                while count <= int((max_time_length / trail_time) + 2):
                    channel_data = []
                    for delay in range(num_delay):
                        channel_band = rawdata[:, int(pre_stimulus_time * fs + delay):
                                                  int(pre_stimulus_time * fs + trail_time * fs * count + delay),
                                                  target_index, block_index]

                        channel_data.append(channel_band)
                    for window in valid_model_box:
                        if count == math.ceil((window + gaze_shifting_time) / trail_time):
                            data_list.append(
                                toTensor(channel_data, count, fs, gaze_shifting_time, subject, num_delay, trail_time))
                            resdata = online_predict(data_list)
                            if resdata is not False:
                                print('PredictFin_' + target_box[int(resdata['target'])] + '_' + str(resdata['time']))
                                await websocket.send_text('PredictFin_' + target_box[int(resdata['target'])]
                                                          + '_' + str(resdata['time']))
                                stop_flag = True
                                break
                    # 停止脑电数据传输
                    if stop_flag is True:
                        break

                    count += 1
                    await asyncio.sleep(trail_time)

    except WebSocketDisconnect:
        print("Websocket disconnected")
    # await websocket.close()

# # 在启动事件中加载模型
# @app.on_event("startup")
# async def load_model(id: int):
#     global model
#     model = SomeModel(id)  # 这里假设SomeModel有一个接受ID参数的构造函数
#     model.load_state_dict(...)  # 这里填入必要的参数以加载模型状态
#
# # 定义一个依赖函数，用于在请求时提供模型对象
# def get_model(id: int):
#     global model
#     return model
#
# # 使用依赖注入加载模型
# @app.get("/predict/{model_id}")
# async def predict(model_id: int, model: SomeModel = Depends(get_model(model_id))):
#     # 在这里使用模型进行预测...
#     result = model.predict(...)  # 这里填写预测所需的代码
#     return {"result": result}
