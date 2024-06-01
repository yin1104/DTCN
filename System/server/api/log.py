"""请求下载日志的，还没想好，可能就是纯纯帮忙打开文件夹？有点懒，再说"""
import csv
import json

from fastapi import APIRouter, HTTPException, status
import os
from pathname import AllPath
from pydantic import BaseModel
import pandas as pd
from Algorithm.ITR import itr


logApp = APIRouter()

# 日志文件夹
log_dir = AllPath['_LOG']


class Report(BaseModel):
    log_title: str
    ground_truth: str
    predicted_ans: str
    zh_ans: str
    signal_acc: str
    signal_time_used: str
    translate_time_used: str = "0"
    total_time: str
    trail_time: str
    dataset: str
    algorithm: str
    subject: str


def online_itr(n, signal_acc, total_time, translate_time_used, ground_truth):
    # t = (total_time - translate_time_used - len(ground_truth) * 4) / len(ground_truth)
    t = (total_time - len(ground_truth) * 4) / len(ground_truth)  # 把语音时间损耗计算到ITR中
    p = signal_acc / 100
    return itr(n, p, t)


def write_log(report: dict):
    log_path = os.path.join(log_dir, report['log_title'])
    report['ITR'] = online_itr(
        40,
        float(report['signal_acc']),
        float(report['total_time']),
        float(report['translate_time_used']),
        report['ground_truth']
    )
    log_file = log_path + '.csv'
    if os.path.exists(log_file):
        try:
            df = pd.read_csv(log_file)
            new_row = report
            df = df.append(new_row, ignore_index=True)
            df.to_csv(log_file, index=False, encoding="utf_8_sig")
        except FileNotFoundError:
            print('日志更新失败')
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail='日志更新失败'
            )
    else:
        try:
            columns = ['log_title', 'ground_truth', 'predicted_ans', 'zh_ans',
                       'signal_acc', 'signal_time_used', 'translate_time_used',
                       'total_time', 'trail_time', 'dataset', 'algorithm',
                       'subject', 'ITR']
            df = pd.DataFrame([report], columns=columns)
            df.to_csv(log_file, index=False, encoding="utf_8_sig")
        except Exception as e:
            print('服务器出错，日志写入失败', e)
            # 如果发生其他错误，打印错误信息
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='服务器出错，日志写入失败'
            )

    print("本轮日志文件夹：", log_path)


def csv_to_json(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append(row)
    form_data = json.dumps(data, indent=2)
    return data


@logApp.post("/report")
async def write_report(report: Report):
    cur_report = {
        'log_title': report.log_title,
        'ground_truth': report.ground_truth,
        'predicted_ans': report.predicted_ans,
        'zh_ans': report.zh_ans,
        'signal_acc': report.signal_acc,
        'signal_time_used': report.signal_time_used,
        'translate_time_used': report.translate_time_used,
        'total_time': report.total_time,
        'trail_time': report.trail_time,
        'dataset': report.dataset,
        'algorithm': report.algorithm,
        'subject': report.subject,
    }
    write_log(cur_report)
    return {"cur_log": report.log_title, "info": "成功写入日志"}


@logApp.get("/log/{log_title}")
async def read_report(log_title: str):
    log_path = os.path.join(log_dir, log_title + '.csv')
    if not os.path.exists(log_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='未找到该日志'
        )
    else:
        data = csv_to_json(log_path)
    return {"log_list": data}


# if __name__ == "__main__":


