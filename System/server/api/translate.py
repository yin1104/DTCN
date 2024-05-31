"""chatgpt翻译与tss语言服务"""
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from CNchat import GetTSS
import os
from pathname import AllPath
from CNchat.chatgptTest import do_request, get_llm_res
import requests

translateApp = APIRouter()


@translateApp.post('/tss/{content}')
async def tss(content: str):
    # 如果已经有语音了，就不用消耗wss了, 请求2ms, 用ws消耗 556ms
    audio_path = os.path.join(AllPath['_AUDIO'], content + '.mp3')
    if os.path.exists(audio_path):
        return {'audio_path': audio_path}
    else:
        GetTSS(content)

    return {'audio_path': audio_path}


@translateApp.post('/llm/{content}')
async def llm(content: str):
    llm_path = os.path.join(AllPath['_LLM'], content + '.txt')
    # 如果已经有这条记录了，就不用消耗LLM的token了
    if os.path.exists(llm_path):
        try:
            # 尝试打开文件
            with open(llm_path, 'r', encoding='utf-8') as file:
                translate_ans = file.read()
        except FileNotFoundError:
            # 如果文件不存在，打印错误信息
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail='数据库连接失败'
            )
        except Exception as e:
            # 如果发生其他错误，打印错误信息
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail='服务器出错'
            )
    else:
        response = do_request(content)
        if response.status_code == 200:
            res = get_llm_res(response.text)
            if res is not False:
                translate_ans = res
                # 存入本地
                with open(llm_path, 'w', encoding='utf-8') as file:
                    # 写入内容
                    file.write(res)
            else:
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail='翻译失败'
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail='服务不支持，请检查Token有效期或网络环境'
            )

    return {'translate': translate_ans}


@translateApp.post("/audio/{content}")
def get_audio(content: str):
    # 构建文件的完整路径
    audio_path = os.path.join(AllPath['_AUDIO'], content + '.mp3')
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail='未找到资源'
        )

    # 返回响应
    return FileResponse(audio_path, media_type="audio/mpeg")

