import requests
import json
from dotenv import load_dotenv

load_dotenv()
GPT_URL = os.environ.get('GPT_URL')
GPT_API_KEY = os.environ.get('GPT_API_KEY')


def get_llm_res(json_data: json):
    data = json.loads(json_data)
    answer = data['choices'][0]['message']['content']
    if answer:
        answer_dict = json.loads(answer)
        answer_content = answer_dict['answer']
        return answer_content
    else:
        return False


def do_request(predict: str):
    url = GPT_URL
    # 参考这篇文章可以引入openAI的chatgpt。但是需要合适的prompt限制。
    # 为了方便使用，我们在模拟的时候可以预设一些回答。
    # https://juejin.cn/post/7206249233115643959
    api_key = GPT_API_KEY
    prompt = """
        
    """
    content = prompt + predict
    data = {
        "model": "这里是你选的模型类型，不同公司的不太一样",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "stream": False
    }

    json_data = json.dumps(data)

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    response = requests.post(url, data=json_data, headers=headers, timeout=60)
    # return response

    if response.status_code == 200:
        print("请求成功！")
        # print("响应body:", response.text)
        # print(get_llm_res(response.text))
        # print("请求成功，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        return response
    else:
        # 请求失败日志
        print("请求失败，状态码:", response.status_code)
        return response
        # print("请求失败，body:", response.text)
        # print("请求失败，X-BC-Request-Id:", response.headers.get("X-BC-Request-Id"))
        # raise HTTPException(
        #     status_code=status.HTTP_502_BAD_GATEWAY,
        #     detail='服务不支持，请检查Token有效期或网络环境'
        # )

