# Load environment variables first
from dotenv import load_dotenv
import uvicorn
import os
import sys
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from starlette.requests import Request
from starlette.responses import Response

from api import translateApp, ssvepApp, logApp
import signal

# import openai

load_dotenv()
IS_PROD = os.environ.get("IS_PROD", False)  # 是否是开发环境

app = FastAPI(
    # openapi_url=None,
    docs_url='/docs',  # 开发环境则保留 Swagger 文档
    title='SSVEP-BCI 在线仿真中文语音拼写系统',
    description='Author: Xinyi Jiang; Email: shawnchiang1104@gmail.com; School: Southeast University',
    version='1.0.1',
)


async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception:
        # you probably want some kind of logging here
        return Response("Internal server error", status_code=500)


app.middleware('http')(catch_exceptions_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # http://localhost:8083, http://localhost:5173
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get_status():
    return HTMLResponse(
        content="<h3>SMART SSVEP后台服务已成功启动. "
                "请启动前端服务地址 (default: http://localhost:5173).</h3> "
    )


@app.on_event("shutdown")
def shutdown_event():
    # 模拟关闭时需要处理的操作
    print("Performing clean shutdown...")
    print("Closing database connection...")
    print("Releasing resources...")


def handle_shutdown(signum, frame):
    sys.exit(0)


app.include_router(translateApp, prefix='/translate', tags=['语言模型'])
app.include_router(ssvepApp, prefix='/ssvep', tags=['信号处理'])
app.include_router(logApp, prefix='/log', tags=['日志管理'])

# 启动命令
if __name__ == '__main__':
    # host不要写成 0.0.0.0
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    uvicorn.run('main:app', host='localhost', port=8081, reload=True, workers=3)
