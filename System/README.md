# 在线模拟项目

没搞Docker，以后再说

## client

1. 装一个nvm
2. 安装node版本18.17或以上
3. 装yarn【不装则 `npm i` ，最好装一下】
4. `cd client`
5. `yarn`
6. `yarn dev`
7. 运行成功

## server

1. python 3.7或以上
2. requirements.txt：主要是fastAPI，Pytorch， scikit-learn，pandas这些
3. 在主目录下新建一个`.env`文件，填空。主要是路径问题，如果没跑起来大概率是路径没搞对。

```txt
# .env文件
OPENAI_API_KEY="sk-your-key"
BACK_PORT=8081 # 服务端口
IS_PROD=True # 布尔值，是否开发环境，没啥关系
TSS_APPID= # 如果需要语音服务，去官网申请id，密码这些
TSS_APISecret=
TSS_APIKey=
TSS_URL=
MAT_FILE_PATH=  # 脑电信号mat文件地址
MODEL_DIR=  # 模型地址
GPT_URL=  # 如果需要大模型服务，同样也是去官方申请API
GPT_API_KEY=
```

## database

注意路径

### 浏览器

仅与以下浏览器版本配置兼容:

```text
Chrome >= 73
Firefox >= 78
Edge >= 79
Safari >= 12.0
iOS >= 12.0
opera >= 53
```

## 端口

```cmd
> netstat -ano|findstr "8081"  # 查看本地 8081端口占用情况
> taskkill /F /PID 17220 # 成功: 已终止 PID 为 17220 的进程。
```

解决 uvicorn 关闭服务后的Backgroud task线程占用问题

![](..\img\系统\1.png)
