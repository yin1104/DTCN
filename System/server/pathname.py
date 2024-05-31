import os
current_dir = os.getcwd()
# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
parent_dir = ""  # 主目录

# 语音文件夹 D:\毕设相关\000应用项目\database\audio\[独一无二的名称:时间+内容].mp3
audio_dir = os.path.join(parent_dir, "database", "audio")
llm_dir = os.path.join(parent_dir, "database", "llm")
ssvep_dir = os.path.join(parent_dir, "database", "ssvep")
log_dir = os.path.join(parent_dir, "database", "log")
model_dir = os.path.join(parent_dir, "database", "model")

AllPath = {
    '_AUDIO': audio_dir,
    '_SSVEP': ssvep_dir,
    '_LOG': log_dir,
    '_MODEL': model_dir,
    '_LLM': llm_dir
}

