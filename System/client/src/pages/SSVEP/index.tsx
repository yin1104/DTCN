import { useCallback, useEffect, useRef, useState } from 'react';
import { Flex, Text, Button } from '@radix-ui/themes';
import { useNavigate, useLocation } from 'react-router-dom';
import { VideoIcon } from '@radix-ui/react-icons'
import { useFullScreen } from '../../hooks/useFullScreen';
import { parseUrlParams, calAcc, getDate } from '../../utils';
import { DATASETINFO } from '../../constant';
import SSVEPStimBlock from '../../components/StimBlock/StimBlock';
import ExpModal from '../../components/ExpModal/ExpModal';
import toast from "react-hot-toast";
import './ssvep.css'

type ExpInfo = {
  log_title: string,
  ground_truth: string,
  predicted_ans: string,
  zh_ans: string,
  signal_acc: string,
  signal_time_used: string,
  translate_time_used: string,
  total_time: string,
  trail_time: string,
  dataset: string,
  algorithm: string,
  subject: string,
}

type ExperimentInfoProps = {
  title: string,
  stimarr: string,
  dataset: string
  subject: string,
  alg: string,
  curWindow: string,
}

type wsRes = {
  letter: string,
  time: string
}
const defaultLetter = ''
/**
 * 当前页面流程： 路由传递信息【数据集，subject，文本，算法】，toast提示，再次确认开始，点击确认后，先等1秒再开始
 */
const SSVEPonLine = () => {
  // const { curSSVEPTarget } = props // 等之后提到组件再说，先做页面版本

  const navigate = useNavigate()
  const curPath = useLocation().search
  const experimentInfo = parseUrlParams(curPath) as ExperimentInfoProps
  const { stimarr, dataset, subject, alg, title, curWindow } = experimentInfo
  // {
  //   title: '20240125',
  //   stimarr: 'wxshaitaiy',
  //   dataset: 'benchmark',
  //   subject: '1',
  //   alg: 'flexEEG'
  // }

  const stimList = DATASETINFO.BenchmarkInfo._TARGETS
  const stimFreqList = DATASETINFO.BenchmarkInfo._FREQS
  const stimPhaseList = DATASETINFO.BenchmarkInfo._PHASES
  const preTargetRecord = window.localStorage.getItem('OnlineTargetRecord') ?? defaultLetter
  const [curTarget, setCurTarget] = useState<string>(preTargetRecord)
  const [allState, setAllState] = useState<string>('rest')
  const [zhRes, setZHRes] = useState<string>('')
  const [tssRes, setTSSRes] = useState<string>('') // TSS语音路径
  const shouldlog = useRef<boolean>(true); // 由于StrictMode模式下，组件会额外渲染一次，用来确保只渲染一次
  const [showModal, setShowModal] = useState<boolean>(false)
  const [expReport, setExpReport] = useState<ExpInfo>()
  const [isPlaying, setIsPlaying] = useState(false) // 点击控制播放音频

  const audioRef = useRef<HTMLAudioElement>(null)

  const isExit = () => {
    if (preTargetRecord.length < stimarr.length) {
      toast.error("中断实验将不被计入结果，重定向至工厂");
    }

    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
    navigate('/experiment')
  }

  const doCloseModal = () => {
    if (showModal) {
      // navigate('/experiment')
      setShowModal(false)
    }
  }

  const doRedict = () => {
    doCloseModal()
    window.localStorage.removeItem('startTime')
    window.localStorage.removeItem('OnlineExpTime')

    navigate('/experiment')
  }

  // 临时处理： State目标
  const wait = async (state: string, time: number) => {
    const res = await new Promise(
      resolve => setTimeout(() => resolve(state), time)
    );
    return res;
  }

  // 获取音频地址
  async function getAudio(content: string) {
    const url = `http://localhost:8081/translate/tss/${encodeURIComponent(content)}`
    let radio_path = ''
    try {
      const response = await fetch(url, {
        method: 'post',
        headers: {
          'Content-type': 'application/json'
        }
      });
      const res = await response.json();
      radio_path = res?.audio_path
    } catch (error) {
      console.log('Request Failed', error);// 丢一个ALERT
    }
    return radio_path
  }

  // 获取llm结果
  async function getLLM(content: string) {
    const url = `http://localhost:8081/translate/llm/${encodeURIComponent(content)}`
    let llm = ''
    try {
      const response = await fetch(url, {
        method: 'post',
        headers: {
          'Content-type': 'application/json'
        }
      });
      const res = await response.json();
      llm = res?.translate
    } catch (error) {
      console.log('Request Failed', error);  // 处理下报错
    }
    return llm
  }

  // 单个字母预测
  const promiseSocket = async (url: string, target_letter: string, subject: string, alg: string, window: number) => {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(url);

      ws.onopen = () => {
        console.log('---> 单个字母请求')
        ws.send(JSON.stringify({
                  Mission: target_letter,
                  Dataset: 'benchmark',
                  Algorithm: alg, 
                  Subject: subject,
                  WindowSize: window,
                }))
      };

      // 监听error事件
      ws.onerror = (error) => {
        reject(error);
      };

      // 监听close事件
      ws.onclose = (event) => {
        if (event.wasClean) {
          // 正常关闭
          console.log('WebSocket connection closed cleanly');
        } else {
          // 非正常关闭
          console.log('WebSocket connection was closed uncleanly');
        }
        console.log(`Code: ${event.code}, Reason: ${event.reason}`);
        if (event.code === 1003) {
          window.alert(`Code: ${event.code}, Reason: ${event.reason}`);
          doRedict();
        }
      };

      // 监听message事件
      ws.onmessage = (event) => {
        console.log('Received data from server:', event.data);
        const info = event.data?.split('_')
        const cur: wsRes = {
          letter: '',
          time: ''
        }
        if (info[0] === "PredictFin") {
          cur['letter'] = info[1];
          cur['time'] = info[2];
          // resolve(info[1]);
          resolve(cur)
          ws.close();
        }
      };
    });
  }

  const doTmpStim = async () => {
    if (allState !== 'stim') {
      // setCurTarget(defaultLetter)
      setAllState('stim')
      // 再次识别之前将目标清空
      // setCurTarget('-1')
    } else {
      window.location.reload()
    }
  }

  const playAudio = useCallback(() => {
    console.log('----》点击音频', isPlaying)
    isPlaying ? audioRef.current?.pause() : audioRef.current?.play().catch(() => setIsPlaying(false))
    if (isPlaying) {
      setIsPlaying(!isPlaying)
    }

  }, [isPlaying])

  // 初始化目标为上一次的识别结果
  useEffect(() => {

    // 定长刺激获取结果
    async function waitALG() {
      // 上一次预测的字母
      const preTargetRecord = window.localStorage.getItem('OnlineTargetRecord') ?? defaultLetter
      // 标红上一次识别的目标字符
      setCurTarget(preTargetRecord)
      // console.log('---> preTarget', preTargetRecord.slice(-1))
      let expSingalTime = +(window.localStorage.getItem('OnlineExpTime') ?? 0)
      
      if (preTargetRecord.length < stimarr.length) {
        let algRes = defaultLetter;
        
        await wait('先等着', 1000) // 1秒确认结果
        setCurTarget(defaultLetter)

        await wait('找下一个字母', 3000) // 3秒给用户找到目标字符准备注视,5s有点长了，看别人视频真的飞快

        doTmpStim() // 这里发送请求，同时开始刺激

        // algRes: 接口传来的预测字母
        // algRes = await wait(stimarr.substr(preTargetRecord.length, 1), 2000) as string
        const algAllRes = await promiseSocket(`ws://localhost:8081/ssvep/model`, stimarr.substr(preTargetRecord.length, 1), subject, alg, (+curWindow)) as wsRes
        algRes = algAllRes['letter'].toLowerCase();
        expSingalTime += (+algAllRes['time'])

        // 更新序号
        window.localStorage.setItem('OnlineTargetRecord', preTargetRecord + algRes)
        window.localStorage.setItem('OnlineExpTime', expSingalTime.toFixed(1))
        // 终止本轮刺激
        location.replace(location.href) // 这个闪烁不会很严重
        // window.location.reload(true)
      }
      let translateCostTime = '0'
      if (preTargetRecord.length === stimarr.length) {
        // console.log('---> 此时先请求chatgpt', preTargetRecord)
        // 从优化的思路，应该后端一条路websocket请求一条龙
        /**
         * !!! 不需要语音翻译就把这里注释掉就行，然后后面的日志可能要修改一下，有问题直接Debug
         */
        const translateStart = new Date()
        const chatgptRes = await getLLM(preTargetRecord.replace(/[^a-zA-Z0-9]/g, ''))
        setZHRes(chatgptRes)

        // const tssUrlRes = await wait('本地语音地址', 2000) as string
        const tssUrlRes = await getAudio(chatgptRes)
        if (tssUrlRes) {
          const url = 'http://localhost:8081/translate/audio/' + chatgptRes
          fetch(url,
            {
              mode: 'cors',
              method: 'POST',
              // redirect: 'follow',
              // headers: {
              //   'Content-type': 'text/plain',
              //   "Access-Control-Allow-Origin": "*",
              //   // 'Accept': '*/*',
              // }
            }
          )
            .then(response => response.blob())
            .then(blob => {
              const audioUrl = URL.createObjectURL(blob);
              audioRef.current!.src = audioUrl
              audioRef.current?.play().catch(() => setIsPlaying(false))
              setTSSRes(tssUrlRes)
            })
            .catch(console.error);
        }

        // 实验总时长
        const endTime = new Date()
        const startTime = +(window.localStorage.getItem('startTime') ?? endTime.getTime())
        const expTime = ((endTime.getTime() - startTime) / 1000).toFixed(2)
        translateCostTime = ((endTime.getTime() - translateStart.getTime()) / 1000).toFixed(2)

        // 目标准确率
        const expAcc = calAcc(stimarr, preTargetRecord)
        // 获取报告
        const logInfo = {
          log_title: title,
          ground_truth: stimarr,
          predicted_ans: preTargetRecord,
          zh_ans: chatgptRes,
          signal_acc: expAcc,
          signal_time_used: (expSingalTime/10).toFixed(2),
          translate_time_used: translateCostTime,
          total_time: expTime,
          trail_time: getDate(),
          dataset: dataset,
          algorithm: alg,
          subject: 'S'+ subject,
        }

        setExpReport(logInfo)
        setShowModal(true)
      }
    }

    // 只执行一次
    if (shouldlog.current) {
      shouldlog.current = false;
      console.log('---> 服了')
      waitALG()
    }
  }, [])

  return (
    <div className='bg ss'>
      <Text className='sstitle'>Benchmark</Text>
      <div className='speller-res'>
        <div className='speller-letter-res'>
          <div className='speller-letter-input'>
            {/* 这里的结果从localStorage中获取， 每次reload更新页面 */}
            {/* <p>{stimList[curTarget].toLowerCase()}</p> */}
            <p>{preTargetRecord}</p>
          </div>
          <div className='speller-chatgpt-res'>
            {/* 有颜色就是说明tss完成，不然就是#a2a2a2
                这里的文字结果从接口处获取 chatgpt 结果
                语言变色结果从 讯飞获取 tss 结果
            */}
            <p style={{ color: '#c48bf3' }}>{zhRes}</p>
          </div>
          <div className='speller-tss-res'>
            {zhRes && (<VideoIcon width={24} height={24} color={tssRes ? '#c48bf3' : '#a2a2a2'} onClick={playAudio} />)}
          </div>
          <div className='speller-tss-res' style={{ width: 100 }}>
            <audio ref={audioRef} />
          </div>
        </div>
        {/* 语音部分：单字母识别，由于刺激刷新需要将本次结果存在localStorage中，
        识别结束调用chatgpt和讯飞，需要先模拟一下ws,然后组合调通后两者语言模型 */}
      </div>
      <div className='benchmark'>
        {
          stimList.map((item, index) => {
            return (
              <SSVEPStimBlock
                stimTarget={item}
                key={index}
                stimIndex={index}
                stimFreq={stimFreqList[index]}
                stimPhase={stimPhaseList[index]}
                onType={true}
                // isTarget={curTarget === index ? 'target' : 'rest'}
                isTarget={curTarget.slice(-1) === stimList[index].toLocaleLowerCase() ? 'target' : allState}
              // needReload={stimToRest}
              // needLateReload={stimToTarget}
              />
            )
          })
        }
      </div>
      <Flex gap='2' align="end" justify="end" style={{ width: '95%' }}>
        <Button onClick={useFullScreen} variant='surface' style={{ width: 80, opacity: 0.5 }}>全屏</Button>
        <Button onClick={isExit} variant='surface' style={{ width: 80, opacity: 0.5 }}>退出</Button>
      </Flex>
      <ExpModal
        visibility={showModal}
        info={expReport}
        handleVis={doCloseModal}
        handleRedict={doRedict}
      />
    </div>
  )
}
export default SSVEPonLine;