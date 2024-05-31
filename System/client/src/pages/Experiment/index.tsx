import { useEffect, useRef, useState } from 'react';
import Navigiter from '../../components/Navigiter/Navigiter';
import { useNavigate } from 'react-router-dom';
import { Button, Flex, Table, Container, Select, Code, Strong } from '@radix-ui/themes';
import Modal from '../../components/Modal/Modal';
import { DATASETINFO, ALGORITHM } from '../../constant';
import './experiment.css'

/**《能用就行》
 * https://github.com/radix-ui/primitives/discussions/831
 */
const defaultTableContent = [
  {
    ITR: "",
    algorithm: "",
    dataset: "",
    ground_truth: "",
    log_title: "",
    predicted_ans: "",
    signal_acc: "",
    signal_time_used: "",
    subject: "",
    total_time: "",
    trail_time: "",
    translate_time_used: "",
    zh_ans: ""
  },
]

const Experiment = () => {
  const allAlgorithms = ALGORITHM.EXIST_ALGORITHMS
  const allDataset = DATASETINFO.SSVEP_DATASET
  const preLogTitle = window.localStorage.getItem('OnlineLogTitle')
  const date = new Date()
  const defaultLogTitle = `${date.getFullYear()}${('0' + (date.getMonth() + 1)).slice(-2)}${('0' + date.getDate()).slice(-2)}`
  // const shouldlog = useRef<boolean>(true);
  const [curDataset, setCurDataSet] = useState<string>(allDataset[0].value)
  const [curSubject, setCurSubject] = useState<number>(1)
  const [curWindow, setCurWindow] = useState<number>(1)
  const [curSubNum, setCurSubNum] = useState<number>(allDataset[0].info._SUBJECTS)
  const [curAlgorithm, setCurAlgorithm] = useState<string>(allAlgorithms[0].name)
  const [curStr, setCurStr] = useState<string>('')
  const [curLogTitle, setCurLogTitle] = useState<string>(defaultLogTitle)
  const [showModalSub, setShowModalSub] = useState<boolean>(false)
  const [showModalTarget, setShowModalTarget] = useState<boolean>(false)
  const [list, setList] = useState(defaultTableContent)

  const navigate = useNavigate()

  const doChooseDataset = (e: { target: { value: string; }; }) => {
    setCurDataSet(e.target.value)

    for (const dataset of allDataset) {
      if (dataset.value === e.target.value) {
        setCurSubNum(dataset.info._SUBJECTS)
      }
    }
  }

  const doChooseAlgorithm = (e: { target: { value: string; }; }) => {
    setCurAlgorithm(e.target.value)
  }

  const doSetSubject = (e: { target: { value: string; }; }) => {
    const sub = Number(e.target.value)
    setCurSubject(sub)
  }

  const doSetWindow = (e: { target: { value: string; }; }) => {
    const sub = Number(e.target.value)
    setCurWindow(sub)
  }

  const doSetTarget = (e: { target: { value: string; }; }) => {
    // 这里需要合法性检测输入的目标字段合法性
    setCurStr(e.target.value)
  }

  const doSetLogTitle = (e: { target: { value: string; }; }) => {
    setCurLogTitle(e.target.value)
  }

  const doCloseModal = () => {
    if (showModalSub) {
      setShowModalSub(false)
    }
  }

  const doCloseModalTarget = () => {
    if (showModalTarget) {
      setShowModalTarget(false)
    }
  }

  const downloadLog = () => {
    window.alert('累得慌偷懒没写，自己去文件夹找')
  }

  const handleExp = () => {
    let pathName = `/ssvep?title=${curLogTitle}`

    if (!Number.isInteger(curSubject) || curSubject < 1 || curSubject > curSubNum) {
      setShowModalSub(true)
      return
    }
    if (curStr.length === 0) {
      setShowModalTarget(true)
      return
    }
    for (const dataset of allDataset) {
      if (dataset.value === curDataset) {
        const evalStr = dataset.info._LOW_TARGETS
        if (curStr.split('').filter(str => evalStr.includes(str)).length !== curStr.length) {
          // console.log('--> 不合法的目标', curStr)
          setShowModalTarget(true)
          return
        }
      }
    }

    // 提交之后保存本次实验日志标题到本地，确保一致性
    window.localStorage.setItem('OnlineLogTitle', curLogTitle)
    // 提交时间作为开始时间
    const startTime = new Date()
    window.localStorage.setItem('startTime', startTime.getTime().toString())

    const experimentInfo = {
      stimarr: curStr,
      dataset: curDataset,
      subject: curSubject,
      alg: curAlgorithm,
      curWindow: curWindow,
    }

    // 这里的路径处理后期需要 refacter到 utils工具函数中
    Object.entries(experimentInfo).forEach((entry) => {
      const [key, value] = entry;
      pathName += (`&${key}=${value}`);
    })

    // 路由跳转
    navigate(pathName)
  }


  // useEffect(() => {
  //   // 清空上一轮实验记录
  //   const preTargetRecord = window.localStorage.getItem('OnlineTargetRecord')
  //   if (preTargetRecord) {
  //     window.localStorage.removeItem('OnlineTargetRecord')
  //   }
  //   const last = window.localStorage.getItem('OnlineExpTime')
  //   if (last) {
  //     window.localStorage.removeItem('OnlineExpTime')
  //   }
  // }, [])

  useEffect(() => {
    // 清空上一轮实验记录
    const preTargetRecord = window.localStorage.getItem('OnlineTargetRecord')
    if (preTargetRecord) {
      window.localStorage.removeItem('OnlineTargetRecord')
    }
    const last = window.localStorage.getItem('OnlineExpTime')
    if (last) {
      window.localStorage.removeItem('OnlineExpTime')
    }

    async function askLog() {
      fetch(`http://localhost:8081/log/log/${curLogTitle}`, {
        method: 'GET',
        headers: {
          'Content-type': 'application/json'
        },
      })
      .then(res => res.json())
      .then(content => {
        setList(content?.log_list)
        console.log(content?.log_list)
      })
      .catch(err => console.log('err:', err))
    }
    // if (shouldlog.current) {
    //   shouldlog.current = false;
    //   askLog()
    // }
    askLog()
  }, [curLogTitle])

  return (
    <div className='bg'>
      <Navigiter curPage={3} />
      <div className='exp-content'>
        {/* 实验设置 */}
        <Container size="1" className='exp-container'>
          <Flex gap="3" justify="center" direction="column" align="center">
            <div className='exp-step bigsize'>
              <div className='exp-step-title stepbig'>
                <Strong>STEP 1. 选择数据集</Strong>
              </div>
              <div className='exp-step-option'>
                <Select.Root
                  defaultValue={allDataset[0].value}
                  onValueChange={(value) => doChooseDataset({ target: { value } })}
                >
                  <Select.Trigger variant="soft" className='exp-step-select' />
                  <Select.Content >
                    {
                      allDataset.map((item) => {
                        return (
                          <Select.Item value={item.value} key={item.name}>{item.name}</Select.Item>
                        )
                      })
                    }
                  </Select.Content>
                </Select.Root>
              </div>
              <div className='exp-step-title'>
                <div> 共 <Code>{curSubNum}</Code> 名受试者，选择 Subject：
                  <input className='exp-step-input' max={35} min={1} defaultValue={1} type="number" onChange={doSetSubject} />
                </div>
              </div>
            </div>
            <div className='exp-step'>
              <div className='exp-step-title'>
                <Strong>STEP 2. 选择算法</Strong>
              </div>
              <div className='exp-step-option'>
                <Select.Root
                  defaultValue={allAlgorithms[0].name}
                  onValueChange={(value) => doChooseAlgorithm({ target: { value } })}
                >
                  <Select.Trigger variant="soft" className='exp-step-select' />
                  <Select.Content>
                    {
                      allAlgorithms.map((item) => {
                        return (
                          <Select.Item value={item.name} key={item.name}>{item.name}</Select.Item>
                        )
                      })
                    }
                  </Select.Content>
                </Select.Root>
              </div>
              <div className='exp-step-title'>
                <div> 若固定窗(非DW)，则选择时长：
                  <input className='exp-step-input' max={1.5} min={0.5} defaultValue={1.0} step={0.1} type="number" onChange={doSetWindow} />s
                </div>
              </div>
            </div>
            <div className='exp-step'>
              <div className='exp-step-title'>
                <Strong> STEP 3. 目标字段</Strong>
              </div>
              <div className='exp-step-option'>
                <input className='exp-step-input exp-text' type="text" placeholder='输入范围：a-z 0-9 _ , . <' onChange={doSetTarget} />
              </div>
            </div>
            <div className='exp-step'>
              <div className='exp-step-title'>
                <Strong> STEP 4. 本次实验日志名称</Strong>
              </div>
              <div className='exp-step-option'>
                <input className='exp-step-input exp-text' type="text" defaultValue={preLogTitle ?? defaultLogTitle} placeholder='所有实验记录会生成在该名称命名日志下' onChange={doSetLogTitle} />
              </div>

            </div>
            <Button className='themeButton' size="3" style={{ width: 360, borderRadius: 40 }} onClick={handleExp}>确认提交，开启实验</Button>
            {/* <Button className='themeButton' size="3" style={{ width: 360, borderRadius: 40 }} onClick={getModel}>加载模型</Button> */}
            <Modal
              visibility={showModalSub}
              title='实验设计错误'
              info={`受试者序号为整数，当前数据集合法区间为 S1-S${curSubNum}，请重新输入`}
              handleVis={doCloseModal}
            />
            <Modal
              visibility={showModalTarget}
              title='实验设计错误'
              info={`目标字符串为空或不在实验范式内，字符范围：a-z 0-9 _ , . <`}
              handleVis={doCloseModalTarget}
            />
          </Flex>
        </Container>
        {/* 日志表单 */}
        <Container size="4" className='exp-container'>
          <Flex justify="center" direction="column" align="center">
            <Flex justify="between" className='exp-log-title'>
              <div><Strong>日志报告：</Strong>{curLogTitle}</div>
              <div className='exp-log-button' onClick={downloadLog}>下载详细日志</div>
            </Flex>
            <div>
              <Table.Root variant='surface' style={{
                width: 1136
              }}>
                <Table.Header>
                  <Table.Row>
                    <Table.ColumnHeaderCell>序号</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>数据集</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>受试者</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>目标输入</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>当前算法</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>识别结果</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>中文转译</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>识别用时</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>转译用时</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>实验用时</Table.ColumnHeaderCell>
                    <Table.ColumnHeaderCell>ITR</Table.ColumnHeaderCell>
                  </Table.Row>
                </Table.Header>
                {list?.length > 0 && (
                  <Table.Body>
                    {
                      list?.map((item, index) => {
                        return (<Table.Row key={index}>
                          <Table.RowHeaderCell>{index}</Table.RowHeaderCell>
                          <Table.Cell>{item?.dataset}</Table.Cell>
                          <Table.Cell>{item?.subject}</Table.Cell>
                          <Table.Cell>{item?.ground_truth}</Table.Cell>
                          <Table.Cell>{item?.algorithm}</Table.Cell>
                          <Table.Cell>{item?.predicted_ans}</Table.Cell>
                          <Table.Cell>{item?.zh_ans}</Table.Cell>
                          <Table.Cell>{item?.signal_time_used} s</Table.Cell>
                          <Table.Cell>{item?.translate_time_used} s</Table.Cell>
                          <Table.Cell>{item?.total_time} s</Table.Cell>
                          <Table.Cell>{parseFloat(item?.ITR).toFixed(2)} bits/min</Table.Cell>
                        </Table.Row>)
                      })
                    }
                  </Table.Body>
                )}
              </Table.Root>
            </div>
          </Flex>
        </Container>
      </div>
    </div>
  )
}
export default Experiment;