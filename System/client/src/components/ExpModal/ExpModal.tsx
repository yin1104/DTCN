import React from 'react';
import { Button, Flex, Dialog, Strong } from '@radix-ui/themes';
import './expModal.css'

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

type ExpModalProps = {
  visibility: boolean,
  info?: ExpInfo,
  handleVis(): void
  handleRedict(): void
};

const ExpModal: React.FC<ExpModalProps> = (props) => {
  const { visibility, info, handleVis, handleRedict } = props

  const sendLog = () => {
    // 写入日志
    fetch("http://localhost:8081/log/report", {
      method: 'POST',
      headers: {
        'Content-type': 'application/json'
      },
      body: JSON.stringify(info)
    })
    .then(res => res.json())
    .then(json => console.log(json))
    .catch(err => console.log('err:', err))

    handleRedict()
  }

  return (
    <Dialog.Root open={visibility}>
      <Dialog.Content style={{ maxWidth: 450, padding: 36 }}>
        <Dialog.Title align="center" className='exp-modal-title'>恭喜完成一轮实验</Dialog.Title>
        <Dialog.Description size="2" mb="6" align="center" style={{lineHeight:2}}>
          日志标题：<Strong style={{color: '#b1a9ff'}}>{info?.log_title}</Strong><br/>
          目标任务：<Strong style={{color: '#b1a9ff'}}>{info?.ground_truth}</Strong><br/>
          预测结果：<Strong style={{color: '#b1a9ff'}}>{info?.predicted_ans}</Strong><br/>
          信号识别准确率：<Strong style={{color: '#b1a9ff'}}>{info?.signal_acc} %</Strong><br/>
          信号识别平均用时：<Strong style={{color: '#b1a9ff'}}>{info?.signal_time_used} s</Strong><br/>
          中文转译用时：<Strong style={{color: '#b1a9ff'}}>{info?.translate_time_used} s</Strong><br/>
          本轮实验总用时：<Strong style={{color: '#b1a9ff'}}>{info?.total_time} s</Strong><br/>
        </Dialog.Description>
        <Flex gap="3" mt="4" justify="center">
          <Button size="3" color='iris' style={{ width: 100, borderRadius: 40, fontSize: 14, cursor: 'pointer' }}  onClick={handleVis}>知道了</Button>
          <Button className='themeButton' size="3" style={{ width: 260, borderRadius: 40, fontSize: 14 }}  onClick={sendLog}>本轮实验结束，返回控制台</Button>
        </Flex>
      </Dialog.Content>
    </Dialog.Root>
  )
}
export default ExpModal;