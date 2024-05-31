import { useState } from 'react';
import Navigiter from '../../components/Navigiter/Navigiter';
import { Flex, Grid, Box, Button, Text } from '@radix-ui/themes';
import SSVEPStim from '../../components/Stim/Stim';
import getScreenFps from '../../utils/getScreenFps';
import './stim.css'


const Stim = () => {
  const [curfps, setFPS] = useState(0)
  const [type, setType] = useState('off')
  const [curState, setCurState] = useState('rest')
  const [stimToRest, setStimToRest] = useState(false) // 只有从刺激态到非刺激态才需要重新reload
  const [stimToTarget, setStimToTarget] = useState(false) // 目标态需要提示1秒再重新reload

  const doGetFPS = () => {
    getScreenFps().then((fps) => {
      setFPS(fps as number)
      // console.log("当前屏幕刷新率为", fps);
      // 当前屏幕刷新率为 61.50061500615006
    });
  }

  const doChangeTypeOn = () => {
    if (type !== 'on') {
      setType('on')
    } 
  }

  const doChangeTypeOff = () => {
    if (type !== 'off') {
      setType('off')
    }   
  }

  const doRest = () => {
    if (curState !== 'rest') {
      if (curState === 'stim') {
        setStimToRest(true) // 刺激到休息，需要重新加载
      }
      setCurState('rest')
    }
  }

  const doStim = () => {
    if (curState !== 'stim') {
      setCurState('stim')
    }
  }

  const doTarget = () => {
    if (curState !== 'target') {
      if (curState === 'stim') {
        setStimToTarget(true) // 刺激到目标，需要重新加载
      }
      setCurState('target')
    }
  }

  const defaultStimProps = {
    stimWidth: 136,
    stimHeight: 136,
    stimTarget: 'C',
    onType: false,
    curOpacity: 1,
    innerNums: 8,
    innerWidth: 20,
  }
  return (
    <div className='bg'>
      <Navigiter curPage={2} />
      <Grid columns="2" gap="3" style={{ width: 750 }} justify="center">
        <Flex direction="column" gap="5">
          <Box height="6">
            <Flex direction="row" gap="5" style={{ width: 320, height: 44 }}>
              <div className={`${type === 'on' ? 'themeButton btn': 'themeButton btn not'}`} style={{ width: 144 }} onClick={doChangeTypeOn}> ON 型刺激</div>
              <div className={`${type === 'off' ? 'themeButton btn': 'themeButton btn not'}`} style={{ width: 144 }} onClick={doChangeTypeOff}> OFF 型刺激</div>
            </Flex>
          </Box>
          <Box grow="1">
            <Flex justify="center" align="center" style={{ background: 'black', width: 320, height: 320 }}>
              <SSVEPStim 
                stimTarget={defaultStimProps.stimTarget}
                isTarget={curState}
                // curOpacity={1.0}
                needReload={stimToRest}
                needLateReload={stimToTarget}
                onType={true}
              />
            </Flex>
          </Box>
        </Flex>

        <Flex direction="column" gap="5">
          <Box grow="1">
            <Flex gap="4" direction="column" justify="between" style={{ width: 360, height: 250, opacity: 0.8 }}>
              <Flex gap="4" align="center">
                <Text size="5" color='iris'>单个刺激模拟</Text><br />
                <div style={{ width: 80, height: 50 }}/>
              </Flex>
              <Flex gap="4" direction="column" className='stim-options'>
                <div>测试你的频率刷新率：<Button onClick={doGetFPS}>{curfps.toFixed(2)}Hz</Button></div>
                <div>选择合法刺激频率：<input className='exp-step-input' style={{width: 50}} max={15.8} min={8} step={0.2} defaultValue={12} type="number" /></div>
                <div>选择合法刺激相位：<input className='exp-step-input' style={{width: 50}} max={1.5} min={0} step={0.5} defaultValue={0} type="number" /></div>
                <div>选择刺激目标内容：<input className='exp-step-input' style={{width: 50}} type="text" defaultValue={'A'}/></div>
              </Flex>
            </Flex>
          </Box>
          <Box height="9">
            <Flex direction="row" gap="5" style={{ width: 360, height: 44 }}>
              <div className={`${curState === 'rest' ? 'themeButton btn': 'themeButton btn not'}`} onClick={doRest}>休息态</div>
              <div className={`${curState === 'stim' ? 'themeButton btn': 'themeButton btn not'}`} onClick={doStim}>刺激态</div>
              <div className={`${curState === 'target' ? 'themeButton btn': 'themeButton btn not'}`} onClick={doTarget}>结果态</div>
            </Flex>
          </Box>
        </Flex>
      </Grid>
    </div>
  )
}
export default Stim;