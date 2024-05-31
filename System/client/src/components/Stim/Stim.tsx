import React, { MutableRefObject, useRef, useCallback } from 'react';
import './stim.css'

type StimProps = {
  stimWidth?: number, // 默认正方形刺激，长方形要多些蛮多，先偷懒
  stimHeight?: number,
  stimTarget?: string,
  onType?: boolean,
  curOpacity?: number,
  innerWidth?: number, // 内部刺激小块的宽度，应该需要计算公式，先偷懒默认的 8*8
  innerNums?: number,
  isTarget?: string, // 当前状态
  needReload?: boolean,
  needLateReload?: boolean,
};

type defaultColor = {
  backColor: string,
  initStimColor: string,
  fontColor: string
};

const defaultStimProps: StimProps = {
  stimWidth: 136,
  stimHeight: 136,
  stimTarget: 'A',
  onType: false,
  curOpacity: 1,
  innerNums: 8,
  innerWidth: 20,
  isTarget: 'rest',
  needReload: false,
  needLateReload: false,
}

// Off型：低层浅色不闪烁，深色小格子闪烁，亮度范围：0.361-0.5
// 灰度值：0.5，0 代表黑，1 代表白
const defaultOffOpacity: defaultColor = {
  backColor: '#808080',  // 低层是 0.5的灰度值，其实用 hsb更好，
  initStimColor: '#5C5C5C',
  fontColor: '#808080'
}

const defaultOnOpacity: defaultColor = {
  backColor: '#222',  // 低层是 0.21的灰度值，其实用 hsb更好，
  initStimColor: '#e6e6e6', // 0.67
  fontColor: '#000'
}

const defaultTargetOpacity: defaultColor = {
  backColor: '#AA1616',  // 红色目标态
  initStimColor: '#AA1616',
  fontColor: 'black'
}

const SSVEPStim = React.memo((props: StimProps) => {
  const { stimWidth, onType, stimTarget, curOpacity, innerNums, isTarget, needReload, needLateReload } = props
  const innerBlockNum = innerNums ?? defaultStimProps.innerNums ?? 8
  const innerBlock = []

  for (let i = 0; i < (innerBlockNum) ** 2; i++) {
    innerBlock.push(i + 1)
  }

  let colorTheme = defaultOffOpacity

  if (onType) {
    colorTheme = defaultOnOpacity
  }

  // 闪烁刺激，但应该是成片的，因此需要更大的组件去封装，此部分需要抽离
  const shiningBlockRef = useRef() as MutableRefObject<HTMLDivElement>
  const askframe = window.requestAnimationFrame
  let count = 0;
  const FREQ = 12
  const PHASE = 0
  const STIM_ARR: Array<string> = []

  for (let i = 0; i < 4*60; i++) {
    const tmp = (1 + Math.sin(2 * Math.PI * FREQ * i / 60 + PHASE)) / 2
    STIM_ARR.push(tmp.toFixed(2))
  }

  const step = useCallback(() => {
    // !startTime && (startTime = currentTime)
    count += 1;
    
    if (count < 60 * 4 && isTarget === 'stim') {
      shiningBlockRef.current.style.opacity = STIM_ARR[count-1]
      console.log('---> count', STIM_ARR[count-1])
      askframe(step)
    } else {
      shiningBlockRef.current.style.opacity = '1'
      count = 0
      console.log('停止')
    }
  }, [isTarget])

  // 根据状态控制，但需要额外封装大实验范式， 尤其是reload，只能reload一个,需要重构
  React.useEffect(() => {
    if (needReload) {
      // window.location.reload() // 重新加载强制中断
      window.location.href = './stim'
    }
  }, [needReload])

  React.useEffect(() => {
    if (needLateReload) {
      setTimeout(() => {
        // window.location.reload() // 目标态等待1秒再刷新,不能立即中断动画，但可以控制颜色一致而没有闪烁感
        window.location.href = './stim'
        /**
         * 使用location.reload() 会刷新页面，刷新页面时页面所有资源（css，js，img等）会重新请求服务器；
         * 建议使用location.href=“当前页url” 代替location.reload() ，使用location.href 浏览器会读取本地缓存资源
         */
      }, 1000)
    }
  }, [needLateReload])

  React.useEffect(() => {
    if (isTarget === 'stim') {
      askframe(step)
    }
    console.log('---> 当前状态', isTarget)
  }, [askframe, isTarget, step])

  return (
    <div className='stim'
      style={{
        width: stimWidth ?? defaultStimProps.stimWidth,
        height: stimWidth ?? defaultStimProps.stimWidth,
        background: isTarget === 'target' ? defaultTargetOpacity.backColor : colorTheme.backColor,
      }}>
      <div className='stim-target-text'
        style={{
          color: colorTheme.fontColor,
          fontSize: 40,
          zIndex: 3,
          fontWeight: 500,
        }}
      >
        {stimTarget ?? defaultStimProps.stimTarget}
      </div>
      <div className='stim-block-container'
        style={{
          width: stimWidth ?? defaultStimProps.stimWidth,
          height: stimWidth ?? defaultStimProps.stimWidth,
        }}>
        <div className='stim-block-grid-tmp'
          style={{
            // background: isTarget ? defaultTargetOpacity.initStimColor : colorTheme.initStimColor,
            opacity: curOpacity ?? defaultStimProps.curOpacity
          }}
          ref={shiningBlockRef}
        >
          {
            innerBlock.map((index) => {
              return <div
                key={index}
                className='stim-block-item'
                style={{
                  background: isTarget === 'target' ? defaultTargetOpacity.initStimColor : colorTheme.initStimColor,
                  // opacity: curOpacity ?? defaultStimProps.curOpacity
                }}
              />
            })
          }
        </div>
      </div>
    </div>
  )
})
export default SSVEPStim;