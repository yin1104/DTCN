import React, { MutableRefObject, useRef, useCallback  } from 'react';
import './stimBlock.css'

type StimProps = {
  stimWidth?: number, // 默认正方形刺激，长方形要多些蛮多，先偷懒
  stimHeight?: number,
  stimTarget?: string,
  onType?: boolean,
  // curOpacity?: number,
  innerWidth?: number, // 内部刺激小块的宽度，应该需要计算公式，先偷懒默认的 8*8
  innerNums?: number,
  isTarget?: string, // 当前状态
  needReload?: boolean,
  needLateReload?: boolean,
  stimFreq: number,
  stimPhase: number,
  stimIndex: number
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
  // curOpacity: 1,
  // innerNums: 10,
  innerNums: 16,
  innerWidth: 20,
  isTarget: 'rest',
  needReload: false,
  needLateReload: false,
  stimFreq: 12,
  stimPhase: 0,
  stimIndex: 0,
}

// Off型：低层浅色不闪烁，深色小格子闪烁，亮度范围：0.361-0.5
// 灰度值：0.5，0 代表黑，1 代表白
const defaultOffOpacity: defaultColor = {
  backColor: '#808080',  // 低层是 0.5的灰度值，其实用 hsb更好，'#808080'
  initStimColor: '#5C5C5C',
  fontColor: '#808080'
}

const defaultOnOpacity: defaultColor = {
  backColor: '#1d1d1d',  // 低层是 0.21的灰度值，其实用 hsb更好，'#363636'
  initStimColor: '#f2f2f2', // 0.67 '#AAAAAA'
  fontColor: 'black'
}

const defaultTargetOpacity: defaultColor = {
  backColor: '#AA1616',  // 红色目标态
  initStimColor: '#AA1616',
  fontColor: 'black'
}

const SSVEPStimBlock = React.memo((props: StimProps) => {
  const { stimWidth, onType, stimTarget, innerNums, isTarget, needReload, needLateReload, stimIndex, stimFreq, stimPhase } = props
  const innerBlockNum = innerNums ?? defaultStimProps.innerNums ?? 10
  const innerBlock = []
  const FREQ = stimFreq ?? 12
  const PHASE = stimPhase ?? 0
  const STIM_ARR: Array<string> = []

  for (let i = 0; i < (innerBlockNum) ** 2; i++) {
    innerBlock.push(i + 1)
  }

  // 按照 freq, phase 计算的 0-1 之间的透明度, 先按照60hz刷新率计算
  for (let i = 0; i < 4*60; i++) {
    const tmp = (1 + Math.sin(2 * Math.PI * FREQ * i / 60 + PHASE)) / 2
    STIM_ARR.push(tmp.toFixed(2))
  }

  let colorTheme = defaultOffOpacity

  if (onType) {
    colorTheme = defaultOnOpacity
  }

  // 闪烁刺激，但应该是成片的，因此需要更大的组件去封装，此部分需要抽离
  const shiningBlockRef = useRef() as MutableRefObject<HTMLDivElement>
  const askframe = window.requestAnimationFrame
  let count = 0;

  const step = useCallback(() => {
    // !startTime && (startTime = currentTime)
    // 按照 freq, phase 计算
    count += 1;
    if (count < 60 * 4 && isTarget === 'stim') {
      shiningBlockRef.current.style.opacity = STIM_ARR[count-1]
      askframe(step)
    } else {
      shiningBlockRef.current.style.opacity = '1'
      count = 0
      // console.log('停止')
    }
  }, [isTarget])

  // 根据状态控制，但需要额外封装大实验范式， 尤其是reload，只能reload一个,需要重构
  React.useEffect(() => {
    if (needReload) {
      window.location.reload() // 重新加载强制中断
    }
  }, [needReload])

  React.useEffect(() => {
    if (needLateReload) {
      setTimeout(() => {
        window.location.reload() // 目标态等待1秒再刷新,不能立即中断动画，但可以控制颜色一致而没有闪烁感
      }, 1000)
    }
  }, [needLateReload])
  
  React.useEffect(() => {
    if (isTarget === 'stim') {
      askframe(step)
    }
  }, [askframe, isTarget, step])

  // console.log('---> 当前状态', isTarget, stimIndex)
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
        <div className='stim-block-grid-all'
          // style={{
          //   // background: isTarget ? defaultTargetOpacity.initStimColor : colorTheme.initStimColor,
          //   opacity: curOpacity ?? defaultStimProps.curOpacity
          // }}
          ref={shiningBlockRef}
          data-id={`stimindex-${stimIndex}`}
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
export default SSVEPStimBlock;