import React from 'react';
import { Flex, Text, Separator } from '@radix-ui/themes';
import { Link } from 'react-router-dom';
import './nav.css'

type NavigiterProps = {
  curPage: number
};

const Navigiter: React.FC<NavigiterProps> = (props: NavigiterProps) => {
  const { curPage } = props

  return (
    <div className='nav'>
      <Text size="2">
        <Flex gap="3" align="center">
          <Link to='/home' className={`${curPage === 1 ? 'hover' : 'nohover'}`}>Home</Link>
          <Separator orientation="vertical"/>
          <Link to='/stim' className={`${curPage === 2 ? 'hover' : 'nohover'}`}>刺激模拟</Link>
          <Separator orientation="vertical" />
          <Link to='/experiment' className={`${curPage === 3 ? 'hover' : 'nohover'}`}>在线实验</Link>
          <Separator orientation="vertical" />
          <Link to='/doc' className={`${curPage === 4 ? 'hover' : 'nohover'}`}>在线文档</Link>
        </Flex>
      </Text>
    </div>
  )
}
export default Navigiter;