import React from 'react';
import { Flex, Text, Button, Card, Box, Code } from '@radix-ui/themes';
import { ArrowRightIcon, TransparencyGridIcon, MixIcon, RocketIcon } from '@radix-ui/react-icons'
import { Link } from 'react-router-dom';
import Navigiter from '../../components/Navigiter/Navigiter';
import './home.css'

const Home: React.FC = () => {

  return (
    <div className='bg'>
      <Navigiter curPage={1} />
      <div className='title'>SMART SSVEP</div>
      <Flex direction="column" gap="4" align="center" style={{ marginBottom: 50 }}>
        <Text>基于 <Code>Deep-Learning</Code> 与 <Code>gpt-3.5-turbo</Code> 的 <Code>SSVEP-BCI</Code> 在线仿真中文拼写系统</Text>
      </Flex>
      <Flex gap="4" direction="row">
        <Card size="3" style={{ width: 300 }} variant="classic">
          <Flex gap="3" align="center">
            <div className='cardIcon iconColor'><MixIcon /></div>
            <Box>
              <Text as="div" size="3" weight="bold" style={{ marginBottom: 5 }}>
                SSVEP-BCI中文拼写
              </Text>
              <Text as="div" size="1" color="gray" >
                Benchmark系统<br />
                支持CPU版本和GPU版本<br />
                跨端可拓展<br />
                即插即用 ......<br />
              </Text>
            </Box>
          </Flex>
        </Card>

        <Card size="3" style={{ width: 300 }}>
          <Flex gap="3" align="center">
            <div className='cardIcon'><TransparencyGridIcon /></div>
            <Box>
              <Text as="div" size="3" weight="bold" style={{ marginBottom: 5 }}>
                在线实验刺激模拟
              </Text>
              <Text as="div" size="1" color="gray">
                支持多种算法<br />
                在线仿真算法策略模拟在线场景<br />
                支持ChatGPT<br />
                支持中文拼写场景 ......<br />
              </Text>
            </Box>
          </Flex>
        </Card>

        <Card size="3" style={{ width: 300 }}>
          <Flex gap="3" align="center">
            <div className='cardIcon iconColor'><RocketIcon /></div>
            <Box>
              <Text as="div" size="3" weight="bold" style={{ marginBottom: 5 }}>
                友好的知识库
              </Text>
              <Text as="div" size="1" color="gray">
                自定义数据库<br />
                自动生成实验日志<br />
                贴心的实验说明手册<br />
                支持中|繁|英自由切换 ......<br />
              </Text>
            </Box>
          </Flex>
        </Card>
      </Flex>
      <div className='start'>
        <Link to='/experiment'>
        <Button variant='soft' style={{ width: 160 }} size="3">开启模拟<ArrowRightIcon /></Button>
        </Link>
      </div>
    </div>
  );
}

export default Home;