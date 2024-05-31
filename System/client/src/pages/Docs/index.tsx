import ReactMarkdown from 'react-markdown' // 解析md
import remarkGfm from 'remark-gfm';  // 解析表格
import MarkdownNavbar from 'markdown-navbar'; // 解析目录
import 'markdown-navbar/dist/navbar.css';
import './doc.css'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { coldarkDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import Navigiter from '../../components/Navigiter/Navigiter';
import { useQuery } from '@tanstack/react-query';

const Docs = () => {
  // RQ 真好用
  const { isLoading, error, data } = useQuery({
    queryKey: ['systemDoc'],
    queryFn: () =>
      fetch('/help.md').then((res) =>
        res.text(),
      ),
  })

  if (isLoading) return 'Loading...'
  if (error) return 'An error has occurred: ' + error.message

  return (
    <div className='docs'>
      <Navigiter curPage={4} />
      <div className='md'>
        <div className='leftSide scroll'>
          {
            data && <MarkdownNavbar
              className='markdownNav'
              source={data}
              // headingTopOffset={10}
              ordered={false} // 是否显示标题题号
            />
          }
        </div>
        <div className='markdown-body content'>
          {data && <ReactMarkdown
            children={data}
            remarkPlugins={[remarkGfm]} //添加对 GitHub 风格的 Markdown 支持
            components={{
              code(props) {
                const { children, className, ...rest } = props
                const match = /language-(\w+)/.exec(className || '')
                return match ? (
                  <SyntaxHighlighter
                    {...rest}
                    PreTag="div"
                    children={String(children).replace(/\n$/, '')}
                    language={match[1]}
                    style={coldarkDark}
                    ref={null}
                  />
                ) : (
                  <code {...rest} className={className}>
                    {children}
                  </code>
                )
              }
            }}
          />}
        </div>
      </div>
    </div>
  )
}
export default Docs;