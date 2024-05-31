// router/lazyLoad.tsx
import { Suspense } from 'react'

const lazyLoad = (Component: React.LazyExoticComponent<() => JSX.Element>) => {
  return (
    <Suspense fallback={<div>loading...</div>}>
      <Component />
    </Suspense>
  )
}

export default lazyLoad
