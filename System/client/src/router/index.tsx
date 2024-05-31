import { lazy } from "react";
import Error from "../pages/404";
import Home from "../pages/Home";
import Stim from "../pages/Stim";
// import Experiment from "../pages/Experiment";
// import Docs from "../pages/Docs";
import SSVEPonLine from "../pages/SSVEP";
import { Navigate, createBrowserRouter } from 'react-router-dom';
import lazyLoad from './lazyLoad'


// const SSVEPonLine = lazy(() => import('../pages/SSVEP')) // 不适合懒加载，会闪屏
const Docs = lazy(() => import('../pages/Docs'))
const Experiment = lazy(() => import('../pages/Experiment'))

// 路由表
const defRouter = [
  {
    path: '/',
    name: '',
    isShow: false,
    element: <Navigate to={'/home'} />
  },
  {
    path: '/home',
    name: 'HOME',
    isShow: true,
    element: <Home />
  },
  {
    path: '/experiment',
    name: '在线实验',
    isShow: false,
    element: lazyLoad(Experiment)
    // element: <Experiment />
  },
  {
    path: '/stim',
    name: '刺激模拟',
    isShow: false,
    element: <Stim/>
  },
  {
    path: '/ssvep',
    name: '',
    isShow: false,
    // element: lazyLoad(SSVEPonLine) // 这个页面用LazyLoad会导致闪烁
    element: <SSVEPonLine/>
  },
  {
    path: '/doc',
    name: '',
    isShow: false,
    element: lazyLoad(Docs)
    // element: <Docs/>
  },
  {
    path: '*',
    name: '404',
    isShow: false,
    element: <Error />
  }
]

const Router = createBrowserRouter(defRouter)

export default Router