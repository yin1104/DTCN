import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: { //同plugins同级
    // port: 8080, 默认是5173,这里可以自己指定
    // 代理解决跨域
    // host: 'localhost',
    // port: 5173,
    // proxy: {
    //   '/translate/audio/': {
    //     target: 'http://localhost:8082/translate/audio/',  // 接口源地址
    //     secure: false,
    //     changeOrigin: true,   // 开启跨域
    //     rewrite: (path) => path.replace(/^\/translate\/audio\//, ''),
    //   }
    // }
  }
})
