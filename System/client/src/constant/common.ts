// 我们自己的后端域名
const DOMAIN = 'localhost'

// 我们自己的后端监听接口，暂时为 8081
const POST = 8081

export const WS_BACKEND_URL =`ws://${DOMAIN}:${POST}`
// import.meta.env.VITE_WS_BACKEND_URL || "ws://127.0.0.1:7001";

export const HTTP_BACKEND_URL =`http://${DOMAIN}:${POST}`
// import.meta.env.VITE_HTTP_BACKEND_URL || "http://127.0.0.1:7001";

// 关闭 WEBSocket服务的code
export const USER_CLOSE_WEB_SOCKET_CODE = 4333;