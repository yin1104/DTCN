export function parseUrlParams(url: string) {
  // 获取 URL 中的查询参数部分
  const paramsStart = url.indexOf('?');
  if (paramsStart === -1) {
    return {}; // 如果没有查询参数，则返回空对象
  }

  const paramString = url.substring(paramsStart + 1);

  // 对每个查询参数进行解析
  const paramsArray = paramString.split('&');
  const params = {};

  for (let i = 0; i < paramsArray.length; i++) {
    const [key, value] = paramsArray[i].split('=');
    params[key] = decodeURIComponent(value);
  }

  return params;
}


export function calAcc(groundTruth: string, predictedAns: string) {
  let count = 0
  for (let i=0; i<groundTruth.length; i++) {
    if (groundTruth[i] === predictedAns[i]) {
      count ++
    }
  }
  return (100*count/groundTruth.length).toFixed(2)
}


export function getDate() {
  const addZero = (t: number) => {
    return t < 10 ? '0'+ t : t
  }
  const time = new Date()
  const Y = time.getFullYear(),
        D = time.getDate(),
        h = time.getHours(),
        m = time.getMinutes(),
        s = time.getSeconds()
  let M = time.getMonth() + 1

  if (M > 12) {
    M = M - 12
  }
  return `${Y}-${addZero(M)}-${addZero(D)} ${addZero(h)}:${addZero(m)}:${addZero(s)}`
} 