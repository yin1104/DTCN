/**
 * @param {number} targetCount 不小于1的整数，表示经过targetCount帧之后返回结果
 * @return {Promise<number>}
 */
const getScreenFps = (() => {
  // 先做一下兼容性处理
  const nextFrame = [
    window.requestAnimationFrame,
    // window.webkitRequestAnimationFrame,
    // window.mozRequestAnimationFrame,
  ].find((fn) => fn);

  if (!nextFrame) {
    console.error("requestAnimationFrame is not supported!");
    return;
  }

  return (targetCount = 50) => {
    // 判断参数是否合规
    if (targetCount < 1) throw new Error("targetCount cannot be less than 1.");
    
    const beginDate = Date.now();
    
    let count = 0;

    return new Promise((resolve) => {
      (function log() {
        nextFrame(() => {
          if (++count >= targetCount) {
            const diffDate = Date.now() - beginDate;
            const fps = (count / diffDate) * 1000;
            return resolve(fps);
          }
          log();
        });
      })();
    });
  };
})();

export default getScreenFps

// 可以传一个不小于1的整数，理论上来说数字越大，结果越精确或者说越平均，但是等待时间也就越长
// 实测50是最理想的，结果比较精确了，小于50结果偏差会比较大
// 参数不传默认值就是50
// 60帧的电脑屏幕测试结果在61和62之间浮动，120帧的手机测试结果在121和122之间浮动

// getScreenFps().then((fps) => {
//   console.log("当前屏幕刷新率为", fps);
//   // 当前屏幕刷新率为 61.50061500615006
// });
