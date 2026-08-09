const $ = (id) => document.getElementById(id);
const state = { running: false };
const palettes = {
  potential: [[39,20,83],[18,110,123],[89,179,107],[249,230,65]],
  field: [[22,12,53],[124,29,111],[224,93,71],[249,233,91]],
  error: [[19,41,31],[74,118,82],[232,185,63],[255,243,166]]
};

function color(value, min, max, palette) {
  const t = max === min ? 0 : Math.max(0, Math.min(1, (value - min) / (max - min)));
  const scaled = t * (palette.length - 1), i = Math.min(palette.length - 2, Math.floor(scaled)), f = scaled - i;
  return palette[i].map((v, c) => Math.round(v + (palette[i + 1][c] - v) * f));
}
function sizeCanvas(canvas) { const dpr = devicePixelRatio || 1; canvas.width = Math.round(canvas.clientWidth*dpr); canvas.height = Math.round(canvas.clientHeight*dpr); return dpr; }
function heatmap(id, matrix, paletteName) {
  const canvas=$(id), dpr=sizeCanvas(canvas), ctx=canvas.getContext('2d');
  const rows=matrix.length, cols=matrix[0].length, flat=matrix.flat(), min=Math.min(...flat), max=Math.max(...flat);
  const image=document.createElement('canvas'); image.width=cols; image.height=rows; const ictx=image.getContext('2d'), pixels=ictx.createImageData(cols,rows);
  for(let y=0;y<rows;y++) for(let x=0;x<cols;x++){const rgb=color(matrix[rows-1-y][x],min,max,palettes[paletteName]),p=(y*cols+x)*4;pixels.data[p]=rgb[0];pixels.data[p+1]=rgb[1];pixels.data[p+2]=rgb[2];pixels.data[p+3]=255}
  ictx.putImageData(pixels,0,0); ctx.imageSmoothingEnabled=false;ctx.drawImage(image,0,0,canvas.width,canvas.height); return {min,max};
}
function vectors(ex,ey){const c=$('vectorCanvas'),dpr=sizeCanvas(c),ctx=c.getContext('2d'),rows=ex.length,cols=ex[0].length,step=Math.max(2,Math.floor(Math.min(rows,cols)/15));ctx.fillStyle='#e7e6df';ctx.fillRect(0,0,c.width,c.height);ctx.strokeStyle='#183028';ctx.fillStyle='#183028';ctx.lineWidth=1.15*dpr;for(let y=step;y<rows-step;y+=step)for(let x=step;x<cols-step;x+=step){const px=x/(cols-1)*c.width,py=(1-y/(rows-1))*c.height,vx=ex[y][x],vy=-ey[y][x],mag=Math.hypot(vx,vy)||1,len=Math.min(c.width/18,mag*c.width/22),dx=vx/mag*len,dy=vy/mag*len;ctx.beginPath();ctx.moveTo(px-dx*.45,py-dy*.45);ctx.lineTo(px+dx*.45,py+dy*.45);ctx.stroke();const a=Math.atan2(dy,dx),tx=px+dx*.45,ty=py+dy*.45;ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-5*dpr*Math.cos(a-.55),ty-5*dpr*Math.sin(a-.55));ctx.lineTo(tx-5*dpr*Math.cos(a+.55),ty-5*dpr*Math.sin(a+.55));ctx.fill()}}
function fmt(v,d=4){return Number(v).toFixed(d)} function rangeText(r){return `${fmt(r.min,3)} — ${fmt(r.max,3)}`}
function toast(text){const t=$('toast');t.textContent=text;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),3200)}
async function run(){if(state.running)return;state.running=true;const button=$('runButton');button.disabled=true;button.firstElementChild.textContent='模拟进行中…';try{const response=await fetch('api/solve',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({nx:+$('nx').value,ny:+$('ny').value,walks:+$('walks').value,seed:+$('seed').value})});const data=await response.json();if(!response.ok)throw new Error(data.error||'求解失败');const pr=heatmap('potentialCanvas',data.potential,'potential'),fr=heatmap('fieldCanvas',data.magnitude,'field'),er=heatmap('errorCanvas',data.stderr,'error');vectors(data.ex,data.ey);$('runtime').textContent=Math.round(data.runtimeMs).toLocaleString();$('rmse').textContent=fmt(data.rmse,6);$('meanError').textContent=fmt(data.stats.stderr.mean,6);$('meanField').textContent=fmt(data.stats.magnitude.mean,4);$('potentialRange').textContent=rangeText(pr);$('fieldRange').textContent=rangeText(fr);$('errorRange').textContent=rangeText(er);toast(`运行 ${data.runId} 已完成`)}catch(e){toast(e.message)}finally{state.running=false;button.disabled=false;button.firstElementChild.textContent='运行模拟'}}
['nx','ny'].forEach(id=>{$(id).addEventListener('input',()=>$(id+'Value').textContent=$(id).value)});$('runButton').addEventListener('click',run);window.addEventListener('resize',()=>{});
fetch('api/status').then(r=>r.json()).then(s=>{$('statusText').textContent=s.ready?'本地求解器在线 · CPU':'求解器不可用';if(s.ready)run()}).catch(()=>{$('statusText').textContent='本地服务异常'});
