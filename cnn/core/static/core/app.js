// ===== canvas drawing =====
const pad = document.getElementById('pad');
const ctx = pad.getContext('2d');
ctx.fillStyle = '#ffffff'; ctx.fillRect(0,0,pad.width,pad.height);
ctx.lineWidth = 18; ctx.lineCap = 'round'; ctx.strokeStyle = '#000000';

let drawing=false;
pad.addEventListener('mousedown', e=>{drawing=true; ctx.beginPath(); ctx.moveTo(e.offsetX,e.offsetY);});
pad.addEventListener('mousemove', e=>{if(!drawing) return; ctx.lineTo(e.offsetX,e.offsetY); ctx.stroke();});
['mouseup','mouseleave'].forEach(ev=>pad.addEventListener(ev, ()=>drawing=false));

document.getElementById('btn-clear').onclick = ()=>{
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,pad.width,pad.height); ctx.fillStyle='#000000';
};

const toBase64PNG = ()=> pad.toDataURL('image/png');

document.getElementById('btn-predict').onclick = async ()=>{
  const res = await fetch('/predict/', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({image: toBase64PNG()})
  });
  const j = await res.json();
  document.getElementById('result').textContent =
    res.ok ? `pred: ${j.pred}, conf: ${j.conf}` : JSON.stringify(j);
};

// ===== SSE log pretty append (htmx sse-swap="message" จะส่ง event.data มา) =====
document.body.addEventListener('htmx:sseMessage', (ev)=>{
  try{
    const d = JSON.parse(ev.detail.data);
    const div = document.getElementById('train-log');
    div.insertAdjacentHTML('beforeend', `<div>${new Date().toLocaleTimeString()} — ${JSON.stringify(d)}</div>`);
    div.scrollTop = div.scrollHeight;
  }catch(e){}
});
