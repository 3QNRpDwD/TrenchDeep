// Viewport-only helpers: never hide, recreate, reposition or focus graph nodes.
(function(root){
  function chooseTarget(available,previous,focused){
    return available.find(id=>!previous.has(id))||(available.includes(focused)?focused:available[0])||null;
  }
  function offset(position,scale,width,height){
    return {left:Math.max(0,(position.x+93)*scale-width/2+12),top:Math.max(0,(position.y+32)*scale-height/2+12)};
  }
  function layout(ids,direction){
    const vertical=direction==='vertical';
    return {width:vertical?476:Math.ceil(ids.length/2)*224+28,height:vertical?Math.ceil(ids.length/2)*120+50:300,
      positions:new Map(ids.map((id,i)=>[id,vertical?{x:14+(i%2)*224,y:25+Math.floor(i/2)*120}:{x:14+Math.floor(i/2)*224,y:45+(i%2)*140}]))};
  }
  function edgePath(a,b,index,width,height,direction){
    if(direction==='vertical'){
      if(a.y===b.y)return `M ${a.x+186} ${a.y+32} L ${b.x} ${b.y+32}`;
      if(b.y-a.y===120&&a.x===b.x)return `M ${a.x+93} ${a.y+65} L ${b.x+93} ${b.y}`;
      const lane=width-10-(index%5)*5;
      return `M ${a.x+186} ${a.y+32} L ${lane} ${a.y+32} L ${lane} ${b.y-12} L ${b.x+93} ${b.y-12} L ${b.x+93} ${b.y}`;
    }
    if(a.x===b.x){const down=b.y>a.y;return `M ${a.x+93} ${a.y+(down?65:0)} L ${b.x+93} ${b.y+(down?0:65)}`;}
    if(b.x-a.x===224&&a.y===b.y)return `M ${a.x+186} ${a.y+32} L ${b.x} ${b.y+32}`;
    const lane=12+(index%5)*5,y=a.y===b.y?(a.y<100?lane:height-lane):145+(index%4)*6;
    return `M ${a.x+186} ${a.y+32} L ${a.x+202} ${a.y+32} L ${a.x+202} ${y} L ${b.x-9} ${y} L ${b.x-9} ${b.y+32} L ${b.x} ${b.y+32}`;
  }
  const api={chooseTarget,offset,layout,edgePath};root.VIEWER_CAMERA=api;
  if(typeof module!=='undefined')module.exports=api;
})(globalThis);
