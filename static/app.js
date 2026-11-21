(function(){
var d=document,$=function(i){return d.getElementById(i)};
var th=$('themeBtn');
if(th)th.onclick=function(){var n=d.documentElement.getAttribute('data-theme')==='dark'?'light':'dark';d.documentElement.setAttribute('data-theme',n);try{localStorage.setItem('ng-theme',n)}catch(e){}window.dispatchEvent(new Event('themechange'))};
var nb=$('navBtn');if(nb)nb.onclick=function(){d.querySelector('.nav').classList.toggle('open')};
var ub=$('userBtn');if(ub)ub.onclick=function(e){e.stopPropagation();$('umenu').classList.toggle('open')};
d.addEventListener('click',function(){var m=$('umenu');if(m)m.classList.remove('open')});
var an=$('announce');
try{if(sessionStorage.getItem('ng-ann'))an.classList.add('hidden')}catch(e){}
if($('annClose'))$('annClose').onclick=function(){an.classList.add('hidden');try{sessionStorage.setItem('ng-ann','1')}catch(e){}};
window.toast=function(msg,cat){var t=d.createElement('div');t.className='toast '+(cat||'info');t.textContent=msg;$('toasts').appendChild(t);setTimeout(function(){t.style.opacity='0';t.style.transition='.4s';setTimeout(function(){t.remove()},400)},4800)};
d.querySelectorAll('.toast').forEach(function(t){setTimeout(function(){t.style.opacity='0';t.style.transition='.4s';setTimeout(function(){t.remove()},400)},5200)});
var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add('in');io.unobserve(e.target)}})},{threshold:.12});
d.querySelectorAll('.reveal').forEach(function(e){io.observe(e)});
var co=new IntersectionObserver(function(es){es.forEach(function(e){if(!e.isIntersecting)return;co.unobserve(e.target);var el=e.target,end=parseFloat(el.dataset.count),dec=parseInt(el.dataset.dec||'0'),suf=el.dataset.suf||'',t0=performance.now();(function f(t){var k=Math.min((t-t0)/1300,1),v=end*(1-Math.pow(1-k,3));el.textContent=v.toFixed(dec)+suf;if(k<1)requestAnimationFrame(f)})(t0)})},{threshold:.4});
d.querySelectorAll('[data-count]').forEach(function(e){co.observe(e)});
d.querySelectorAll('[data-pw]').forEach(function(b){b.onclick=function(){var i=b.parentNode.querySelector('input');i.type=i.type==='password'?'text':'password';b.textContent=i.type==='password'?'Show':'Hide'}});
var pm=d.querySelector('[data-meter]');
if(pm){var pi=$(pm.dataset.meter);pi.oninput=function(){var v=pi.value,s=0;if(v.length>=6)s++;if(v.length>=10)s++;if(/[A-Z]/.test(v)&&/[a-z]/.test(v))s++;if(/\d/.test(v)&&/[^A-Za-z0-9]/.test(v))s++;var i=pm.firstElementChild;i.style.width=(v?s*25:0)+'%';i.style.background=['#ef4a4f','#ef4a4f','#f2a93b','#1ec97f','#1ec97f'][s]}}
})();
