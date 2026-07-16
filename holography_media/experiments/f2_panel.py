"""F2 qualitative panel: target vs four reconstructions on a smooth
natural-spectrum target at Bayfol-like defaults, 405 nm."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from holomedia import (NPDDRecorder, MediumParams, SlabBPM, media_in_the_loop,
                       media_blind_sgd, media_blind_gs, oracle_ideal, psnr)
torch.set_default_dtype(torch.float64)
N,DX=256,0.1
# smooth "natural" target: low-pass filtered random field, nonneg
torch.manual_seed(3)
g=torch.rand(N); G=torch.fft.fft(g)
f=torch.fft.fftfreq(N,d=DX); G=G*torch.exp(-(f/1.2)**2)
t=torch.fft.ifft(G).real; t=(t-t.min()); t=t/t.max()
p=MediumParams()
rec=NPDDRecorder(N,DX,t_total=10,n_steps=100,params=p)
bpm=SlabBPM(N,DX,0.405,p.thickness,n_z=12)
_,ro,_=media_in_the_loop(t,rec,bpm,n_iters=200,verbose=False)
_,rb=media_blind_sgd(t,rec,bpm,n_iters=200)
_,rg=media_blind_gs(t,rec,bpm)
_,rc=oracle_ideal(t,rec,bpm,n_iters=200)
x=np.arange(N)*DX
fig,axs=plt.subplots(5,1,figsize=(7,8),sharex=True)
for ax,(y,ttl) in zip(axs,[(t,"target"),
    (rg,f"media-blind GS  ({psnr(rg,t):.1f} dB)"),
    (rb,f"media-blind SGD  ({psnr(rb,t):.1f} dB)"),
    (ro,f"media-in-the-loop (ours)  ({psnr(ro,t):.1f} dB)"),
    (rc,f"oracle ideal medium  ({psnr(rc,t):.1f} dB)")]):
    yy=(y/ y.max()).numpy() if hasattr(y,'numpy') else y
    ax.plot(x,yy,lw=1.2); ax.set_ylabel(ttl,fontsize=7,rotation=0,ha='right',va='center')
    ax.set_yticks([])
axs[-1].set_xlabel("x (µm)")
fig.suptitle("F2: natural-spectrum target, PVA/AA-like medium, 405 nm",fontsize=10)
fig.tight_layout(); fig.savefig("figures/figD_panel.png",dpi=200)
print("panel PSNRs:", {k:round(v,2) for k,v in
  dict(gs=psnr(rg,t),blind=psnr(rb,t),ours=psnr(ro,t),oracle=psnr(rc,t)).items()})
