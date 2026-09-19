import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
data = json.loads((OUT / 'plot_data.json').read_text(encoding='utf-8'))
records = data['records']
BLUE, RED, GREY = '#1675A9', '#CF4C47', '#394B59'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.labelcolor': GREY,
    'text.color': GREY, 'axes.titleweight': 'bold', 'savefig.facecolor': 'white',
    'svg.fonttype': 'none'})
axes = [('vx','vx','m/s'), ('vy','vy','m/s'), ('yaw_rate','yaw','rad/s')]
def arr(key):
    return np.array([np.nan if r[key] is None else r[key] for r in records])
ok = arr('trial_completed') == 1
valid = arr('measurement_steps') > 0
x = np.arange(len(records))
def save(fig, name):
    fig.savefig(OUT / (name+'.png'), dpi=180, bbox_inches='tight')
    fig.savefig(OUT / (name+'.svg'), bbox_inches='tight')
    plt.close(fig)
def style(ax):
    ax.grid(alpha=.16)
    ax.set_axisbelow(True)
legend = [Line2D([],[],marker='_',markersize=13,lw=0,color=GREY,label='Command'),
          Line2D([],[],marker='o',lw=0,color=BLUE,label='Completed: mean ± temporal SD'),
          Line2D([],[],marker='X',lw=0,color=RED,label='Failed: observed mean ± SD')]

fig, axs = plt.subplots(3,1,figsize=(17,10),sharex=True)
fig.suptitle('Velocity tracking by trial | Sep18 model_11000', x=.065, ha='left', y=.985, fontsize=19, weight='bold')
fig.text(.065,.948,'43 trials · 32 completed (74.4%) · 0.2 s warmup + up to 3 s measurement · single seed',fontsize=11)
for ax,(a,label,unit) in zip(axs,axes):
    for j in x[~ok]: ax.axvspan(j-.46,j+.46,color=RED,alpha=.07,lw=0)
    ax.scatter(x,arr(a),marker='_',s=165,color=GREY,zorder=4)
    for mask,color,marker in [(ok,BLUE,'o'),(~ok & valid,RED,'X')]:
        ax.errorbar(x[mask],arr(a+'_actual_mean')[mask],yerr=arr(a+'_actual_std')[mask],
                    fmt=marker,ms=4.7,color=color,elinewidth=1,capsize=2,zorder=3)
    for j in x[~valid]:
        ax.text(j,.025,'N/A',rotation=90,color=RED,fontsize=8,ha='center',transform=ax.get_xaxis_transform())
    for boundary in [.5,6.5,12.5,18.5]: ax.axvline(boundary,color=GREY,alpha=.25,lw=.8)
    ax.axhline(0,color=GREY,lw=.7,alpha=.35)
    ax.set_ylabel(label+' ('+unit+')')
    style(ax)
axs[0].legend(handles=legend,loc='upper right',ncol=3,fontsize=9)
for center,name in [(0,'Stand'),(3.5,'Pure vx'),(9.5,'Pure vy'),(15.5,'Pure yaw'),(30.5,'Mixed')]:
    axs[0].text(center,1.025,name,ha='center',transform=axs[0].get_xaxis_transform(),weight='bold')
axs[-1].set_xticks(x,[r['trial_id'] for r in records],rotation=90,fontsize=8)
axs[-1].set_xlim(-.7,len(records)-.3)
axs[-1].set_xlabel('Independent trials (not a time axis)')
fig.text(.065,.018,'Red shading: early termination. T0015 / T0022 have no measured samples. Error bars are within-trial SD, not confidence intervals.',fontsize=10)
fig.subplots_adjust(top=.875,bottom=.12,left=.065,right=.985,hspace=.19)
save(fig,'tracking_overview')

fig, axs = plt.subplots(2,3,figsize=(15,9))
fig.suptitle('Command response | forward tracking stronger than turning',x=.07,ha='left',y=.985,fontsize=18,weight='bold')
fig.text(.07,.943,'Points show per-trial means; dashed diagonal indicates ideal tracking. Failed points cover short, unequal windows.',fontsize=10)
for row in range(2):
    for col,(a,mode,unit) in enumerate(axes):
        ax=axs[row,col]
        mask=np.array([r['mode']==(mode if row==0 else 'mixed') for r in records]) & valid
        target,actual=arr(a),arr(a+'_actual_mean')
        values=np.r_[target[mask],actual[mask],0]
        lo,hi=values.min(),values.max(); pad=(hi-lo)*.12
        lo-=pad; hi+=pad
        ax.plot([lo,hi],[lo,hi],'--',color=GREY,alpha=.6,lw=1)
        for success,color,marker in [(True,BLUE,'o'),(False,RED,'X')]:
            m=mask & (ok==success)
            ax.scatter(target[m],actual[m],c=color,marker=marker,s=42,zorder=3)
            if row==0:
                for j in x[m]: ax.annotate(records[j]['trial_id'],(target[j],actual[j]),xytext=(5,5),textcoords='offset points',fontsize=8,color=color)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_title(('Pure '+mode if row==0 else 'Mixed: '+mode),loc='left')
        ax.set_xlabel('Command ('+unit+')'); ax.set_ylabel('Actual mean ('+unit+')')
        ax.axhline(0,color=GREY,alpha=.2,lw=.7); ax.axvline(0,color=GREY,alpha=.2,lw=.7)
        style(ax)
fig.legend(handles=[Line2D([],[],marker='o',lw=0,color=BLUE,label='Completed'),Line2D([],[],marker='X',lw=0,color=RED,label='Failed (observed)')],loc='lower right',bbox_to_anchor=(.98,.005),ncol=2,frameon=False)
fig.text(.07,.02,'No samples: pure yaw T0015 (command −0.25); mixed T0022. Both remain failures in completion statistics.',fontsize=10)
fig.subplots_adjust(top=.88,bottom=.105,left=.07,right=.985,hspace=.40,wspace=.30)
save(fig,'command_response')

fig,axs=plt.subplots(2,2,figsize=(13,9))
fig.suptitle('Tracking accuracy and carrying stability',x=.07,ha='left',y=.985,fontsize=19,weight='bold')
fig.text(.07,.944,'Completion and error must be read together: failed trials end within 0.90 s of trial start.',fontsize=11)
ax=axs[0,0]
modes=['stand','vx','vy','yaw','mixed']
counts=[sum(r['mode']==m for r in records) for m in modes]
done=[sum(r['mode']==m and r['trial_completed']==1 for r in records) for m in modes]
rates=np.array(done)/counts*100
ax.bar(modes,rates,color=BLUE,width=.6)
for j,(n,c,p) in enumerate(zip(done,counts,rates)): ax.text(j,p+2,f'{n}/{c}\n{p:.1f}%',ha='center',fontsize=10)
ax.set_ylim(0,125); ax.set_yticks([0,25,50,75,100]); ax.set_ylabel('Completed (%)'); ax.set_title('A  Completion by mode',loc='left'); style(ax)
ax=axs[0,1]
for i,key in enumerate(['mae','rmse']):
    vals=[next(r[key] for r in data['metrics'] if r['group']=='completed' and r['axis']==a) for a,_,_ in axes]
    bars=ax.bar(np.arange(3)+(i-.5)*.32,vals,width=.32,color=[BLUE,'#E8A044'][i],label=key.upper())
    ax.bar_label(bars,fmt='%.3f',padding=3,fontsize=10)
ax.set_xticks(range(3),['vx\n(m/s)','vy\n(m/s)','yaw\n(rad/s)']); ax.set_ylim(0,.39)
ax.set_ylabel('Error (axis-specific units)'); ax.set_title('B  Completed trials only (n=32)',loc='left'); ax.legend(frameon=False);style(ax)
ax=axs[1,0]
for mask,col,mark in [(ok,BLUE,'o'),(~ok&valid,RED,'X')]:
    ax.scatter(arr('bilateral_hand_contact_fraction')[mask]*100,arr('box_tilt_max_deg')[mask],color=col,marker=mark,s=42,alpha=.85)
ax.set_xlabel('Bilateral hand contact fraction (%)');ax.set_ylabel('Maximum observed box tilt (deg)');ax.set_title('C  Contact and box tilt',loc='left');style(ax)
ax=axs[1,1]
failed=[r for r in records if not r['trial_completed']]
colors=[RED if r['termination_reason']=='excessive_box_tilt' else '#E8A044' for r in failed]
ax.barh([r['trial_id'] for r in failed],[r['survival_duration_s'] for r in failed],color=colors)
ax.axvline(.2,color=GREY,ls='--',lw=1,label='Warmup: 0.2 s');ax.set_xlim(0,1.06);ax.invert_yaxis()
ax.set_xlabel('Time from trial start to termination (s)');ax.set_title('D  Failures: 8 excessive tilt, 3 box drop',loc='left');style(ax)
fig.text(.07,.023,'Blue circles: completed; red crosses: failed. D: red = excessive tilt, amber = box drop. No post-termination values are imputed.',fontsize=9)
fig.subplots_adjust(top=.865,bottom=.105,left=.075,right=.98,hspace=.43,wspace=.32)
save(fig,'stability_diagnostics')
print('Saved 3 PNG and 3 SVG figures.')
