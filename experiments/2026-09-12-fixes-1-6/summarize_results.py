"""Regenerate comparison data and figures from the saved, full episode results."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
MODELS={'original':'holdout_original.json','seed42':'holdout_seed42_best.json','seed43':'holdout_seed43_best.json'}
LABELS={'original':'Original policy / corrected plant','seed42':'Retrained seed 42 / selected checkpoint','seed43':'Retrained seed 43 / selected checkpoint'}
COLORS={'original':'#c65f36','seed42':'#4977ad','seed43':'#247c61'}


def main():
    loaded={key:json.loads((HERE/name).read_text()) for key,name in MODELS.items()}
    assert all(len(d['episodes'])==140 for d in loaded.values())
    comparison={}
    for key,data in loaded.items():
        episodes=data['episodes']
        comparison[key]={'episodes':len(episodes),'completed':sum(r['completed'] for r in episodes),
                         'passed':sum(r['passed'] for r in episodes),'scenarios':data['summary']}
    all_files=list(HERE.glob('baseline_*.json'))+list(HERE.glob('holdout_*.json'))
    all_files+=list(HERE.glob('training_seed*/eval_*.json'))
    all_episodes=[r for p in all_files for r in json.loads(p.read_text())['episodes']]
    comparison['total_benchmark_episodes']=len(all_episodes)
    comparison['total_completed_episodes']=sum(r['completed'] for r in all_episodes)
    (HERE/'comparison.json').write_text(json.dumps(comparison,indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),layout='constrained')
    titles=['Nominal hover, 30 s','Wind 0.5 N, 10 s','Randomized aircraft, 30 s']
    for ax,scenario,title in zip(axes,['hover30','wind10','randomized_hover30'],titles):
        for i,(key,data) in enumerate(loaded.items()):
            rows=[r for r in data['episodes'] if r['scenario']==scenario]
            values=np.array([r['altitude_rmse_m']*100 for r in rows if r['completed']])
            ax.bar(i,np.mean(values),color=COLORS[key],width=.62,alpha=.85)
            ax.scatter(i+np.linspace(-.16,.16,len(values)),values,color=COLORS[key],s=14,alpha=.65)
            ax.text(i,np.mean(values)+1.5,f'{np.mean(values):.2f} cm',ha='center',weight='bold')
        ax.set_xticks(range(3),['Original','Seed 42','Seed 43'])
        ax.set_title(title)
        ax.set_ylabel('Altitude RMSE (cm)')
        ax.grid(axis='y',alpha=.18)
    fig.suptitle('Held-out evaluation: 20 unused seeds per scenario',fontsize=15,weight='bold')
    fig.supxlabel('Bars: completed-run means; dots: individual completed runs. Randomized completion: original 20/20, seed 42 19/20, seed 43 20/20.',fontsize=9)
    fig.savefig(HERE/'heldout-comparison.png',dpi=170)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for seed,key in [(42,'seed42'),(43,'seed43')]:
        rows=json.loads((HERE/f'training_seed{seed}'/'progress.json').read_text())
        rows=[r for r in rows if r['label']!='1015808']
        x=[r['additional_timesteps']/1e6 for r in rows]
        axes[0].plot(x,[r['summary']['hover30']['completed_mean_altitude_rmse_m']*100 for r in rows],'-o',label=f'Seed {seed}',color=COLORS[key])
        axes[1].plot(x,[r['score'][0] for r in rows],'-o',label=f'Seed {seed}',color=COLORS[key])
    axes[0].axhline(5,ls='--',color='#777777',lw=1,label='5 cm target')
    axes[0].set_ylabel('Nominal-hover mean altitude RMSE (cm)')
    axes[1].set_ylabel('Episodes passing all criteria / 35')
    axes[1].set_ylim(0,35)
    for ax in axes:
        ax.set_xlabel('Additional training transitions (millions)')
        ax.grid(alpha=.18)
        ax.legend()
    fig.suptitle('More training did not consistently improve the fixed development suite',fontsize=13,weight='bold')
    fig.savefig(HERE/'training-progress.png',dpi=170)
    plt.close(fig)
    print(json.dumps({k:v for k,v in comparison.items() if k.startswith('total_')}))


if __name__=='__main__':main()
