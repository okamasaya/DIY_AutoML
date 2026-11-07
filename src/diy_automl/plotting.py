"""Five original plot types, using the common workflow's saved tables."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_run(directory):
    root=Path(directory)
    sheets=pd.read_excel(root/"report.xlsx",sheet_name=None)
    figures=[]
    if 'UL_Result' in sheets:
        km=sheets['UL_Result'].query("Model == 'KMeans'").sort_values('Cluster')
        fig,ax=plt.subplots()
        ax.errorbar(km.Cluster,km.Silhouette,yerr=km.Silhouette_SD,marker='o')
        ax.set(xlabel='k',ylabel='Silhouette',title='KMeans: pointwise SD (not seed variability or CI)')
        figures.append(('ul_silhouette',fig))
        fig,axes=plt.subplots(3,1,figsize=(7,9),sharex=True)
        for ax,metric in zip(axes,['Silhouette','Davies-Bouldin','Calinski-Harabasz']):
            ax.plot(km.Cluster,km[metric],marker='o');ax.set_ylabel(metric)
        axes[-1].set_xlabel('k');fig.tight_layout();figures.append(('ul_internal',fig))
        fig,ax=plt.subplots()
        for metric in ['ARI','NMI']: ax.plot(km.Cluster,km[metric],marker='o',label=metric)
        ax.set(xlabel='k',ylabel='Target agreement',title='KMeans: external diagnostics');ax.legend()
        figures.append(('ul_external',fig))
    if 'DL_History' in sheets:
        history=sheets['DL_History']
        for metric in ['accuracy','loss']:
            fig,ax=plt.subplots()
            ax.plot(history.Epoch,history[metric],label='train')
            ax.plot(history.Epoch,history[f'val_{metric}'],label='validation')
            ax.set(xlabel='Epoch',ylabel=metric,title=f'MLP {metric}');ax.legend()
            figures.append((f'dl_{metric}',fig))
    dest=root/'figures';dest.mkdir(exist_ok=True)
    for name,fig in figures: fig.savefig(dest/f'{name}.png',dpi=120,bbox_inches='tight')
    return figures
