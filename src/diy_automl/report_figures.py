"""Figure plans keep plotting inputs separate from drawing and workbook editing."""
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix,adjusted_rand_score
from .supervised import metrics


class Plans:
    def __init__(self):
        self.figures=[]
        self.skipped=[]

    def add(self,id,kind,sheet,title,scope,data,**extra):
        self.figures.append(dict(id=id,kind=kind,sheet=sheet,title=title,scope=scope,data=data,**extra))

    def skip(self,id,reason):
        self.skipped.append(dict(id=id,reason=reason))

    def confusion(self,id,sheet,title,truth,pred,classes,scope,expected=None,**extra):
        if truth is None:
            self.skip(id,'Missing true labels: supply the original CSV with matching SHA256')
            return
        if len(classes)>30:
            self.skip(id,'Confusion matrices support at most 30 classes')
            return
        truth=np.asarray(truth);pred=np.asarray(pred)
        labels=np.arange(len(classes))
        if truth.shape!=pred.shape or not np.isin(truth,labels).all() or not np.isin(pred,labels).all():
            raise ValueError('Invalid saved labels for confusion matrix')
        if expected is not None:
            estimates=metrics(truth,pred,labels)
            if any(not np.isclose(expected[key],value,atol=1e-12,rtol=0) for key,value in estimates.items()):
                raise ValueError('Saved classification scores differ from figure predictions')
        self.add(id,'confusion',sheet,title,scope,{'counts':confusion_matrix(truth,pred,labels=labels)},
                 labels=[f'{i}: {str(c)[:24]}' for i,c in enumerate(classes)],**extra)


def numeric_group(frame,models,x,ys,errors=None):
    data={}
    for index,model in enumerate(models):
        group=frame[frame.Model==model].sort_values(x,kind='stable')
        data[f'x_{index}']=group[x].to_numpy(dtype=float)
        for j,y in enumerate(ys):
            data[f'y_{index}_{j}']=group[y].to_numpy(dtype=float)
            if errors is not None:
                data[f'sd_{index}_{j}']=group[errors[j]].to_numpy(dtype=float)
    return data


def build_plans(manifest,sheets,arrays,max_models=3,raw=None):
    p=Plans();schema=manifest['schema_version'];classes=manifest['classes']
    raw=raw or {}
    if schema==2:
        target=raw.get('target',arrays.get('target',arrays.get('ul_truth')))
        if 'SL_Result' in sheets:
            if sheets['SL_Result'].Model.tolist()!=list(manifest.get('sl_models',{})):
                raise ValueError('SL table order differs from saved prediction identities')
            for i,row in sheets['SL_Result'].iloc[:max_models].iterrows():
                truth=target[arrays['sl_test']] if target is not None else None
                p.confusion(f'sl_cm_{i}','SL_Result',str(row.Model),truth,arrays[f'sl_prediction_{i}'],classes,
                            'Exploratory holdout: display in declared order, not test ranking',expected=row,source_row=int(i))
        if 'UL_Result' in sheets:
            table=sheets['UL_Result']
            from .clustering import models
            identities=[(name,k) for name,k,_ in models(manifest['settings']['seed'])]
            if list(zip(table.Model,table.Cluster))!=identities:
                raise ValueError('UL table order differs from saved label identities')
            selected=table[np.isfinite(table.Silhouette)].sort_values('Silhouette',ascending=False,kind='stable').iloc[:max_models]
            X=raw.get('features',arrays.get('features'))
            coords=explained=None;projection='unavailable'
            if X is not None and X.ndim==2 and X.shape[0]<=5000 and X.shape[1]<=100:
                Z=(X-arrays['ul_scaler_mean'])/arrays['ul_scaler_scale']
                if X.shape[1]==1:
                    coords=np.column_stack([Z[:,0],np.zeros(len(Z))]);projection='single_feature'
                elif np.all(np.ptp(Z,axis=0)==0):
                    coords=np.zeros((len(Z),2));projection='constant'
                else:
                    model=PCA(n_components=2,svd_solver='full')
                    coords=model.fit_transform(Z);explained=model.explained_variance_ratio_;projection='pca'
            for i,row in selected.iterrows():
                labels=arrays[f'ul_labels_{i}']
                if target is not None and (len(labels)!=len(target) or not np.isclose(row.ARI,adjusted_rand_score(target,labels),rtol=0,atol=1e-12)):
                    raise ValueError('Saved target ARI differs from figure labels')
                data={'metrics':row[['Silhouette','Silhouette_SD','Silhouette_SE','Davies-Bouldin','Calinski-Harabasz','ARI']].to_numpy(dtype=float)}
                if coords is not None:
                    if len(coords)!=len(labels): raise ValueError('UL projection/label row mismatch')
                    data.update(coords=coords,labels=labels)
                if explained is not None: data['explained']=explained
                p.add(f'ul_candidate_{i}','cluster','UL_Result',f'{row.Model}, k={int(row.Cluster)}',
                      'Display only by internal Silhouette; target ARI is diagnostic, not selection',data,
                      source_row=int(i),projection=projection,
                      projection_note='Raw features unavailable or projection exceeds 5000 rows / 100 features' if coords is None else '')
            if selected.empty: p.skip('ul_candidates','No finite Silhouette candidates to display')
        if 'DL_Result' in sheets:
            truth=target[arrays['dl_test']] if target is not None else None
            p.confusion('dl_cm','DL_Result','MLP 32-16-8',truth,arrays['dl_predictions'],classes,
                        'Single fixed configuration; final test only, no best-epoch selection',expected=sheets['DL_Result'].iloc[0])
            history=sheets['DL_History']
            p.add('dl_history','history','DL_History','MLP training / validation',
                  'Learning history only; test is not used for epoch selection',
                  {key:history[key].to_numpy(dtype=float) for key in ['Epoch','accuracy','val_accuracy','loss','val_loss']})
    elif schema in [3,7]:
        tuned=schema==7
        title=manifest['selected_model']
        if sheets['SL_Test'].Model.tolist()!=[title]: raise ValueError('Selected test model identity differs')
        if tuned: title=f'Candidate {manifest["selected_candidate"]}: '+title
        p.confusion('sl_selected_cm','SL_Test',title,arrays['truth'],arrays['predictions'],classes,
                    'Chosen using training CV only; this is the only tested configuration',expected=sheets['SL_Test'].iloc[0])
        mean=sheets['SL_Search_Mean' if tuned else 'SL_CV_Mean']
        sd=sheets['SL_Search_SD' if tuned else 'SL_CV_SD']
        metric=manifest['settings']['metric']
        chosen=int(manifest['selected_candidate']) if tuned else mean.Model.tolist().index(manifest['selected_model'])
        labels=[f'{int(row.Candidate)}: {row.Model}' if tuned else str(row.Model) for _,row in mean.iterrows()]
        p.add('sl_training_cv','cv','SL_Search_Mean' if tuned else 'SL_CV_Mean',f'Training CV mean {metric}',
              'Shared training folds; error bars are fold SD, not CI; display order is declared order',
              {'mean':mean[metric].to_numpy(dtype=float),'sd':sd[metric].to_numpy(dtype=float)},labels=labels,selected=chosen)
    elif schema==4:
        frame=sheets['UL_Seed_Summary'];models=frame.Model.drop_duplicates().tolist()
        p.add('ul_seed_summary','grouped','UL_Seed_Summary','Fixed-data initialization sensitivity',
              'Error bars: initialization-seed SD; no confidence intervals or automatic choice',
              numeric_group(frame,models,'Cluster',['Target ARI_mean','Silhouette_mean'],['Target ARI_std','Silhouette_std']),
              labels=models,ylabels=['Target ARI','Silhouette'],xlabel='k',with_sd=True)
        frame=sheets['UL_Pair_Summary'];models=frame.Model.drop_duplicates().tolist()
        p.add('ul_seed_pairs','grouped','UL_Pair_Summary','Between-seed assignment agreement',
              'Dependent pairs: descriptive mean and SD only, no SEM/CI',
              numeric_group(frame,models,'Cluster',['mean'],['std']),labels=models,ylabels=['Assignment ARI'],xlabel='k',with_sd=True)
    elif schema==5:
        frame=sheets['DL_History'];data={};seeds=manifest['settings']['training_seeds']
        for i,seed in enumerate(seeds):
            group=frame[frame['Training Seed']==seed]
            for key in ['Epoch','accuracy','val_accuracy','loss','val_loss']:
                data[f'{key}_{i}']=group[key].to_numpy(dtype=float)
        p.add('dl_all_histories','repeated_history','DL_History','All requested training seeds',
              'Fixed split; every seed shown, no best test seed or epoch choice',data,labels=[str(seed) for seed in seeds])
        frame=sheets['DL_By_Seed']
        p.add('dl_seed_metrics','seed_metrics','DL_By_Seed','Conditional training-seed sensitivity',
              'All seeds in requested order; fixed data and split; no CI and no seed selection',
              {'seed':frame['Training Seed'].to_numpy(dtype=float),**{key:frame[key].to_numpy(dtype=float) for key in ['Accuracy','Precision','Recall','F1']}})
    elif schema==6:
        frame=sheets['UL_Sample_Summary'];models=frame.Model.drop_duplicates().tolist()
        p.add('ul_subsample_reference','grouped','UL_Sample_Summary','Row-omission sensitivity',
              'Subset-specific scaling; valid reference comparisons only; subset SD, not CI',
              numeric_group(frame,models,'Cluster',['Reference ARI_Mean','Target ARI_Mean'],
                            ['Reference ARI_Across_Subsamples_SD','Target ARI_Across_Subsamples_SD']),
              labels=models,ylabels=['Reference ARI (valid comparisons)','Target ARI'],xlabel='k',with_sd=True)
        frame=sheets['UL_Overlap_Summary'];models=frame.Model.drop_duplicates().tolist()
        p.add('ul_subsample_pairs','grouped','UL_Overlap_Summary','Shared-row assignment agreement',
              'Valid shared-row comparisons only; dependent pair SD, not CI; unscored pairs excluded',
              numeric_group(frame,models,'Cluster',['Mean'],['SD']),labels=models,ylabels=['Assignment ARI'],xlabel='k',with_sd=True)
    else:
        raise ValueError('Excel figures support completed source schemas 2 through 7')
    dataset=Path(str(manifest['settings'].get('csv','dataset')).replace('\\','/')).stem[:40]
    for figure in p.figures: figure['title']=dataset+' | '+figure['title']
    return p.figures,p.skipped


def draw_figure(plan,path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib import font_manager,rc_context
    names={font.name for font in font_manager.fontManager.ttflist}
    family='Noto Sans CJK JP' if 'Noto Sans CJK JP' in names else 'DejaVu Sans'
    with rc_context({'font.family':family,'font.size':10}):
        return _draw(plan,path,Figure,FigureCanvasAgg,family)


def _draw(plan,path,Figure,Canvas,family):
    kind=plan['kind'];d=plan['data']
    height=max(4.,len(plan.get('labels',[]))*.34+1.5) if kind=='cv' else 5.4
    fig=Figure(figsize=(9.,height),layout='constrained');Canvas(fig)
    if kind=='confusion':
        ax=fig.subplots();counts=d['counts']
        image=ax.imshow(counts,cmap='Blues',vmin=0)
        ax.set(xticks=np.arange(len(counts)),yticks=np.arange(len(counts)),
               xticklabels=plan['labels'],yticklabels=plan['labels'],xlabel='Predicted class',ylabel='True class')
        ax.tick_params(axis='x',labelrotation=45)
        for (r,c),value in np.ndenumerate(counts):
            ax.text(c,r,str(int(value)),ha='center',va='center',color='white' if value>counts.max()/2 else 'black',fontsize=max(6,11-len(counts)//5))
        fig.colorbar(image,ax=ax,shrink=.8)
    elif kind=='cv':
        ax=fig.subplots();idx=np.arange(len(d['mean']))
        ax.barh(idx,d['mean'],xerr=d['sd'],color=['#198c83' if i==plan['selected'] else '#779fca' for i in idx],capsize=3)
        ax.set(yticks=idx,yticklabels=plan['labels'],xlabel='CV mean (error bars: fold SD, not CI)')
        ax.invert_yaxis();ax.set_xlim(left=0,right=max(1.02,float(np.nanmax(d['mean']+np.nan_to_num(d['sd'])))*1.03))
    elif kind=='cluster':
        ax,notes=fig.subplots(1,2,gridspec_kw={'width_ratios':[1.35,1.]})
        if 'coords' in d:
            points=d['coords'];dots=ax.scatter(points[:,0],points[:,1],c=d['labels'],cmap='tab10',s=15,alpha=.8,vmin=0,vmax=9)
            handles,labels=dots.legend_elements(num=None)
            ax.legend(handles,labels,title='Cluster',fontsize=7,ncol=2)
            if plan['projection']=='pca':
                ax.set(xlabel=f'PC1 ({d["explained"][0]:.1%})',ylabel=f'PC2 ({d["explained"][1]:.1%})')
            elif plan['projection']=='single_feature': ax.set(xlabel='Single standardized feature',ylabel='Zero axis (not PCA)')
            else: ax.set(xlabel='No variation after scaling',ylabel='No PCA separation')
        else:
            ax.text(.5,.5,'Projection unavailable\n'+plan['projection_note'],ha='center',va='center',wrap=True,transform=ax.transAxes);ax.set_axis_off()
        notes.set_axis_off()
        labels=['Silhouette','Pointwise Silhouette SD','Pointwise Silhouette SE','Davies-Bouldin','Calinski-Harabasz','Target ARI (diagnostic)']
        text='\n\n'.join(f'{label}: {value:.5g}' for label,value in zip(labels,d['metrics']))
        notes.text(0,.98,text,va='top',transform=notes.transAxes)
    elif kind in ['history','repeated_history']:
        axes=fig.subplots(1,2)
        for ax,metric in zip(axes,['accuracy','loss']):
            if kind=='history':
                ax.plot(d['Epoch'],d[metric],label='train');ax.plot(d['Epoch'],d['val_'+metric],label='validation')
            else:
                for i,label in enumerate(plan['labels']):
                    line,=ax.plot(d[f'Epoch_{i}'],d[f'{metric}_{i}'],alpha=.7,label=str(label))
                    ax.plot(d[f'Epoch_{i}'],d[f'val_{metric}_{i}'],color=line.get_color(),linestyle='--',alpha=.7)
            ax.set(xlabel='Epoch',ylabel=metric)
            if metric=='accuracy': ax.set_ylim(0,1.03)
            else: ax.set_ylim(bottom=0)
            if kind=='repeated_history': ax.set_title('Solid: train; dashed: validation',fontsize=9)
            ax.legend(fontsize=6,ncol=2 if kind=='repeated_history' else 1)
    elif kind=='seed_metrics':
        axes=fig.subplots(2,2)
        for ax,metric in zip(axes.flat,['Accuracy','Precision','Recall','F1']):
            index=np.arange(len(d['seed']))
            ax.plot(index,d[metric],marker='o');ax.set(xticks=index,xticklabels=[str(int(s)) for s in d['seed']],xlabel='Training seed (requested order)',ylabel=metric,ylim=(0,1.03))
    elif kind=='grouped':
        axes=np.atleast_1d(fig.subplots(1,len(plan['ylabels'])))
        for j,(ax,label) in enumerate(zip(axes,plan['ylabels'])):
            drawn=False
            for i,model in enumerate(plan['labels']):
                x=d[f'x_{i}'];y=d[f'y_{i}_{j}'];sd=d[f'sd_{i}_{j}']
                finite=np.isfinite(y)
                if finite.any():
                    line,=ax.plot(x[finite],y[finite],marker='o',label=model)
                    defined=finite & np.isfinite(sd)
                    if defined.any(): ax.errorbar(x[defined],y[defined],yerr=sd[defined],fmt='none',capsize=2,color=line.get_color())
                    drawn=True
            ax.set(xlabel=plan['xlabel'],ylabel=label)
            if drawn: ax.legend(fontsize=7)
            else: ax.text(.5,.5,'No valid comparisons',ha='center',transform=ax.transAxes)
    else:
        raise ValueError('Unknown figure kind')
    from textwrap import fill
    fig.suptitle(fill(plan['title'],90)+'\n'+fill(plan['scope'],110),fontsize=11)
    fig.savefig(path,dpi=130)
    fig.clear()
    return family
