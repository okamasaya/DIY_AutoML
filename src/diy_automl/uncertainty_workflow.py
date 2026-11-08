"""Read immutable saved predictions and produce a separate schema-9 uncertainty report."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image

from .excel_report import resolve_data
from .metric_uncertainty import (MAX_CLASSES, MAX_DRAWS, MAX_EVALUATIONS, MAX_ROWS,
                                 MAX_SCORE_VALUES, configuration, evaluate_metric_uncertainty,
                                 observations)
from .reporting import fail, prepare, save, summary, write_json
from .supervised import METRICS, candidates, metrics
from .validation import validate

SCOPE = {
    'Target': 'Sampling uncertainty of each fixed fitted model; no model or seed selection',
    'Unit': 'Rows assumed independent and representative; not verified by this report',
    'SE': 'Sample SD of bootstrap estimates (ddof=1); not SD/sqrt(replicates)',
    'Interval': 'Marginal paired percentile interval; linear quantiles; no simultaneous coverage claim',
    'Macro': 'Fixed full class list; zero_division=0; macro F1 is mean of class F1 values',
    'Excluded': 'Training/search/refit variability; group/time dependence; distribution shift; systematic and numerical error',
    'Zero width': 'Does not imply no unknown uncertainty; absent classes contain no recall information',
    'Probability': 'Metric confidence intervals are not probabilities of individual predictions',
    'Comparison': 'No significance test, automatic recommendation, or ranking by test scores',
    'Data': 'Previously examined datasets remain development examples, not new external validation',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluation_descriptors(manifest, sheets):
    specs=[]
    def add(model,role,sheet,row):
        specs.append(dict(id=f'evaluation_{len(specs)}',model=model,role=role,source_sheet=sheet,source_row=row))
    schema=manifest['schema_version']
    if schema==2:
        if 'SL_Result' in sheets:
            for i,(name,_) in enumerate(candidates()):
                add(name,'exploratory SL; declared order; not test ranking','SL_Result',i)
        if 'DL_Result' in sheets: add('MLP','fixed fitted DL model','DL_Result',0)
    elif schema in (3,7):
        add(manifest['selected_model'],'selected using training CV only','SL_Test',0)
    elif schema==5:
        for i,seed in enumerate(manifest['settings']['training_seeds']):
            add(f'MLP seed {seed}','fixed training seed; all seeds retained; not selection','DL_By_Seed',i)
    return specs


def extract_evaluations(root, manifest, arrays, sheets, csv=None):
    schema = manifest['schema_version']
    if schema not in (2, 3, 5, 7):
        raise ValueError('Uncertainty supports classification source schemas 2, 3, 5 and 7; UL uses separate stability diagnostics')
    labels = np.arange(len(manifest['classes']))
    result = []
    def add(model, role, truth, prediction, train, test, sheet, row, validation=()):
        y, pred, _ = observations(truth, prediction, labels)
        groups = [np.asarray(v) for v in (train, test, validation)]
        if any(v.size and not np.issubdtype(v.dtype,np.integer) for v in groups):
            raise ValueError('Evaluation split indices must be integers')
        train, test, validation = [v.astype(np.int64) for v in groups]
        all_indices = np.concatenate([train, test, validation])
        if (any(v.ndim != 1 for v in (train,test,validation)) or not len(train) or len(test) != len(y)
                or (all_indices < 0).any() or len(np.unique(all_indices)) != len(all_indices)):
            raise ValueError('Invalid, duplicate or overlapping evaluation split indices')
        stored = sheets[sheet].iloc[row]
        for metric, value in metrics(y, pred, labels).items():
            if not np.isclose(float(stored[metric]), value, rtol=0, atol=1e-12):
                raise ValueError('Source metric differs from saved predictions')
        result.append({'id': f'evaluation_{len(result)}', 'model': model, 'role': role,
                       'source_sheet': sheet, 'source_row': row, 'truth': y, 'predictions': pred,
                       'train': train, 'test': test, 'validation': validation})
    if schema == 2:
        raw, _ = resolve_data(root, manifest, arrays, csv)
        if 'target' not in raw:
            raise ValueError('Source lacks target labels; provide the original hash-matching --csv')
        y = raw['target']
        if 'SL_Result' in sheets:
            rows = sheets['SL_Result']
            names = [name for name, _ in candidates()]
            column = 'Model' if 'Model' in rows else 'Method'
            if rows[column].tolist() != names:
                raise ValueError('Source SL candidate order differs')
            for i, name in enumerate(names):
                add(name, 'exploratory SL; declared order; not test ranking', y[arrays['sl_test']],
                    arrays[f'sl_prediction_{i}'], arrays['sl_train'], arrays['sl_test'], 'SL_Result', i)
        if 'DL_Result' in sheets:
            add('MLP', 'fixed fitted DL model', y[arrays['dl_test']], arrays['dl_predictions'],
                arrays['dl_train'], arrays['dl_test'], 'DL_Result', 0, arrays['dl_validation'])
    elif schema in (3,7):
        name = manifest['selected_model']
        if name not in [n for n,_ in candidates()] or sheets['SL_Test'].iloc[0].Model != name:
            raise ValueError('Source selected model differs')
        add(name, 'selected using training CV only', arrays['truth'], arrays['predictions'],
            arrays['train'], arrays['test'], 'SL_Test', 0)
    else:
        for row, seed in enumerate(manifest['settings']['training_seeds']):
            add(f'MLP seed {seed}', 'fixed training seed; all seeds retained; not selection',
                arrays['truth'], arrays[f'seed_{seed}_predictions'], arrays['train'], arrays['test'],
                'DL_By_Seed', row, arrays['validation'])
    if not result:
        raise ValueError('No saved classification predictions; UL metrics require separate stability diagnostics')
    if len(result) > MAX_EVALUATIONS:
        raise ValueError(f'At most {MAX_EVALUATIONS} evaluations are supported')
    if [{k:e[k] for k in ('id','model','role','source_sheet','source_row')} for e in result] != evaluation_descriptors(manifest,sheets):
        raise ValueError('Source evaluation identity differs')
    return result


def make_tables(evaluations, class_labels, settings):
    overall, classwise, bootstrap, units, arrays = [], [], [], [], {}
    c = len(class_labels)
    repeats, confidence, seed, resampling = configuration(
        settings['bootstrap_repeats'], settings['confidence'], settings['bootstrap_seed'], settings['resampling'])
    if (repeats * sum(len(e['truth']) for e in evaluations) > MAX_DRAWS
            or repeats * len(evaluations) * (4+3*c) > MAX_SCORE_VALUES):
        raise ValueError('Combined bootstrap draw or score storage limit exceeded')
    for e in evaluations:
        label = {'Evaluation': e['id'], 'Model': e['model'], 'Role': e['role']}
        scores, classes, data = evaluate_metric_uncertainty(
            e['truth'], e['predictions'], np.arange(c), repeats, confidence, seed, resampling)
        for frame in (scores, classes):
            for position,(key,value) in enumerate(label.items()):
                frame.insert(position,key,value)
        classes.insert(4, 'Class Label', [f'class {i}: {class_labels[i]}' for i in classes['Encoded Class']])
        overall.append(scores); classwise.append(classes)
        frame = pd.DataFrame(data['samples'], columns=METRICS)
        frame.insert(0,'Replicate',np.arange(repeats))
        for position,(key,value) in enumerate(label.items()): frame.insert(position,key,value)
        bootstrap.append(frame)
        support = np.bincount(e['truth'],minlength=c)
        notes = ['Row independence and representativeness unverified']
        if (support == 0).any(): notes.append('Class absent in test; zero_division=0 is a convention, not evidence')
        if (support < 10).any(): notes.append('At least one class has fewer than 10 observed test rows')
        if repeats < 1000: notes.append('Fewer than 1000 replicates; tail quantiles can be coarse')
        if repeats*(1-confidence)/2 < 20: notes.append('Fewer than 20 expected replicates per percentile tail')
        units.append({**label, 'Test Rows':len(e['truth']), 'Training Rows':len(e['train']),
                      'Validation Rows':len(e['validation']), 'Classes':c,
                      'Observed Test Classes':int((support>0).sum()),
                      'Minimum Class Support':int(support.min()), 'Unit':'row', 'Notes':'; '.join(notes)})
        arrays.update({e['id']+'__'+key:value for key,value in data.items()})
        arrays.update({e['id']+'__'+key:e[key] for key in ('train','test','validation')})
    scope = dict(SCOPE)
    scope['Resampling'] = ('Class-count-fixed stratified sampling; excludes prevalence variation'
                           if resampling == 'stratified' else
                           'Ordinary row sampling; class counts vary; absent resampled classes use zero_division=0')
    sheets = {'Metric_Uncertainty':pd.concat(overall,ignore_index=True),
              'Class_Uncertainty':pd.concat(classwise,ignore_index=True),
              'Evaluation_Units':pd.DataFrame(units), 'Bootstrap':pd.concat(bootstrap,ignore_index=True),
              'Scope':summary(list(scope.items())),
              'Summary':summary([('Purpose','Fixed-model metric uncertainty; no retraining'),
                                 ('Evaluations',len(evaluations)), ('Replicates',repeats),
                                 ('Confidence',confidence), ('Resampling',resampling),
                                 ('SE','Bootstrap SD, not SD/sqrt(B)'),
                                 ('Limitations','See Scope and Evaluation_Units; no quality certification')])}
    return sheets, arrays


def draw_intervals(frame, destination):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(13, max(6, 3+len(frame)//4*.32)), layout='constrained')
    FigureCanvasAgg(figure)
    for metric, ax in zip(METRICS, figure.subplots(2,2).flat):
        rows = frame[frame.Metric == metric]
        positions = np.arange(len(rows))
        ax.hlines(positions, rows.Lower, rows.Upper, color='#2574a9', linewidth=2)
        ax.scatter(rows.Estimate, positions, color='#162f45', s=22, zorder=3)
        ax.set_yticks(positions, [f'{e}: {m}' for e,m in zip(rows.Evaluation,rows.Model)], fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(-.02,1.02); ax.set_title(metric)
        ax.grid(axis='x',alpha=.2)
    confidence = frame.Confidence.iloc[0]
    figure.suptitle(f'Fixed-model metric estimates and {confidence:.1%} marginal percentile intervals\n'
                   'Row-bootstrap assumptions apply; source order retained; not a ranking', fontsize=11)
    figure.savefig(destination,dpi=120,metadata={'Software':'diy-automl metric uncertainty'})
    figure.clear()


def evaluate_uncertainty(source, output, bootstrap_repeats=1000, confidence=.95,
                         bootstrap_seed=43, resampling='stratified', csv=None):
    repeats, confidence, seed, resampling = configuration(bootstrap_repeats, confidence, bootstrap_seed, resampling)
    source, output = Path(source), Path(output)
    if output.resolve().is_relative_to(source.resolve()):
        raise ValueError('Output must be separate from the source run')
    manifest = validate(source)
    source_digests = {name:digest(source/name) for name in ('manifest.json','report.xlsx','arrays.npz')}
    with np.load(source/'arrays.npz',allow_pickle=False) as stored:
        source_arrays = {key:stored[key] for key in stored.files}
    source_sheets = pd.read_excel(source/'report.xlsx',sheet_name=None)
    evaluations = extract_evaluations(source,manifest,source_arrays,source_sheets,csv)
    settings = dict(source=str(source.resolve()), bootstrap_repeats=repeats, confidence=confidence,
                    bootstrap_seed=seed, resampling=resampling,
                    limits=dict(rows=MAX_ROWS,classes=MAX_CLASSES,evaluations=MAX_EVALUATIONS,
                                draws=MAX_DRAWS,score_values=MAX_SCORE_VALUES))
    # Check aggregate limits before creating output or allocating bootstrap arrays.
    if (repeats*sum(len(e['truth']) for e in evaluations)>MAX_DRAWS
            or repeats*len(evaluations)*(4+3*len(manifest['classes']))>MAX_SCORE_VALUES):
        raise ValueError('Combined bootstrap draw or score storage limit exceeded')
    out, record = prepare(output,9,settings)
    try:
        record.update(source_schema=manifest['schema_version'],source_version=manifest['version'],
                      source_artifacts=source_digests,classes=manifest['classes'],
                      evaluations=[{k:e[k] for k in ('id','model','role','source_sheet','source_row')} for e in evaluations],
                      scope=SCOPE, stage='bootstrap')
        write_json(out/'manifest.json',record)
        shutil.copy2(source/'manifest.json',out/'source_manifest.json')
        shutil.copy2(source/'report.xlsx',out/'source_report.xlsx')
        sheets, arrays = make_tables(evaluations,manifest['classes'],settings)
        save(out,record,sheets,arrays,finalize=False)
        record['stage']='plot'; write_json(out/'manifest.json',record)
        draw_intervals(sheets['Metric_Uncertainty'],out/'metric_intervals.png')
        book = load_workbook(out/'report.xlsx')
        plot = book.create_sheet('Intervals')
        plot['A1']='Description'
        plot['A2']='Fixed-model marginal percentile intervals; assumptions and exclusions: see Scope.'
        image=Image(out/'metric_intervals.png')
        image.height=round(image.height*1000/image.width); image.width=1000
        plot.add_image(image,'A4'); book.save(out/'report.xlsx'); book.close()
        record['sheets']['Intervals']=1
        if any(digest(source/name)!=value for name,value in source_digests.items()):
            raise ValueError('Source changed during uncertainty evaluation')
        record.update(status='completed',stage='completed',figure_count=1,
                      artifacts={name:digest(out/name) for name in ('report.xlsx','arrays.npz','source_manifest.json',
                                                                   'source_report.xlsx','metric_intervals.png')})
        write_json(out/'manifest.json',record)
        return out
    except Exception as error:
        fail(out,record,error)
        raise
