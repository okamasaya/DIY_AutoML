"""Validate schema-9 bootstrap inputs, samples, tables and embedded PNG without refitting."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from .metric_uncertainty import (MAX_CLASSES, MAX_DRAWS, MAX_EVALUATIONS, MAX_ROWS,
                                 MAX_SCORE_VALUES, configuration)
from .supervised import metrics
from .ul_subsample_validation import require, same_table


def validate_uncertainty(directory):
    from .uncertainty_workflow import SCOPE, evaluation_descriptors, make_tables
    root=Path(directory)
    m=json.loads((root/'manifest.json').read_text())
    require(m['schema_version']==9 and m['status']=='completed' and m['stage']=='completed',
            'Uncertainty evaluation did not complete')
    names={'report.xlsx','arrays.npz','source_manifest.json','source_report.xlsx','metric_intervals.png'}
    require(set(m['artifacts'])==names,'Uncertainty artifact set differs')
    for name,digest in m['artifacts'].items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,f'Artifact checksum mismatch: {name}')
    original=json.loads((root/'source_manifest.json').read_text())
    require(original['schema_version']==m['source_schema'] and original['schema_version'] in (2,3,5,7)
            and original['status']=='completed' and original['version']==m['source_version'], 'Invalid source snapshot')
    require(m['source_artifacts']['manifest.json']==m['artifacts']['source_manifest.json']
            and m['source_artifacts']['report.xlsx']==m['artifacts']['source_report.xlsx'], 'Source snapshot checksum differs')
    require(m['source_artifacts']['arrays.npz']==original['artifacts']['arrays.npz'], 'Source array identity differs')
    require(m['classes']==original['classes'] and m['scope']==SCOPE,'Class mapping or scope differs')
    settings=m['settings']
    configuration(settings['bootstrap_repeats'],settings['confidence'],settings['bootstrap_seed'],settings['resampling'])
    require(settings['limits']==dict(rows=MAX_ROWS,classes=MAX_CLASSES,evaluations=MAX_EVALUATIONS,
                                    draws=MAX_DRAWS,score_values=MAX_SCORE_VALUES),'Resource limits differ')
    with np.load(root/'arrays.npz',allow_pickle=False) as saved:
        arrays={key:saved[key] for key in saved.files}
    original_sheets=pd.read_excel(root/'source_report.xlsx',sheet_name=None)
    require(m['evaluations']==evaluation_descriptors(original,original_sheets),'Evaluation model, role or order differs')
    evaluations=[]
    require(1<=len(m['evaluations'])<=20,'Invalid evaluation count')
    for i,e in enumerate(m['evaluations']):
        require(e['id']==f'evaluation_{i}','Evaluation order differs')
        prefix=e['id']+'__'
        values={k:arrays[prefix+k] for k in ('truth','predictions','train','test','validation')}
        y,pred=values['truth'],values['predictions']
        split=[values[k] for k in ('train','test','validation')]
        require(all(a.ndim==1 and np.issubdtype(a.dtype,np.integer) for a in split), 'Invalid split types')
        all_indices=np.concatenate(split)
        require(len(values['train'])>0 and len(values['test'])==len(y)
                and (all_indices>=0).all() and len(np.unique(all_indices))==len(all_indices),'Split overlap or size differs')
        row=original_sheets[e['source_sheet']].iloc[e['source_row']]
        for metric,value in metrics(y,pred,np.arange(len(m['classes']))).items():
            require(np.isclose(row[metric],value,atol=1e-12,rtol=0),'Frozen source metric differs from predictions')
        evaluations.append({**e,**values})
    # Reconstruct every draw using the recorded seed and recompute every result.
    expected,expected_arrays=make_tables(evaluations,m['classes'],settings)
    require(set(arrays)==set(expected_arrays),'Unexpected uncertainty arrays')
    for key,value in expected_arrays.items():
        require(np.array_equal(arrays[key],value),f'Bootstrap array differs: {key}')
    sheets=pd.read_excel(root/'report.xlsx',sheet_name=None)
    require(set(sheets)==set(expected)|{'Intervals'},'Uncertainty sheet names differ')
    require({k:len(v) for k,v in sheets.items()}==m['sheets'],'Uncertainty sheet row counts differ')
    for name,frame in expected.items(): same_table(sheets[name],frame,f'Uncertainty table differs: {name}')
    require(sheets['Intervals'].columns.tolist()==['Description'] and len(sheets['Intervals'])==1,'Interval caption differs')
    with ZipFile(root/'report.xlsx') as book:
        images=[name for name in book.namelist() if name.startswith('xl/media/')]
        require(len(images)==1 and book.read(images[0])==(root/'metric_intervals.png').read_bytes(),'Embedded interval PNG differs')
    book=load_workbook(root/'report.xlsx')
    try:
        require(sum(len(sheet._images) for sheet in book)==1 and len(book['Intervals']._images)==1,'Image sheet differs')
        anchor=book['Intervals']._images[0].anchor._from
        require((anchor.col,anchor.row)==(0,3),'Image anchor differs')
    finally: book.close()
    require(m['figure_count']==1,'Figure count differs')
    return m
