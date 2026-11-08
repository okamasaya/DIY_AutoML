"""Render an immutable saved run into a separate, self-contained illustrated workbook."""
import hashlib
import importlib.metadata
import platform
from datetime import datetime,timezone
from math import ceil
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.drawing.image import Image
from openpyxl.utils import get_column_letter

from .clustering_subsample import integer
from .data import load_csv
from .report_figures import build_plans,draw_figure
from .reporting import source_hashes,write_json
from .validation import validate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve_data(root,manifest,arrays,csv=None,auto_csv=True):
    """Only a hash-matching CSV can supplement a historical run without raw data."""
    raw={key:arrays[key] for key in ['features','target'] if key in arrays}
    chosen=None
    if auto_csv and csv is None and manifest['schema_version']==2 and ('features' not in raw or 'target' not in raw):
        candidate=Path(manifest['settings']['csv'])
        if candidate.is_file(): csv=candidate
    if csv is not None:
        path=Path(csv)
        if digest(path)!=manifest.get('input_sha256'): raise ValueError('CSV SHA256 differs from the saved input')
        data=load_csv(path,manifest['settings']['target'])
        if data.classes!=manifest['classes'] or data.feature_names!=manifest['features']:
            raise ValueError('CSV label or feature mapping differs from the saved run')
        for key,value in [('features',data.X),('target',data.y)]:
            if key in raw and not np.array_equal(raw[key],value): raise ValueError('Saved raw data differs from CSV')
            raw[key]=value
        chosen=path
    return raw,chosen


def row_pixels(sheet,row):
    dimension=sheet.row_dimensions.get(row)
    if dimension is not None and dimension.hidden: return 0.
    points=dimension.height if dimension is not None and dimension.height is not None else sheet.sheet_format.defaultRowHeight or 15.
    return max(0.,float(points))*96/72


def next_row(sheet,start,height):
    current=start;pixels=0.
    while pixels<height+24:
        if current-start>10000: raise ValueError('Too many hidden or zero-height rows for image layout')
        pixels+=row_pixels(sheet,current);current+=1
    return current


def placement(sheet):
    if sheet.max_row<=40 or (sheet.max_column>12 and sheet.max_row<=200):
        return 1,sheet.max_row+3,'below'
    return sheet.max_column+3,2,'right'


def render_excel_report(source,output,max_models=3,csv=None):
    max_models=integer(max_models,'max_models',1,3)
    source=Path(source);out=Path(output)
    m=validate(source)
    if m['schema_version'] not in range(2,8): raise ValueError('Unsupported source schema')
    with np.load(source/'arrays.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    raw,chosen=resolve_data(source,m,arrays,csv)
    sheets=pd.read_excel(source/'report.xlsx',sheet_name=None)
    plans,skipped=build_plans(m,sheets,arrays,max_models,raw)
    if not plans: raise ValueError('No eligible figures; provide the original hash-matching CSV if true labels are missing')
    out.mkdir(parents=True,exist_ok=False)
    from . import __version__
    record={'schema_version':8,'status':'running','version':__version__,
            'source_schema':m['schema_version'],'source_version':m['version'],
            'source_directory':str(source.resolve()),'max_models':max_models,'skipped':skipped,
            'source_hashes':source_hashes(),'source_artifacts':{},'figures':[],
            'python':platform.python_version(),'created_utc':datetime.now(timezone.utc).isoformat(),
            'dependencies':{name:importlib.metadata.version(name) for name in ['matplotlib','pillow','openpyxl','numpy','scikit-learn']}}
    started=time.perf_counter()
    write_json(out/'render_manifest.json',record)
    try:
        for name in ['manifest.json','report.xlsx','arrays.npz']:
            shutil.copy2(source/name,out/name)
            record['source_artifacts'][name]=digest(out/name)
        if chosen is not None:
            shutil.copy2(chosen,out/'source_input.csv')
            record['input_csv_sha256']=digest(out/'source_input.csv')
        folder=out/'figures';folder.mkdir()
        workbook=load_workbook(out/'report.xlsx')
        if any(sheet._images for sheet in workbook.worksheets):
            raise ValueError('The source numerical workbook must not already contain images')
        layouts={sheet.title:placement(sheet) for sheet in workbook.worksheets}
        row_by_sheet={name:layout[1] for name,layout in layouts.items()}
        column_by_sheet={name:layout[0] for name,layout in layouts.items()}
        plot_data={}
        for plan in plans:
            path=folder/(plan['id']+'.png')
            font=draw_figure(plan,path)
            image=Image(path)
            width=min(800,int(image.width));height=round(image.height*width/image.width)
            image.width=width;image.height=height
            sheet=workbook[plan['sheet']];row=row_by_sheet[sheet.title];col=column_by_sheet[sheet.title]
            anchor=f'{get_column_letter(col)}{row}'
            sheet.add_image(image,anchor)
            row_by_sheet[sheet.title]=next_row(sheet,row,height)
            description={key:value for key,value in plan.items() if key!='data'}
            description.update(file=str(path.relative_to(out)),sha256=digest(path),anchor=anchor,
                               width=width,height=height,font=font,layout=layouts[sheet.title][2],
                               data_keys={key:plan['id']+'__'+key for key in plan['data']})
            plot_data.update({description['data_keys'][key]:value for key,value in plan['data'].items()})
            record['figures'].append(description)
        workbook.save(out/'report_with_figures.xlsx')
        workbook.close()
        np.savez_compressed(out/'figure_data.npz',**plot_data)
        record.update(status='completed',seconds=time.perf_counter()-started,
                      figure_count=len(plans),artifacts={name:digest(out/name) for name in ['report_with_figures.xlsx','figure_data.npz']},
                      workbook_bytes=(out/'report_with_figures.xlsx').stat().st_size)
        write_json(out/'render_manifest.json',record)
        return out
    except Exception as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(out/'render_manifest.json',record)
        raise
