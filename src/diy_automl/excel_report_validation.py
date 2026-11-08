"""Check copied evidence, figure inputs, unchanged cells and actual embedded images."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils.cell import coordinate_to_tuple
from openpyxl.utils.units import pixels_to_EMU

from .clustering_subsample import integer
from .excel_report import digest,placement,resolve_data,row_pixels
from .report_figures import build_plans
from .ul_subsample_validation import require
from .validation import validate


def validate_excel_report(directory):
    root=Path(directory)
    r=json.loads((root/'render_manifest.json').read_text())
    require(r['schema_version']==8 and r['status']=='completed','Rendering did not complete')
    integer(r['max_models'],'max_models',1,3)
    require(set(r['dependencies'])=={'matplotlib','pillow','openpyxl','numpy','scikit-learn'},'Renderer dependency record differs')
    m=validate(root)
    require(r['source_schema']==m['schema_version'] and r['source_version']==m['version'],'Source provenance differs')
    require(set(r['source_artifacts'])=={'manifest.json','report.xlsx','arrays.npz'},'Source artifact set differs')
    for name,expected in r['source_artifacts'].items(): require(digest(root/name)==expected,'Copied source checksum differs')
    require(set(r['artifacts'])=={'report_with_figures.xlsx','figure_data.npz'},'Rendered artifact set differs')
    for name,expected in r['artifacts'].items(): require(digest(root/name)==expected,'Rendered artifact checksum differs')
    require((root/'source_input.csv').exists()==('input_csv_sha256' in r),'CSV provenance differs')
    csv=root/'source_input.csv' if 'input_csv_sha256' in r else None
    if csv is not None: require(digest(csv)==r['input_csv_sha256']==m['input_sha256'],'Supplemental CSV checksum differs')
    with np.load(root/'arrays.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in stored.files}
    raw,_=resolve_data(root,m,arrays,csv,auto_csv=False)
    sheets=pd.read_excel(root/'report.xlsx',sheet_name=None)
    plans,skipped=build_plans(m,sheets,arrays,r['max_models'],raw)
    require(r['skipped']==skipped,'Omitted figures differ')
    require(len(plans)==r['figure_count']==len(r['figures']) and len(plans)>0,'Figure count differs')
    with np.load(root/'figure_data.npz',allow_pickle=False) as stored:
        actual_data={key:stored[key] for key in stored.files}
    expected_keys=set()
    by_sheet={}
    for plan,record in zip(plans,r['figures']):
        expected={key:value for key,value in plan.items() if key!='data'}
        for key,value in expected.items(): require(record[key]==value,'Figure identity or display policy differs')
        require(record['file']==f"figures/{plan['id']}.png",'Unexpected PNG path')
        require(digest(root/record['file'])==record['sha256'],'Figure PNG checksum differs')
        require(record['data_keys']=={key:plan['id']+'__'+key for key in plan['data']},'Figure data mapping differs')
        for key,value in plan['data'].items():
            name=record['data_keys'][key]
            require(name in actual_data and value.shape==actual_data[name].shape,'Figure input shape differs')
            require(np.allclose(value,actual_data[name],atol=1e-10,rtol=1e-10,equal_nan=True),'Figure inputs differ from saved evidence')
            expected_keys.add(name)
        by_sheet.setdefault(record['sheet'],[]).append(record)
    require(set(actual_data)==expected_keys,'Unexpected figure inputs')
    base=load_workbook(root/'report.xlsx');rendered=load_workbook(root/'report_with_figures.xlsx')
    try:
        require(base.sheetnames==rendered.sheetnames,'Numerical sheet names/order changed')
        for original,illustrated in zip(base.worksheets,rendered.worksheets):
            require(original.max_row==illustrated.max_row and original.max_column==illustrated.max_column,'Numerical sheet dimensions changed')
            require(list(original.values)==list(illustrated.values),'Numerical cells changed')
            records=by_sheet.get(original.title,[])
            require(len(illustrated._images)==len(records),'Embedded image count differs')
            original_col,original_row,layout=placement(original)
            bottom=None
            for record,image in zip(records,illustrated._images):
                row,col=coordinate_to_tuple(record['anchor'])
                require(col==original_col and row>=original_row and record['layout']==layout,'Figure overlaps the data table')
                expected_width=min(800,int(image.width))
                expected_height=round(image.height*expected_width/image.width)
                require(record['width']==expected_width and record['height']==expected_height,'Image aspect ratio or display size differs')
                anchor=image.anchor
                require(hasattr(anchor,'_from') and anchor._from.row==row-1 and anchor._from.col==col-1,'Embedded anchor differs')
                require(anchor._from.rowOff==0 and anchor._from.colOff==0,'Unexpected drawing offset')
                require(anchor.ext.cx==pixels_to_EMU(record['width']) and anchor.ext.cy==pixels_to_EMU(record['height']),'Embedded display size differs')
                top=sum(row_pixels(original,i) for i in range(1,row))
                require(bottom is None or top>=bottom+23.9,'Figures overlap vertically')
                bottom=top+record['height']
                import hashlib
                require(hashlib.sha256(image._data()).hexdigest()==record['sha256'],'Embedded PNG differs from its checked source')
    finally:
        base.close();rendered.close()
    return r
