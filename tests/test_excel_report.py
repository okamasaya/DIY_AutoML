from pathlib import Path
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans,AgglomerativeClustering
from sklearn.mixture import GaussianMixture

from diy_automl import excel_report as renderer
from diy_automl.excel_report_validation import validate_excel_report
from diy_automl.report_figures import build_plans

ROOT=Path(__file__).parents[1]
SOURCES={
    2:ROOT/'validation/current/wine-run',
    3:ROOT/'validation/current/wine-select',
    4:ROOT/'validation/current/wine-repeat-ul',
    5:ROOT/'validation/dl-repeat-0.4.0a2/wine',
    6:ROOT/'validation/ul-subsample-0.4.0a3/wine',
    7:ROOT/'validation/sl-tuning-0.4.0a4/wine',
}


@pytest.fixture(scope='module')
def illustrated(tmp_path_factory):
    output=tmp_path_factory.mktemp('excel')/'report'
    renderer.render_excel_report(SOURCES[2],output,csv=ROOT/'data/reference/wine.csv')
    return output


@pytest.mark.parametrize('schema',[2,3,4,5,6,7])
def test_all_source_schemas(tmp_path,schema,illustrated):
    out=illustrated if schema==2 else renderer.render_excel_report(SOURCES[schema],tmp_path/f'report-{schema}')
    m=validate_excel_report(out)
    assert m['source_schema']==schema
    assert m['figure_count']==(8 if schema==2 else 2)
    source_manifest=json.loads((out/'manifest.json').read_text())
    assert source_manifest['version']!=m['version']
    from diy_automl import __version__
    assert m['version']==__version__


def test_data_and_original_files_unchanged(illustrated):
    for name in ['manifest.json','report.xlsx','arrays.npz']:
        assert (illustrated/name).read_bytes()==(SOURCES[2]/name).read_bytes()
    m=validate_excel_report(illustrated)
    sl=[f for f in m['figures'] if f['id'].startswith('sl_cm_')]
    assert [f['source_row'] for f in sl]==[0,1,2]
    assert all('not test ranking' in f['scope'] for f in sl)
    selected=pd.read_excel(SOURCES[2]/'report.xlsx',sheet_name='UL_Result')
    expected=selected[np.isfinite(selected.Silhouette)].sort_values('Silhouette',ascending=False,kind='stable').iloc[:3].index.tolist()
    assert [f['source_row'] for f in m['figures'] if f['kind']=='cluster']==expected
    assert all(f['width']<=800 for f in m['figures'])
    assert {f['layout'] for f in m['figures']}=={'below','right'}


def test_report_does_not_refit_models(tmp_path,monkeypatch):
    def no_fit(*args,**kwargs): raise AssertionError('A classifier or clustering model was refit')
    for cls in [Pipeline,KMeans,AgglomerativeClustering,GaussianMixture]: monkeypatch.setattr(cls,'fit',no_fit)
    out=renderer.render_excel_report(SOURCES[2],tmp_path/'no-refit',max_models=1,csv=ROOT/'data/reference/wine.csv')
    m=validate_excel_report(out)
    assert m['figure_count']==4


def test_legacy_run_csv_hash_validation(tmp_path):
    wrong=tmp_path/'wrong.csv';wrong.write_text('x,target\n1,0\n2,1\n')
    with pytest.raises(ValueError,match='SHA256'):
        renderer.render_excel_report(SOURCES[2],tmp_path/'out',csv=wrong)
    assert not (tmp_path/'out').exists()


def test_source_failure_and_no_overwrite(tmp_path):
    bad=tmp_path/'bad';shutil.copytree(SOURCES[2],bad)
    p=bad/'manifest.json';m=json.loads(p.read_text());m['status']='failed';p.write_text(json.dumps(m))
    with pytest.raises(ValueError): renderer.render_excel_report(bad,tmp_path/'out')
    existing=tmp_path/'existing';existing.mkdir();(existing/'keep').write_text('unchanged')
    with pytest.raises(FileExistsError): renderer.render_excel_report(SOURCES[3],existing)
    assert (existing/'keep').read_text()=='unchanged'


@pytest.mark.parametrize('value',[0,4,True,1.2])
def test_invalid_display_limit(tmp_path,value):
    with pytest.raises(ValueError): renderer.render_excel_report(SOURCES[3],tmp_path/'invalid',max_models=value)
    assert not (tmp_path/'invalid').exists()


def test_render_failure_separate_from_source(tmp_path,monkeypatch):
    def fail(*args): raise RuntimeError('injected plotting failure')
    monkeypatch.setattr(renderer,'draw_figure',fail)
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError): renderer.render_excel_report(SOURCES[3],out)
    assert json.loads((out/'render_manifest.json').read_text())['status']=='failed'
    assert json.loads((out/'manifest.json').read_text())['status']=='completed'
    with pytest.raises(ValueError): validate_excel_report(out)


def test_projection_edge_cases_and_missing_truth():
    root=SOURCES[2];manifest=json.loads((root/'manifest.json').read_text())
    sheets=pd.read_excel(root/'report.xlsx',sheet_name=None)
    with np.load(root/'arrays.npz') as stored: original={key:stored[key] for key in stored.files}
    for width,expected in [(1,'single_feature'),(2,'constant')]:
        arrays=dict(original);arrays['ul_scaler_mean']=np.zeros(width);arrays['ul_scaler_scale']=np.ones(width)
        figures,_=build_plans(manifest,sheets,arrays,raw={'features':np.zeros((178,width))})
        clusters=[f for f in figures if f['kind']=='cluster']
        assert len(clusters)==3 and all(f['projection']==expected for f in clusters)
    arrays={key:value for key,value in original.items() if key!='ul_truth'}
    figures,skipped=build_plans(manifest,sheets,arrays)
    assert all(f['kind']!='confusion' for f in figures)
    assert len(skipped)==4
    assert all(f['projection']=='unavailable' for f in figures if f['kind']=='cluster')


def test_selected_models_and_all_seed_policy(tmp_path):
    selected=json.loads((SOURCES[7]/'manifest.json').read_text())
    out=renderer.render_excel_report(SOURCES[7],tmp_path/'selected')
    report=json.loads((out/'render_manifest.json').read_text())
    assert [f['id'] for f in report['figures']]==['sl_selected_cm','sl_training_cv']
    assert f"Candidate {selected['selected_candidate']}" in report['figures'][0]['title']
    base=SOURCES[5];manifest=json.loads((base/'manifest.json').read_text())
    sheets=pd.read_excel(base/'report.xlsx',sheet_name=None)
    with np.load(base/'arrays.npz') as stored: arrays={k:stored[k] for k in stored.files}
    plans,_=build_plans(manifest,sheets,arrays)
    assert plans[0]['labels']==[str(seed) for seed in manifest['settings']['training_seeds']]
    assert len([key for key in plans[0]['data'] if key.startswith('Epoch_')])==10


def checksum(out,record_key,name):
    p=out/'render_manifest.json';m=json.loads(p.read_text());m[record_key][name]=renderer.digest(out/name);p.write_text(json.dumps(m))


@pytest.mark.parametrize('kind',['numeric-cell','anchor','embedded-image','inputs','png','policy'])
def test_rendered_corruption_detected(illustrated,tmp_path,kind):
    out=tmp_path/'modified';shutil.copytree(illustrated,out)
    if kind in ['numeric-cell','anchor','embedded-image']:
        path=out/'report_with_figures.xlsx';wb=load_workbook(path)
        if kind=='numeric-cell': wb['SL_Result']['B2']=99
        elif kind=='anchor': wb['SL_Result']._images[0].anchor._from.row=0
        else: wb['SL_Result']._images.pop()
        wb.save(path);wb.close();checksum(out,'artifacts','report_with_figures.xlsx')
    elif kind=='inputs':
        path=out/'figure_data.npz'
        with np.load(path) as stored: values={key:stored[key] for key in stored.files}
        values['sl_cm_0__counts'][0,0]+=1
        np.savez_compressed(path,**values);checksum(out,'artifacts','figure_data.npz')
    elif kind=='png':
        with (out/'figures/sl_cm_0.png').open('ab') as stream: stream.write(b'corrupt')
    else:
        path=out/'render_manifest.json';m=json.loads(path.read_text());m['figures'][0]['scope']='Picked using test score';path.write_text(json.dumps(m))
    with pytest.raises(ValueError): validate_excel_report(out)


def test_standalone_workbook_contains_images(illustrated,tmp_path):
    target=tmp_path/'only.xlsx';shutil.copy2(illustrated/'report_with_figures.xlsx',target)
    wb=load_workbook(target)
    assert sum(len(sheet._images) for sheet in wb)==8
    wb.close()
