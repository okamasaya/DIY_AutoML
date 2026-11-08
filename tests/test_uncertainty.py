from itertools import product
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from openpyxl import load_workbook
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import Pipeline

from diy_automl.metric_uncertainty import (evaluate_metric_uncertainty, interval_rows,
                                           resample_indices, score_samples)
from diy_automl.selection import bootstrap
from diy_automl.supervised import METRICS, metrics
from diy_automl.uncertainty_validation import validate_uncertainty
from diy_automl import uncertainty_workflow as workflow
from diy_automl.validation import validate

ROOT=Path(__file__).parents[1]
SOURCES={2:ROOT/'validation/current/wine-run',3:ROOT/'validation/current/wine-select',
         5:ROOT/'validation/dl-repeat-0.4.0a2/wine',7:ROOT/'validation/sl-tuning-0.4.0a4/wine'}
CSV=ROOT/'data/reference/wine.csv'


@pytest.mark.parametrize('resampling',['stratified','ordinary'])
def test_metric_and_class_replicates_against_sklearn(resampling):
    y=np.array([0,0,0,1,1,2,2]);pred=np.array([0,1,2,1,0,2,0]);labels=np.arange(4)
    overall,classes,data=evaluate_metric_uncertainty(y,pred,labels,repeats=37,seed=11,resampling=resampling)
    for b,idx in enumerate(data['indices']):
        np.testing.assert_allclose(data['samples'][b],list(metrics(y[idx],pred[idx],labels).values()),rtol=0,atol=1e-15)
        p,r,f,_=precision_recall_fscore_support(y[idx],pred[idx],labels=labels,zero_division=0)
        np.testing.assert_allclose(data['class_samples'][b],np.stack([p,r,f],axis=-1),rtol=0,atol=1e-15)
    np.testing.assert_allclose(overall.Estimate,list(metrics(y,pred,labels).values()),rtol=0,atol=1e-15)
    np.testing.assert_allclose(overall['Bootstrap SE'],data['samples'].std(axis=0,ddof=1))
    bounds=np.quantile(data['samples'],[.025,.975],axis=0,method='linear')
    np.testing.assert_allclose(overall[['Lower','Upper']].to_numpy(),bounds.T)
    absent=classes[classes['Encoded Class']==3]
    assert absent.Status.eq('class_absent_in_test').all()
    assert absent['Test Support'].eq(0).all()


def test_exact_ordinary_accuracy_sampling_variance():
    # Enumerate all 4**4 equally probable ordinary bootstrap draws, independently of RNG.
    y=np.array([0,0,1,1]);pred=np.array([0,1,0,1])
    draws=np.array(list(product(range(4),repeat=4)))
    _,_,samples,_,_=score_samples(y,pred,np.arange(2),draws)
    assert np.isclose(samples[:,0].mean(),.5)
    assert np.isclose(samples[:,0].var(ddof=0),.5*(1-.5)/4)


def test_bootstrap_se_is_not_monte_carlo_error_of_mean():
    samples=np.array([[.1],[.2],[.5],[.8]])
    result=interval_rows(np.array([.4]),samples,['score'],.8)
    assert np.isclose(result['Bootstrap SE'].iloc[0],np.std(samples,ddof=1))
    assert not np.isclose(result['Bootstrap SE'].iloc[0],np.std(samples,ddof=1)/np.sqrt(4))


def test_stratified_compatibility_and_counts():
    y=np.array([0]*7+[1]*3);pred=np.array([0,0,0,1,1,0,0,1,0,1])
    overall,_,data=evaluate_metric_uncertainty(y,pred,np.arange(2),repeats=101)
    old_samples,old_ci=bootstrap(y,pred,repeats=101,classes=np.arange(2))
    np.testing.assert_allclose(data['samples'],old_samples.to_numpy(),rtol=0,atol=1e-15)
    np.testing.assert_allclose(overall[['Estimate','Lower','Upper']],old_ci[['Estimate','Lower','Upper']],rtol=0,atol=1e-15)
    assert np.all(np.sum(y[data['indices']]==1,axis=1)==3)
    ordinary=resample_indices(y,101,resampling='ordinary')
    assert len(np.unique(np.sum(y[ordinary]==1,axis=1)))>1
    assert np.array_equal(data['indices'],resample_indices(y,101))


def test_perfect_predictions_zero_width_is_flagged():
    frame,classes,_=evaluate_metric_uncertainty(np.array([0,0,1,1]),np.array([0,0,1,1]),np.arange(2),repeats=20)
    assert frame['Zero Width'].all() and frame['Bootstrap SE'].eq(0).all()
    assert classes['Zero Width'].all()
    assert 'Does not imply' in workflow.SCOPE['Zero width']


@pytest.mark.parametrize('kwargs',[
    {'repeats':1},{'repeats':True},{'seed':-1},{'confidence':1},{'confidence':float('nan')},
    {'confidence':True},{'resampling':'cluster'},{'repeats':10001},
])
def test_invalid_bootstrap_settings(kwargs):
    with pytest.raises(ValueError):
        evaluate_metric_uncertainty(np.array([0,1]),np.array([0,1]),np.arange(2),**kwargs)


@pytest.mark.parametrize('y,pred,classes',[
    ([0.,1.],[0,1],[0,1]),([0,1],[0],[0,1]),([0,2],[0,1],[0,1]),
    ([0,1],[0,1],[1,0]),([0,1],[0,1],[0]),([],[],[0,1]),
])
def test_invalid_observations(y,pred,classes):
    with pytest.raises(ValueError): evaluate_metric_uncertainty(y,pred,classes,repeats=20)


@pytest.fixture(scope='module')
def completed(tmp_path_factory):
    return workflow.evaluate_uncertainty(SOURCES[3],tmp_path_factory.mktemp('uncertainty')/'out',bootstrap_repeats=31)


@pytest.mark.parametrize('schema',[2,3,5,7])
def test_all_supported_sources_and_no_classifier_refit(tmp_path,monkeypatch,schema):
    def no_fit(*args,**kwargs): raise AssertionError('Unexpected classifier fit')
    monkeypatch.setattr(Pipeline,'fit',no_fit)
    before={str(p.relative_to(SOURCES[schema])):p.read_bytes() for p in SOURCES[schema].rglob('*') if p.is_file()}
    out=workflow.evaluate_uncertainty(SOURCES[schema],tmp_path/'result',bootstrap_repeats=17,csv=CSV if schema==2 else None)
    m=validate(out,9)
    count={2:6,3:1,5:10,7:1}[schema]
    assert len(m['evaluations'])==count and m['sheets']['Metric_Uncertainty']==count*4
    assert m['sheets']['Class_Uncertainty']==count*3*3 and m['figure_count']==1
    assert all((SOURCES[schema]/name).read_bytes()==value for name,value in before.items())
    assert {str(p.relative_to(SOURCES[schema])) for p in SOURCES[schema].rglob('*') if p.is_file()}==set(before)


def test_ordinary_mode_workflow_and_moved_result(tmp_path):
    out=workflow.evaluate_uncertainty(SOURCES[3],tmp_path/'ordinary',bootstrap_repeats=19,resampling='ordinary')
    moved=tmp_path/'moved';shutil.copytree(out,moved);shutil.rmtree(out)
    m=validate_uncertainty(moved)
    assert m['settings']['resampling']=='ordinary'
    scope=pd.read_excel(moved/'report.xlsx',sheet_name='Scope')
    assert scope.Value.str.contains('class counts vary',regex=False).any()


def test_no_overwrite_or_nested_source_output(tmp_path,completed):
    with pytest.raises(FileExistsError): workflow.evaluate_uncertainty(SOURCES[3],completed,bootstrap_repeats=20)
    with pytest.raises(ValueError,match='separate'):
        workflow.evaluate_uncertainty(SOURCES[3],SOURCES[3]/'nested',bootstrap_repeats=20)


@pytest.mark.parametrize('case',['bad_settings','ul_only','failed_source','wrong_csv','limit'])
def test_input_failure_before_output_creation(tmp_path,monkeypatch,case):
    source=SOURCES[3];kwargs={}
    if case=='bad_settings': kwargs['bootstrap_repeats']=1
    elif case=='ul_only': source=ROOT/'validation/current/wine-repeat-ul'
    elif case=='failed_source':
        source=tmp_path/'failed';shutil.copytree(SOURCES[3],source)
        p=source/'manifest.json';m=json.loads(p.read_text());m['status']='failed';p.write_text(json.dumps(m))
    elif case=='wrong_csv':
        source=SOURCES[2];kwargs['csv']=ROOT/'data/reference/iris.csv'
    else:
        monkeypatch.setattr(workflow,'MAX_DRAWS',10)
    out=tmp_path/'output'
    with pytest.raises(ValueError): workflow.evaluate_uncertainty(source,out,**kwargs)
    assert not out.exists()


def test_plot_failure_retains_failed_status_and_original(tmp_path,monkeypatch):
    def bad_plot(*args): raise RuntimeError('plot failed')
    monkeypatch.setattr(workflow,'draw_intervals',bad_plot)
    before=(SOURCES[3]/'manifest.json').read_bytes()
    out=tmp_path/'failed'
    with pytest.raises(RuntimeError,match='plot failed'):
        workflow.evaluate_uncertainty(SOURCES[3],out,bootstrap_repeats=20)
    assert json.loads((out/'manifest.json').read_text())['status']=='failed'
    assert (out/'arrays.npz').exists() and (SOURCES[3]/'manifest.json').read_bytes()==before
    with pytest.raises(ValueError): validate_uncertainty(out)


@pytest.mark.parametrize('kind',['indices','samples','class_samples','overlap','table','model','scope','embedded_image'])
def test_corruption_rejected_even_after_updated_artifact_hash(tmp_path,completed,kind):
    out=tmp_path/kind;shutil.copytree(completed,out)
    manifest_path=out/'manifest.json';m=json.loads(manifest_path.read_text())
    if kind in ('indices','samples','class_samples','overlap'):
        with np.load(out/'arrays.npz') as saved: arrays={k:saved[k] for k in saved.files}
        if kind=='overlap': arrays['evaluation_0__train'][0]=arrays['evaluation_0__test'][0]
        else:
            array=arrays['evaluation_0__'+kind]
            array.flat[0]=((array.flat[0]+1)%len(arrays['evaluation_0__truth'])) if kind=='indices' else array.flat[0]+.1
        np.savez_compressed(out/'arrays.npz',**arrays)
    elif kind in ('table','embedded_image'):
        book=load_workbook(out/'report.xlsx')
        if kind=='table': book['Metric_Uncertainty']['G2']=.999
        else: book['Intervals']._images.clear()
        book.save(out/'report.xlsx');book.close()
    elif kind=='model': m['evaluations'][0]['model']='Wrong model'
    else: m['scope']['SE']='SD/sqrt(B)'
    m['artifacts']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in m['artifacts']}
    manifest_path.write_text(json.dumps(m))
    with pytest.raises(ValueError): validate_uncertainty(out)


def test_cli_and_tensorflow_free_import(tmp_path):
    # A finder that refuses TensorFlow proves saved DL results can be evaluated without it.
    code="""
import importlib.abc,sys
class NoTF(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,*args):
        if fullname.split('.')[0] in ('tensorflow','keras'): raise AssertionError('TensorFlow import attempted')
sys.meta_path.insert(0,NoTF())
from diy_automl.cli import main
main()
"""
    out=tmp_path/'cli'
    result=subprocess.run([sys.executable,'-c',code,'uncertainty','--run',str(SOURCES[5]),
                           '--output',str(out),'--bootstrap-repeats','11'],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert len(validate_uncertainty(out)['evaluations'])==10


def test_current_sl_only_run_without_original_csv(tmp_path,monkeypatch):
    from diy_automl import run_csv
    source=run_csv(CSV,'target',tmp_path/'source',methods=('sl',))
    def no_fit(*args,**kwargs): raise AssertionError('Unexpected classifier fit')
    monkeypatch.setattr(Pipeline,'fit',no_fit)
    # The current run saves truth, so the report remains usable without its CSV path.
    p=source/'manifest.json';m=json.loads(p.read_text());m['settings']['csv']='/not/available.csv';p.write_text(json.dumps(m))
    out=workflow.evaluate_uncertainty(source,tmp_path/'result',bootstrap_repeats=20)
    assert len(validate_uncertainty(out)['evaluations'])==5
