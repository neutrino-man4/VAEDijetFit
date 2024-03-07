import os,sys,time
import numpy as np
from array import array
import optparse, json
import subprocess
import ROOT
import CMS_lumi, tdrstyle
import copy
from Utils import *
import pandas as pd

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option("-x","--xsec",dest="xsec",type=float,default=0.01,help="Injected signal cross section in pb")
    parser.add_option("-M","-M",dest="mass",type=float,default=3500.,help="Injected signal mass")
    parser.add_option("--lnn",dest="lnn",default=True,action="store_true",help="If true, read lnn uncertainty from file")
    parser.add_option("-i","--inputDir",dest="inputDir",default='./',help="directory with all quantiles h5 files")
    parser.add_option("--qcd","--qcd",dest="qcdFile",default='qcd.h5',help="QCD h5 file")
    parser.add_option("--sig","--sig",dest="sigFile",default='signal.h5',help="Signal h5 file")
    parser.add_option("-l","--load_data",dest="load_data",action="store_true",help="Load orthogonal data")
    parser.add_option("-C","--correlate",dest="correlateB",action="store_true",help="Coorelate background shape among quantiles")
    parser.add_option("--res", "--res", dest="sig_res", type="choice", choices=("na", "br"), default="na", help="resonance type: narrow [na] or broad [br]")
    parser.add_option("--out", '--out', dest="out_dir", type=str, default="./", help='output directory to store all results (plots, datacards, root files, etc.)')
    parser.add_option("-b", "--blinded", dest="blinded", action="store_true", default=False, help="Blinding the signal region for the fit.")
    parser.add_option("-R", "--run_combine", dest="run_combine", action="store_true", default=False, help="Run combine FitDiagnostics and Significance (set to False for the scan)")
    parser.add_option("--config","--config", dest="config", type=int, default=4, help="quantiles config")
    parser.add_option("--run_toys", "--run_toys", dest="run_toys", action="store_true", default=False, help="Prepare datacards to run for signal injection toys")
    parser.add_option("--ftest_thresh", type=float, default=0.05, help="Threshold to prefer a function in the f-test")
    parser.add_option("--err_thresh", type=float, default=0.5, help="Threshold on fit unc to be included in f-test")
    parser.add_option("--mjj_min", type=float, default=-1.0, help="Minimum mjj for the fit")
    parser.add_option("--mjj_max", type=float, default=-1.0, help="Maximum mjj for the fit")
    parser.add_option("--rebin", default=False, action="store_true", help="""Rebin dijet bins to make sure no bins less than 5 evts""")
    parser.add_option("--save-all", default=False, action="store_true", help="""Choose whether to save all outputs or only the best fit outputs. \n If used, output will be stored in out_dir/mjj_min-mjj_max""")
    parser.add_option("--no-dynamic",default=False,action="store_true",help="""Use this option when best fit range is not required and original range is desired""")
    parser.add_option("--cleanup", default=False, action="store_true", help="""Use if you want to cleanup directory before next run""")
    parser.add_option("--bestfit", default=False, dest="best_fitrange",action="store_true", help="""Proceed only if best fit range is found.""")
    (options,args) = parser.parse_args()

    xsec=options.xsec
    mass=options.mass
    inputDir=options.inputDir
    outputDir=options.out_dir
    qcdFile=options.qcdFile
    sigFile=options.sigFile
    sig_res=options.sig_res
    config=options.config
    ftest_thresh=options.ftest_thresh
    err_thresh=options.err_thresh
    mjj_min=options.mjj_min
    mjj_max=options.mjj_max
    
    if not options.save_all:
        dijet_cmd = "python dijetfit.py -i %s --xsec %f -M %.0f --ftest_thresh %.2f --err_thresh %.2f --out %s --qcd %s --sig %s --config %d" % (inputDir, xsec, mass, ftest_thresh, err_thresh,outputDir,qcdFile,sigFile,config)
    else:
        dijet_cmd = "python dijetfit.py -i %s --xsec %f -M %.0f --ftest_thresh %.2f --err_thresh %.2f --qcd %s --sig %s --config %d" % (inputDir, xsec, mass, ftest_thresh, err_thresh,qcdFile,sigFile,config)
    
    if options.run_combine:
        dijet_cmd+= " -R"
    if options.load_data:
        dijet_cmd+= " -l"
    if options.correlateB:
        dijet_cmd+= " -C"
    if options.blinded:
        dijet_cmd+= " -b"
    if options.run_toys:
        dijet_cmd+= " --run_toys"
    if options.rebin:
        dijet_cmd+= " --rebin"
    if options.cleanup:
        dijet_cmd+= " --cleanup"
    if options.lnn:
        dijet_cmd+= " --lnn"
    if options.best_fitrange:
        dijet_cmd+= " --bestfit"
    
    run_fit = True
    last_change = 'start' # flag to store whether the last change was made at the start value of the fit or the end value. 
    iter=0

    if options.no_dynamic:
        dijet_cmd+=" --mjj_min %.0f" % mjj_min
        dijet_cmd+=" --mjj_max %.0f" % mjj_max
        print(dijet_cmd)
        subprocess.call(dijet_cmd,  shell = True, executable = '/bin/bash')
        print("Exiting. Best fit range will not be found.")
        run_fit=False
    while (run_fit):
        if (iter>2):
            print('Will probably not converge. Exiting')
            break
        run_fit=False 
        dijet_cmd_iter=copy.deepcopy(dijet_cmd)
        # if (iter>0):
        #     print "Running iteration %d of fit. Will load pre-existing QCD histograms."%(iter)
        #     dijet_cmd_iter+=" -l"
        
        if mjj_min > 0:
            dijet_cmd_iter+= " --mjj_min %.0f" % mjj_min
        if mjj_max > 0:
            dijet_cmd_iter+= " --mjj_max %.0f" % mjj_max
        
        if options.save_all: # If all outputs are to be stored, then create a separate directory. This is only for the purpose of comparing between different fit ranges and the best fit. 
            outputDir_iter=os.path.join(outputDir,'%d-%dGeV'%(mjj_min,mjj_max))
            
        else:
            outputDir_iter=copy.deepcopy(outputDir)
        
        dijet_cmd_iter+= " --out %s" % outputDir_iter
        # Print command to be run, and then run it. 
        print(dijet_cmd_iter)
        subprocess.call(dijet_cmd_iter,  shell = True, executable = '/bin/bash')
        
        iter+=1
        if options.config==1:
            params_file=os.path.join(outputDir_iter,'fit_params_q90.json')
        elif options.config==4:
            params_file=os.path.join(outputDir_iter,'fit_params_q99.json')
        elif options.config==8:
            params_file=os.path.join(outputDir_iter,'fit_params_q70.json')
        else:
            sys.exit(0)
            
        with open(params_file) as f:
            fit_params=json.load(f)
        if (fit_params['bkgfit_prob'] < 0.05):# and fit_params['sbfit_prob'] < 0.05):
            run_fit=True
            print("Fit will be run again in the range %.0f to %.0f GeV" % (mjj_min,mjj_max))
        else:
            print("Best binning found: %.0f to %.0f GeV. Exiting." % (mjj_min,mjj_max))
        
        # Change binning ranges if fit is too poor    
        if (run_fit):
            if (mjj_min < 1460.): mjj_min = 1460.
            elif (mjj_max < 0 or mjj_max > 6800.): mjj_max = 6800. # Start fit in the range [1460,6800]
            elif (last_change=='end' and (mjj_min+400.<=mass)):
                mjj_min+=200.
                last_change = 'start'
            elif (mjj_max-500.>=mass):
                if (mjj_max>5000): mjj_max -= 500.
                else: mjj_max -= 200.
                last_change='end'
            elif (mjj_min+250. <= mass): mjj_min+=100.
            else:
                print("Boundaries have not been changed")
                run_fit = False