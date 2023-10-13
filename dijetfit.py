import h5py, math, commands, random
from array import array
import numpy as np
import time, sys, os, optparse, json
import pathlib2
import csv
import pandas as pd
from Utils import *

import ROOT
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetOptTitle(0)
import CMS_lumi, tdrstyle
tdrstyle.setTDRStyle()
ROOT.gROOT.SetBatch(True)
ROOT.RooRandom.randomGenerator().SetSeed(random.randint(0, 1e+6))
import pdb
from Fitter import Fitter
from DataCardMaker import DataCardMaker
from Utils import *
import string_constants as strc
 
#python dijetfit.py -i /afs/cern.ch/work/i/izoi/public/forJennifer/run_41025/ --sig signal_XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrowReco.h5 --qcd bkg_XToYYprimeTo4Q_MX2000_MY80_MYprime170_narrowReco_0.25.h5 --xsec 0.25 -M 2000.0 --out test --config 4

def get_generated_events(filename):

   with open('files_count.json') as f:
      data = json.load(f)
 
   N = 0
   found = False
   for k in data.keys():
      if k in filename or k.replace('_EXT','') in filename: 
         N += data[k][0]
         found = True
   
   if not found: 
      print("######### no matching key in files_count.json for file "+filename+", EXIT !!!!!!!!!!!!!!")
      sys.exit()

   print(" in get_generated_events N = ",N) 
   return N 

def makeData(options, dataFile, q, iq, quantiles, hdata, last_q, minMJJ=0, maxMJJ=1e+04):
 
   file = h5py.File(options.inputDir+"/"+dataFile,'r')
   sel_key_q = 'sel_q30' if q == 'q0' else 'sel_' + q # selection column for quantile q (use rejected events of q70 for q100)
   print(iq,len(quantiles),q)
   if iq<len(quantiles)-1: 
      print("Current quantile file: %s, reading quantile between %s and %s" % (file, q, quantiles[iq+1]))
   
   else: print("Current quantile file: %s, reading quantile between %s and %s" % (file, q, quantiles[iq]))

   data = file['eventFeatures'][()] 
   mjj_idx = np.where(file['eventFeatureNames'][()] == 'mJJ')[0][0]
   data = data[ (minMJJ <= data[:,mjj_idx]) & (data[:,mjj_idx] <= maxMJJ)]

   # if quantile = total, fill histogram with all data and return
   if q=='total':
    for e in range(data.shape[0]): hdata.Fill(data[e][mjj_idx])
    return

   # else, if quantile = real quantile, fill with orthogonal data
   sel_idx = np.where(file['eventFeatureNames'][()] == sel_key_q)[0][0] # 0=rejected 1=accepted
 
   if q==last_q:
    for e in range(data.shape[0]):
     if data[e][sel_idx]==1: hdata.Fill(data[e][mjj_idx])
   elif q=='q0': #if 30% quantile is rejected then events are in the 0-30% less anomalous slice
    for e in range(data.shape[0]):
     if data[e][sel_idx]==0: hdata.Fill(data[e][mjj_idx]) 
   else: 
     print(".... checking orthogonality wrt",quantiles[iq+1],"quantile....")
     sel_key_iq = 'sel_' + quantiles[iq+1] # selection column for quantile q -- ex: if q=q30 it must reject q50 and accept q30
     sel_idx_iq = np.where(file['eventFeatureNames'][()] == sel_key_iq)[0][0] # 0=rejected 1=accepted
     for e in range(data.shape[0]):
      if data[e][sel_idx_iq]==0 and data[e][sel_idx]==1: hdata.Fill(data[e][mjj_idx])

def prepare_output_directory(out_dir, clean_up=True):
    if not os.path.exists(out_dir):
        pathlib2.Path(out_dir).mkdir(parents=True, exist_ok=True)
        return
    if clean_up:
        os.system('rm '+out_dir+'/{*.root,*.txt,*.C,*.png,*.csv}') 

def rebin(options,binsx,bins_fine,quantiles):

      #decide binning based on the template quantile
      print("PREPARE HISTO_INIT TO GET COSTUME BINNING!!")
      if options.config==1 or options.config==8: 
         histo_init = ROOT.TH1F("mjj_qcd_%s"%quantiles[-2],"mjj_qcd_%s"%quantiles[-2],bins_fine,binsx[0],binsx[-1])
         makeData(options,options.qcdFile,quantiles[-2],4,quantiles,histo_init,quantiles[-2],binsx[0],binsx[-1])
      elif options.config==4:
         histo_init = ROOT.TH1F("mjj_qcd_%s"%quantiles[0],"mjj_qcd_%s"%quantiles[0],bins_fine,binsx[0],binsx[-1])
         makeData(options,options.qcdFile,quantiles[0],0,quantiles,histo_init,quantiles[-2],binsx[0],binsx[-1])
      else: sys.exit()
      

      fine_bin_size = 4
      if(options.mjj_max < 0. and options.rebin): 
         options.mjj_max = get_mjj_max(options.inputDir+"/"+options.qcdFile) + 5.0
         print("MJJ MAX %.2f" % options.mjj_max)
         options.mjj_max = max(1.2*options.mass, options.mjj_max)
         print("MJJ MAX %.2f" % options.mjj_max)
      
      if(options.mjj_min > 0 and options.mjj_min < binsx[-1]):
         start_idx = 0
         while(binsx[start_idx] < options.mjj_min):
               start_idx +=1
               print(start_idx,options.mjj_min)
         binsx = binsx[start_idx:]

         print("(options.mjj_min - binsx[0])",(options.mjj_min - binsx[0]))
         if(abs(options.mjj_min - binsx[0]) < 50.):
               binsx[0] = options.mjj_min
         else:
               binsx.insert(0, options.mjj_min)
         print("Will start fit from %.0f GeV" % binsx[0])

      if(options.mjj_max > 0 and options.mjj_max < binsx[-1]):
         print("rebinning with max mjj %.2f" % options.mjj_max)
         end_idx = len(binsx)-1
         while(binsx[end_idx]   > options.mjj_max and end_idx > 0): 
               end_idx -=1
         binsx = binsx[:end_idx]

         if(abs(options.mjj_max - binsx[-1]) < 50.):
               binsx[-1] = options.mjj_max
         else:
               binsx.append(options.mjj_max)
         print("Will end fit at %.0f GeV" % binsx[-1])
         print(binsx)
      print("BEFORE roundTo",binsx)
      roundTo(binsx, fine_bin_size)

      if(options.rebin):
         bins_nonzero = get_rebinning(binsx, histo_init)
         print("Rebinning to avoid zero bins!")
         print("old", binsx)
         print("new", bins_nonzero)
         bins = bins_nonzero
      else:
         bins = binsx
      roobins = ROOT.RooBinning(len(bins)-1, array('d', bins), "mjjbins")
      bins_fine = int(binsx[-1] - binsx[0])/fine_bin_size
      return bins_fine, roobins, binsx


if __name__ == "__main__":

   #some configuration
   parser = optparse.OptionParser()
   parser.add_option("--xsec","--xsec",dest="xsec",type=float,default=0.0,help="Injected signal cross section in pb")
   parser.add_option("-M","-M",dest="mass",type=float,default=3500.,help="Injected signal mass")
   parser.add_option("--lnn",dest="lnn",default=True,action="store_true",help="Read lnn uncertainty from file")
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
   parser.add_option("--bestfit", default=False, dest="best_fitrange",action="store_true", help="""Proceed only if best fit range is found.""")
   parser.add_option("--cleanup", default=False, action="store_true", help="""Use if you want to cleanup directory before next run""")
   (options,args) = parser.parse_args()
   
   seed=-1 
   mass = options.mass
   signal_name=os.path.split(options.sigFile)[-1]
   signal_name=signal_name.replace('Reco.h5','').replace('signal_','')

   mass_label=(int)(mass)
   sig_res = options.sig_res
   out_dir = options.out_dir
   prepare_output_directory(out_dir, options.cleanup) # by default, cleanup = False
   binsx = [1460, 1530, 1607, 1687, 1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438,
             2546, 2659, 2775, 2895, 3019, 3147, 3279, 3416, 3558, 3704, 3854,
             4010, 4171, 4337, 4509, 4700, 4900,  5100, 5300, 5500, 5800,
             6100, 6400, 6800]
   bins_fine = int(binsx[-1]-binsx[0])          
   
   
   df=pd.read_csv(os.path.join(options.inputDir,'efficiencies.csv'),names=['sig','deff','peff','meff','eff'])
   peff=df[df['sig']==signal_name].peff.values[0]
   meff=df[df['sig']==signal_name].meff.values[0]
   deff=df[df['sig']==signal_name].deff.values[0]
   eff =df[df['sig']==signal_name].eff.values[0] # Product of presel eff, mass cut eff and deta cut eff.
   
   #original binning
   #binsx = [1455,1530,1607,1687,1770,1856,1945,2037,2132,2231,2332,2438,2546,2659,2775,2895,3019,3147,3279,3416,3558,3704,3854,4010,4171,4337,4509,4686,4869,5058,5253,5500,5663,5877,6100,6400,6800]
   #roobins = ROOT.RooBinning(len(binsx)-1, array('d',binsx), "mjjbins") 
   most_an='q90'
   
   if options.config==1:
        quantiles = ['q0', 'q30', 'q50', 'q70', 'q90', 'total'] #most anomalous is q90
        fractions = [1, 0.20/0.30, 0.20/0.30, 0.20/0.30, 0.10/0.30] 
        most_an='q90'
   elif options.config==2:    
        quantiles = ['q0', 'q30', 'q50', 'q70', 'q90', 'q95','total'] #most anomalous is q95
        fractions = [1, 0.20/0.30, 0.20/0.30, 0.20/0.30, 0.05/0.30, 0.05/0.30]
   elif options.config==3:    
        quantiles = ['q0', 'q30', 'q50', 'q70', 'q90', 'q95', 'q99', 'total'] #most anomalous is q99
        fractions = [1, 0.20/0.30, 0.20/0.30, 0.20/0.30, 0.05/0.30, 0.04/0.30, 0.01/0.30]
   elif options.config==4:    
        quantiles = ['q90', 'q95', 'q99', 'total']
        fractions = [1, 0.04/0.05, 0.01/0.05]
        most_an='q99'
   elif options.config==5:    
        quantiles = ['q95', 'q99', 'total']
        fractions = [1, 0.01/0.04]
   elif options.config==6:    
        quantiles = ['q70', 'q90', 'q95', 'q99', 'total']
        fractions = [1, 0.05/0.20, 0.04/0.20, 0.01/0.20]
   elif options.config==7:    
        quantiles = ['q50', 'q70', 'q90', 'q95', 'q99', 'total']
        fractions = [1, 1, 0.05/0.20, 0.04/0.20, 0.01/0.20]
   elif options.config==8:
        quantiles = ['q0', 'q30', 'q50', 'q70', 'total'] #most anomalous is q70
        fractions = [1, 0.20/0.30, 0.20/0.30, 0.30/0.30] 
        most_an='q70'
   
   # change signal fit intervall according to resonance width
   if sig_res == "na":
      sig_mjj_limits = (0.8*mass,1.2*mass)
   else:
      sig_mjj_limits = (0.4*mass,1.4*mass)
   
   print("Using entire mass range for signal fit")
   sig_mjj_limits = (options.mjj_min,options.mjj_max)
   
   bins_sig_fit = array('f',truncate([binsx[0]+ib for ib in range(bins_fine+1)],*sig_mjj_limits))
   large_bins_sig_fit = array('f',truncate(binsx,*sig_mjj_limits))
   roobins_sig_fit = ROOT.RooBinning(len(large_bins_sig_fit)-1, array('d',large_bins_sig_fit), "mjjbins_sig")
   print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  ")
   lumi = 137.2 #inv fb
   sig_incl = 150000.
   sig_xsec = 1000.*options.xsec # transform from pb to fb
   sig_scale = sig_xsec*lumi/sig_incl #rescale xsec from pb to fb   
   print(" sig_xsec ",sig_xsec,"lumi",lumi,"sig_incl",sig_incl,"sig_scale",sig_scale)
   print(" &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&  ")

   ################################### FIRST PREPARE DATA ###################################
   '''
    MAKE HISTOGRAMS:
    for each quantile + 'bottom rejected' 10% + all events:
      fill mjj histogram for 
      - signal: histos_sig
      - background: histos_qcd
      and save in data_mjj_X_qY.root
   '''       
   histos_sig = []
   histos_qcd = []
   tag_eff={}



   if not options.load_data:
      
      bins_fine, roobins, binsx = rebin(options, binsx, bins_fine, quantiles)

      #Background data preparation
      for iq,q in enumerate(quantiles):
      
         histos_qcd.append( ROOT.TH1F("mjj_qcd_%s"%q,"mjj_qcd_%s"%q,bins_fine,binsx[0],binsx[-1]) )
         print
         makeData(options,options.qcdFile,q,iq,quantiles,histos_qcd[-1],quantiles[-2],binsx[0],binsx[-1]) #first fill orthogonal data histos
         print("************ Found",histos_qcd[-1].GetEntries(),"/",histos_qcd[-1].Integral(),"background events for quantile",q)
         print

      #Signal data preparation 
      for iq,q in enumerate(quantiles):
         
         histos_sig.append( ROOT.TH1F("mjj_sig_%s"%q,"mjj_sig_%s"%q,len(bins_sig_fit)-1,bins_sig_fit ) )
         print
         makeData(options,options.sigFile,q,iq,quantiles,histos_sig[-1],quantiles[-2],binsx[0],binsx[-1]) #first fill orthogonal data histos 
         print("************ Found",histos_sig[-1].GetEntries(),"/",histos_sig[-1].Integral(),"signal events for quantile",q)
         print
         
      for h in histos_sig: h.SaveAs(os.path.join(out_dir, "data_"+h.GetName()+".root"))
      for h in histos_qcd: h.SaveAs(os.path.join(out_dir, "data_"+h.GetName()+".root"))
      #sys.exit()
      
   else: #let's make it faster if you have run once already!

      bins_fine, roobins, binsx = rebin(options, binsx, bins_fine, quantiles)

      print('=== loading histogram data from file ===')
      #Load signal data
      for q in quantiles:
      
         fname = os.path.join(out_dir, "data_mjj_sig_%s.root"%q) #JEN
         q_datafile = ROOT.TFile.Open(fname,'READ')
         histos_sig.append(q_datafile.Get("mjj_sig_%s"%q))
         histos_sig[-1].SetDirectory(ROOT.gROOT)
         q_datafile.Close()

      #Load background data
      for q in quantiles:
         
         fname = os.path.join(out_dir, "data_mjj_qcd_%s.root"%q)
         q_datafile = ROOT.TFile.Open(fname,'READ')
         histos_qcd.append(q_datafile.Get("mjj_qcd_%s"%q))
         histos_qcd[-1].SetDirectory(ROOT.gROOT)
         q_datafile.Close()
        
      for q,h in enumerate(histos_sig): print("************ Found",h.GetEntries(),"/",h.Integral(),"signal events for quantile",quantiles[q])
      for q,h in enumerate(histos_qcd): print("************ Found",h.GetEntries(),"/",h.Integral(),"background events for quantile",quantiles[q])

   sum_n_histos_qcd = sum([h.GetEntries() for h in histos_qcd[:-1]])
   sum_n_histos_sig = sum([h.GetEntries() for h in histos_sig[:-1]])

   for iq,q in enumerate(quantiles):
      tag_eff[q]=histos_sig[iq].GetEntries()/histos_sig[-1].GetEntries()
   
   print("************************************************************************************** ")
   print("TOTAL SIGNAL EVENTS",histos_sig[-1].GetEntries(),"/",histos_sig[-1].Integral(), " (sum histos = )", sum_n_histos_sig)
   print("TOTAL BACKGROUND EVENTS",histos_qcd[-1].GetEntries(),"/",histos_qcd[-1].Integral(), " (sum histos = )", sum_n_histos_qcd)
   print
   print("************************************************************************************** ")
      

   ################################### NOW MAKE THE FITS ###################################
   '''
    for each quantile:
      - fit signal shape -> gauss mu & std ??
      - fit background shape -> exponential lambda ??
      - chi2 ??
   ''' 

   nParsToTry = [2, 3, 4]
   best_i = [0]*len(quantiles)
   best_probs=[0.]*len(quantiles)
   nPars_QCD = [0]*len(quantiles)
   qcd_fname = [""]*len(quantiles)
   chi2s = [0]*len(quantiles)
   ndofs = [0]*len(quantiles)
   probs = [0]*len(quantiles)
   fit_params = [0]*len(quantiles)
   fit_errs = [0]*len(quantiles)
   dcb = True #use Double Crystal Ball for signal templates
   fit_parameters={}
   lnn={}
   lnn_up={}
   lnn_down={}
   uncertainties_UD = ['JES','JER','JMS','JMR','pdf','prefire','pileup','btag','PS_ISR','PS_FSR','F','R','RF','top_ptrw','lund_sys','lund_bquark','lund_stat','lund_pt']
   uncertainties_norm = ['lund_bad_matching_unc']
   uncertainty_dict={}

   for iq,q in enumerate(quantiles):
      
      #if q!=most_an and q!='total': continue
      #if q!='total': continue # Check for inclusive only limits
      # try:
      #    if options.config==4:
      #       df=pd.read_csv(os.path.join(options.inputDir,'uncertainties_updated_with_lund_%s_4category.csv'%q))
      #    else:
      #       df=pd.read_csv(os.path.join(options.inputDir,'uncertainties_updated_with_lund_%s.csv'%q))
         
      #    signame=os.path.split(options.sigFile)[-1].replace('signal_','').replace('Reco.h5','')
      #    lnn[q]=df[df['signal_name']==signame].total.values[0]
      #    print(lnn[q])
      # except:
      #    lnn[q]=20.
      #    print('fetching uncertainties failed. Defaulting to 0.2 for systematics and 0. for lund ')
      
      try:
         if options.config==4:
            df=pd.read_csv(os.path.join(options.inputDir,'csv','uncertainties_UP+DOWN_%s_4category.csv'%q))
         else:
            print("Set config to 4")
            sys.exit(0)
         uc_dict={}
         for uc in uncertainties_UD:
            uc_dict[uc+'_up']=1+0.01*df[df['signal_name']==signame][uc+'_up'].values[0]
            uc_dict[uc+'_down']=1+0.01*df[df['signal_name']==signame][uc+'_up'].values[0]
            
         uc_dict['lund_bad_matching_unc']=1+0.01*df[df['signal_name']==signame]['lund_bad_matching_unc'].values[0]
         lnn[q]=uc_dict
      except:
         lnn[q]=20.
         print('fetching uncertainties failed. Defaulting to 0.2 for systematics and 0. for lund ')
         signame=os.path.split(options.sigFile)[-1].replace('signal_','').replace('Reco.h5','')
      
      print("########## FIT SIGNAL AND SAVE PARAMETERS for quantile "+q+"    ############")
      sig_outfile = ROOT.TFile(os.path.join(out_dir, "sig_fit_%s.root"%q),"RECREATE")
      
      ### create signal model: 
      # option dcb == False: gaussian centered at mass-center with sigma in {2%,10%of mass center} + crystal ball for asymmetric tail (pure functional form)
      # option dcb == True: Double Crystal Ball as in B2G-20-009 (some parameters are fixed in the Fitter.py as tested on XYY... does it work for all signals?)

      # setup fit parameters according to resonance type
      if sig_res == "na": # fit narrow signal
        print('===> fitting narrow signal model')
        alpha = (0.6, 0.45, 1.05) # this applies only to dcb=False
      else: # else fit broad signal
        print('===> fitting broad signal model')
        alpha = None #use default alpha 
      
      fitter=Fitter(['mjj_fine'])
      if dcb: fitter.signalResonanceDCB('model_s',"mjj_fine", mass=mass)
      else: fitter.signalResonance('model_s',"mjj_fine", mass=mass, alpha=alpha)
       
      ### fit the signal model to actual sig histogram data

      fitter.w.var("MH").setVal(mass)
      fitter.importBinnedData(histos_sig[iq],['mjj_fine'],'data')
      fres = fitter.fit('model_s','data',[ROOT.RooFit.Save(1)],out_dir)
      fres.Print()
      print("**********************************************************************************************")
      print("signal fit result for quantile "+q)
      fres.status()
      print(" estimated distance to minimum ",fres.edm())
      if fres.edm() < 1.00e-03:
         print(" FIT CONVERGED ")
      else:
         print(" FIT DIDN'T CONVERGE ")
         #break
      print("**********************************************************************************************")

      ### compute chi-square of compatibility of signal-histogram and signal model for sanity ch05eck
      # plot fit result to signal_fit_q.png for each quantile q

      mjj_fine = fitter.getVar('mjj_fine')
      mjj_fine.setBins(len(bins_sig_fit))
      chi2_fine = fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s.png"%q), out_dir)
      fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_log.png"%q), out_dir, 0, True)
      chi2 = fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_binned.png"%q), out_dir, roobins_sig_fit)
      fitter.projection("model_s","data","mjj_fine", os.path.join(out_dir, "signal_fit_%s_log_binned.png"%q), out_dir, roobins_sig_fit, True)

      # write signal histogram and model with params

      sig_outfile.cd()
      histos_sig[iq].Write()

      if dcb: graphs={'mean':ROOT.TGraphErrors(),'sigma':ROOT.TGraphErrors(),'alpha1':ROOT.TGraphErrors(),'n1':ROOT.TGraphErrors(),'alpha2':ROOT.TGraphErrors(),'n2':ROOT.TGraphErrors()}
      else: graphs={'mean':ROOT.TGraphErrors(),'sigma':ROOT.TGraphErrors(),'alpha':ROOT.TGraphErrors(),'sign':ROOT.TGraphErrors(),'scalesigma':ROOT.TGraphErrors(),'sigfrac':ROOT.TGraphErrors()}
      for var,graph in graphs.iteritems():
         value,error=fitter.fetch(var)
         graph.SetPoint(0,mass,value)
         graph.SetPointError(0,0.0,error)

      sig_outfile.cd()
      for name,graph in graphs.iteritems(): graph.Write(name)
      
      sig_outfile.Close() 

      print("#############################")
      print("for quantile ",q)
      print("signal fit chi2 (fine binning)",chi2_fine)
      print("signal fit chi2 (large binning)",chi2)
      print("#############################")

      print
      print
      print("############# FIT BACKGROUND AND SAVE PARAMETERS for quantile "+q+"   ###########")

      sb1_edge = 2232
      sb2_edge = 2776

      regions = [("SB1", binsx[0], sb1_edge),
                 ("SB2", sb2_edge, binsx[-1]),
                 ("SR", sb1_edge, sb2_edge),
                 ("FULL", binsx[0], binsx[-1])]

      blind_range = ROOT.RooFit.Range("SB1,SB2")
      full_range = ROOT.RooFit.Range("FULL")
      fit_ranges = [(binsx[0], sb1_edge), (sb2_edge, binsx[-1])]

      histos_sb_blind = []
      h = apply_blinding(histos_qcd[iq], ranges=fit_ranges)
      histos_sb_blind.append(h )
      num_blind = histos_sb_blind[-1].GetEntries()

      if options.blinded:
         fitting_histogram = histos_sb_blind[-1]
         data_name = "data_qcd_blind"
         fit_range = blind_range
         chi2_range = fit_ranges
         norm = ROOT.RooFit.Normalization(num_blind, ROOT.RooAbsReal.NumEvent)
      else:
         fitting_histogram = histos_qcd[iq]
         data_name = "data_qcd"
         fit_range = full_range
         chi2_range = None
         norm = ROOT.RooFit.Normalization(histos_qcd[iq].GetEntries(),
                                        ROOT.RooAbsReal.NumEvent)

      print("############# INJECT SIGNAL DATA GENERATING FROM SIGNAL PDF for quantile "+q+" ###########")
      #QCD is taken from the histogram
     
      try:
         f = ROOT.TFile("/tmp/aritra/cache%i.root"%(random.randint(0, 1e+6)),"RECREATE")
      except:
         print('tmpdir for user not found, creating cache file in current dir')
         f = ROOT.TFile("cache%i.root"%(random.randint(0, 1e+6)),"RECREATE")  
      f.cd()
      w=ROOT.RooWorkspace("w","w")

      # this is needed to get the right mjj observable (leave it)
      fitter_QCD=Fitter(['mjj_fine'])
      fitter_QCD.qcdShape('model_b','mjj_fine',2)
      fitter_QCD.importBinnedData(histos_qcd[iq],['mjj_fine'],'data_qcd')
      mjj = fitter_QCD.getVar('mjj_fine')
      mjj.setBins(bins_fine)
      ###

      model_s = fitter.getFunc('model_s')      
      model_s.Print("v")

      print

      # signal xsec set to 0 by default, so hdatasig hist not filled !
      if sig_xsec != 0 and not options.run_toys:     
         num_sig_evts = int(histos_sig[iq].GetEntries()*sig_scale)
         print("Generate", num_sig_evts, "signal events!" )
         datasig = model_s.generateBinned(ROOT.RooArgSet(mjj),num_sig_evts)
         hdatasig = datasig.createHistogram("mjj_fine")
      else: # just set same bins as qcd hist, do not fill!
         print(" xsec is zero!")
         hdatasig = ROOT.TH1F("mjj_generate_sig_%s"%q,"mjj_generate_sig_%s"%q,histos_qcd[iq].GetNbinsX(),histos_qcd[iq].GetXaxis().GetXmin(),histos_qcd[iq].GetXaxis().GetXmax())
      print(" ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"  )
      hdatasig.SetName("mjj_generate_sig_%s"%q)
      
      # signal+background fit (total histo) => since signal xsec = 0 per default, this is only background data & fit!

      sb_outfile = ROOT.TFile(os.path.join(out_dir, 'sb_fit_%s.root'%q),'RECREATE')
      sb_outfile.cd()
      htot = ROOT.TH1F()
      htot = histos_qcd[iq].Clone("mjj_generate_tot_%s"%q) #clone from qcd histo
      htot.Add(hdatasig)
      hdatasig.Write("mjj_generate_sig_%s"%q)
      histos_qcd[iq].Write("mjj_generate_qcd_%s"%q)
      htot.Write("mjj_generate_tot_%s"%q)

      w.Delete()
      f.Close()
      f.Delete()
      sb_outfile.Close()
      fitter.delete()
      
      print
      print

      qcd_fnames = [""]*len(nParsToTry)
      nParsToTry_converged = []
      chi2s[iq] = [0]*len(nParsToTry)
      ndofs[iq] = [0]*len(nParsToTry)
      probs[iq] = [0]*len(nParsToTry)
      fit_params[iq] = [0]*len(nParsToTry)
      fit_errs[iq] = [0]*len(nParsToTry)

      for i, nPars in enumerate(nParsToTry):
         print("Trying %i parameter background fit" % nPars)

         ### create background model: 2-parameter (p1 & p2) exponential (generic functional form, not based on data)
         fitter_QCD=Fitter(['mjj_fine'])
         fitter_QCD.qcdShape('model_b','mjj_fine',nPars)

         ### fit background model to actual qcd histogram data (all cuts applied)
         sb_outfile = ROOT.TFile(os.path.join(out_dir, 'sb_fit_%s.root'%q),'READ')
         my_histo = sb_outfile.Get("mjj_generate_tot_%s"%q)

         fitter_QCD.importBinnedData(my_histo,['mjj_fine'],'data_qcd')
         fres = fitter_QCD.fit('model_b','data_qcd',[ROOT.RooFit.Save(1), ROOT.RooFit.Verbose(0),  ROOT.RooFit.Minos(1), ROOT.RooFit.Minimizer("Minuit2")],out_dir)
         fres.Print()
         fres = fitter_QCD.fit('model_b','data_qcd',[ROOT.RooFit.Save(1), ROOT.RooFit.Verbose(0),  ROOT.RooFit.Minos(1), ROOT.RooFit.Minimizer("Minuit2")],out_dir)
         fres.Print()

         print("**********************************************************************************************")
         print("qcd fit result for quantile "+q+" and function parameters "+str(nPars))
         fres.status()
         print(" estimated distance to minimum ",fres.edm())
         #if fres.edm() < 1.00e-03:
         #   print " FIT CONVERGED "
         #   nParsToTry_converged.append(nPars)
         #else:
         #   print " FIT DIDN'T CONVERGE "
         #   break
         #print "**********************************************************************************************"

         qcd_fnames[i] = str(nPars) + 'par_qcd_fit%i_quantile%s.root' % (i,q)
         qcd_outfile = ROOT.TFile(os.path.join(out_dir, qcd_fnames[i]),'RECREATE')

         ### compute chi-square of compatibility of qcd-histogram and background model for sanity check
         # plot fit result to qcd_fit_q_binned.png for each quantile q

         chi2_fine = fitter_QCD.projection("model_b","data_qcd","mjj_fine", os.path.join(out_dir, qcd_fnames[i].replace(".root",".png")), out_dir, 0, True) # chi2 => sanity check
         chi2_binned = fitter_QCD.projection("model_b","data_qcd","mjj_fine", os.path.join(out_dir, qcd_fnames[i].replace(".root","_binned.png")), out_dir, roobins, True)

         ### write background histogram

         qcd_outfile.cd()
         histos_qcd[iq].Write() # ??? => write histo into qcd_outfile

         ### plot background data with fit

         mjj = fitter_QCD.getVar('mjj_fine')
         mjj.setBins(bins_fine)
         model = fitter_QCD.getFunc('model_b')
         dataset = fitter_QCD.getData('data_qcd')

         ################ Oz
         #rescale so pdfs are in evts per 100 GeV
         low = roobins.lowBound()
         high = roobins.highBound()
         n = roobins.numBoundaries() - 1

         #RootFit default normalization is full range divided by number of bins
         default_norm = (high - low)/ n
         rescale = 100./ default_norm

         fit_norm = ROOT.RooFit.Normalization(rescale,ROOT.RooAbsReal.Relative)

         linear_errors = False

         frame = mjj.frame()
         dataset.plotOn(frame, ROOT.RooFit.Name("data_qcd"), ROOT.RooFit.Invisible(), ROOT.RooFit.Binning(roobins), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2), ROOT.RooFit.Rescale(rescale))
         model.plotOn(frame, ROOT.RooFit.VisualizeError(fres, 1, linear_errors), ROOT.RooFit.FillColor(ROOT.kRed - 7), ROOT.RooFit.LineColor(ROOT.kRed - 7), ROOT.RooFit.Name(fres.GetName()), fit_norm)
         model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kRed + 1), ROOT.RooFit.Name("model_b"),  fit_norm)

         useBinAverage = True
         hpull = frame.pullHist("data_qcd", "model_b", useBinAverage)
         hresid = frame.residHist("data_qcd", "model_b", False, useBinAverage)
         dhist = ROOT.RooHist(frame.findObject(data_name, ROOT.RooHist.Class()))

         #get fractional error on fit
         central = frame.getCurve("model_b");
         curve =  frame.getCurve("fitresults");
         upBound = ROOT.TGraph(central.GetN());
         loBound = ROOT.TGraph(central.GetN());
         norm = get_roohist_sum(dhist)

         for j in range(curve.GetN()):
            if( j < central.GetN() ): upBound.SetPoint(j, curve.GetX()[j], curve.GetY()[j]);
            else: loBound.SetPoint( 2*central.GetN() - j, curve.GetX()[j], curve.GetY()[j]);


         fit_hist = model.createHistogram("h_model_fit", mjj, ROOT.RooFit.Binning(roobins))
         fit_hist.Scale(norm / fit_hist.Integral())

         #Get hist of pulls:  (data - fit) / tot_unc
         hresid_norm = get_pull_hist(model, frame, central, curve, hresid, fit_hist,  binsx)


         #abs because somtimes order is reversed
         err_on_sig = abs(upBound.Eval(options.mass) - loBound.Eval(options.mass))/2.
         frac_err_on_sig = err_on_sig / central.Eval(options.mass)
         bkg_fit_frac_err = frac_err_on_sig

        #redraw data (so on top of model curves)
         if(options.rebin):
            dataset.plotOn(frame, ROOT.RooFit.Name(data_name),   ROOT.RooFit.Binning(roobins), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
                       ROOT.RooFit.Rescale(rescale))
         else:
            dataset.plotOn(frame, ROOT.RooFit.Name(data_name),  ROOT.RooFit.XErrorSize(0), ROOT.RooFit.Binning(roobins), ROOT.RooFit.DataError(ROOT.RooAbsData.SumW2),
                       ROOT.RooFit.Rescale(rescale))

         framePulls = mjj.frame()
         framePulls.addPlotable(hpull, "X0 P E1")
         my_chi2, my_ndof = calculateChi2(hpull, nPars, excludeZeros = True, dataHist = dhist)
         my_prob = ROOT.TMath.Prob(my_chi2, my_ndof)

         # plot full qcd fit results with pull factors to mjj_qcd_q.pdf for each quantile q

         PlotFitResults(frame,fres.GetName(),nPars,framePulls,"data_qcd",
                        "model_b",my_chi2, my_ndof,
                        histos_qcd[iq].GetName()+"{}".format(
                           "_pars%i_blinded"%nPars if options.blinded else "_pars%i"%nPars), out_dir)

         # write qcd model with params

         graphs = {}
         for p in range(nPars): graphs['p%i'%(p+1)] = ROOT.TGraphErrors()
         for var,graph in graphs.iteritems():
            print(var)
            value,error=fitter_QCD.fetch(var)
            graph.SetPoint(0,mass,value)
            graph.SetPointError(0,0.0,error)


         #largest_frac_err = 0.
         bkg_fit_params = dict()
         for var, graph in graphs.iteritems():
            print(var)
            value, error = fitter_QCD.fetch(var)
            bkg_fit_params[var] = (value, error)
            graph.SetPoint(0, mass, value)
            graph.SetPointError(0, 0.0, error)
            #frac_err = abs(error/value)
            #largest_frac_err = max(frac_err, largest_frac_err)
         bkg_fit_params['cov'] = convert_matrix(fres.covarianceMatrix())
         print(bkg_fit_params['cov'])

         qcd_outfile.cd()       
         for name,graph in graphs.iteritems(): graph.Write(name) # ??? => saving params of bg fit -> load later for datacard

         qcd_outfile.Close()

         print("#############################")
         print(" for quantile ",q)
         print("bkg fit chi2 (fine binning)",chi2_fine)
         print("bkg fit chi2 (large binning)",chi2_binned)
         print("bkg fit chi2",chi2)
         print("#############################")

         print("bkg fit chi2/nbins (fine binning) ", chi2_fine)
         print("My chi2, ndof, prob", my_chi2, my_ndof, my_prob)
         print("My chi/ndof, chi2/nbins", my_chi2/my_ndof, my_chi2/(my_ndof + nPars)) #the different between custom and RooFit chi2 is that RooFit uses ndof = nbins instead of ndof=nbins-npars
         print("#############################")
         
         chi2s[iq][i] = my_chi2
         ndofs[iq][i] = my_ndof
         probs[iq][i] = my_prob
         fit_params[iq][i] = bkg_fit_params
         fit_errs[iq][i] = bkg_fit_frac_err
         fitter_QCD.delete()
      
      print(nParsToTry)
      print(nParsToTry_converged)
      print(ndofs)
      print(ndofs[iq])
      print(chi2s)
      print(chi2s[iq])
      print(best_i[iq])
      best_i[iq] = f_test(nParsToTry, ndofs[iq], chi2s[iq], fit_errs[iq], thresh = options.ftest_thresh, err_thresh = options.err_thresh) #f_test(nParsToTry_converged, ndofs[iq], chi2s[iq])
      nPars_QCD[iq] = nParsToTry[best_i[iq]] #nParsToTry_converged[best_i[iq]]
      qcd_fname[iq] = qcd_fnames[best_i[iq]]
      best_probs[iq] = probs[iq][best_i[iq]]
      print(" qcd_fname[iq] ",qcd_fname[iq])
      print("\n Chose %i parameters based on F-test ! \n" % nPars_QCD[iq])

      print
      print 


      print
      print
      
      fit_parameters['bkgfit_prob']=best_probs[iq]
      fit_parameters['nPars']=nPars_QCD[iq]
      fit_parameters['quantile']=q

      print("############ MAKE PER CATEGORY (quantile ",q," ) DATACARD AND WORKSPACE AND RUN COMBINE #############")

      card=DataCardMaker(q, out_dir)

      if dcb: card.addSignalShapeDCB('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0}, {'CMS_res_j':1.0})
      #else: card.addSignalShape('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0}, {'CMS_res_j':1.0})
      #if dcb: card.addSignalShapeDCB('model_signal_mjj','mjj', './XToYY_CASE_shapes/case_interpolation_M%s.0.root'%(str(mass_label)), {'CMS_scale_j':1.0}, {'CMS_res_j':1.0})
      
      ### !!! compute amount of signal to be injected 
      if q=='total': 
         tag_eff[q]=1.
         #eff=peff*deff*meff # Remove meff if mass range is not reduced
      # In case we are not restricting the range

      if sig_xsec==0: constant =lumi*eff*tag_eff[q] #1680. #   might give too large yields for combine to converge ???
      else: constant = sig_scale  #scaling factor to signal histo integral used by addFixedYieldFromFile --> should return r=1 in FitDiagnostics
      print(" constant = scaling factor to signal integral generated with 1 pb = 1000 fb xsec = ",constant)
      # add signal pdf from model_s, taking integral number of events with constant scaling factor for sig
      card.addFixedYieldFromFile('model_signal_mjj',0, os.path.join(out_dir, 'sig_fit_%s.root'%q), histos_sig[iq].GetName(), norm=constant)
      card.addSystematic("CMS_scale_j","param",[0.0,0.01])
      card.addSystematic("CMS_res_j","param",[0.0,0.035])
      if q!='total':
         for uc in uncertainties_UD:    
            card.addSystematic(uc,"lnN",{"model_signal_mjj":'%0.04f/%0.04f'%(lnn[q][uc+'_up'],lnn[q][uc+'_down'])})
      else:
         card.addSystematic("norm_unc","lnN",{"model_signal_mjj":1.2})
      # add bg pdf
      card.addQCDShapeNoTag('model_qcd_mjj','mjj', os.path.join(out_dir, qcd_fname[iq]), nPars_QCD[iq])
      card.addFloatingYield('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), histos_qcd[iq].GetName(),0,1e+8) #in reality we won't have this value from MC but we can put whatever since it's a flatParam
      for i in range(1,nPars_QCD[iq]+1): card.addSystematic("CMS_JJ_p%i"%i,"flatParam",[])
      card.addSystematic("shapeBkg_model_qcd_mjj_JJ_%s__norm"%q,"flatParam",[]) # integral -> anzahl events -> fuer skalierung der genormten roofit histogramm

      card.importBinnedData(os.path.join(out_dir, 'sb_fit_%s.root'%q), 'mjj_generate_tot_%s'%q,["mjj"],'data_obs',1.0)
      card.makeCard()
      card.delete()

      # run combine on datacard -> create workspaces workspace_JJ_0.0_quantile.root
      # -M Significance: profile likelihood
      if options.run_combine == True:
            cmd = 'cd {out_dir} && ' \
            'text2workspace.py datacard_JJ_{label}.txt -o workspace_JJ_{xsec}_{label}.root && ' \
            'combine -M FitDiagnostics workspace_JJ_{xsec}_{label}.root -m {mass} -n _M{mass}_xsec{xsec}_{label} && ' \
            'combine -M AsymptoticLimits workspace_JJ_{xsec}_{label}.root -m {mass} -n _limits_{mass}_xsec{xsec}_{label} && ' \
            'combine -M Significance workspace_JJ_{xsec}_{label}.root -m {mass} -n significance_{xsec}_{label} && ' \
            'combine -M Significance workspace_JJ_{xsec}_{label}.root -m {mass} --pvalue -n pvalue_{xsec}_{label}'.format(out_dir=out_dir, mass=mass, xsec=sig_xsec, label=q, SEED=seed)
      
      else:
            cmd = 'cd {out_dir} && ' \
            'text2workspace.py datacard_JJ_{label}.txt -o workspace_JJ_{xsec}_{label}.root'.format(out_dir=out_dir, mass=mass, xsec=sig_xsec, label=q)
      print(cmd)
      os.system(cmd)
      fitter_QCD.delete()
      #run and visualize s+b fit as sanity check (sb_fit_mjj_qcd_q.root.pdf)
      CHI2,NDOF=checkSBFit('{out_dir}/workspace_JJ_{xsec}_{label}.root'.format(out_dir=out_dir, xsec=sig_xsec,label=q),q,roobins,histos_qcd[iq].GetName()+"_M{mass}_xsec{xsec}.root".format(mass=mass,xsec=sig_xsec), nPars_QCD[iq], out_dir)
      fit_parameters['sbfit_prob']=ROOT.TMath.Prob(CHI2,NDOF)
      print(" %%%%%%%%%%%%%%%%%%%%%%%% done with quantile ",q)

      with open(os.path.join(out_dir,'fit_params_%s.json'%(q)), 'w') as f:
         json.dump(fit_parameters,f)
      
      if q==most_an:
         f_signif_name = ('{out_dir}/higgsCombinepvalue_{xsec}_{label}.'
                     + 'Significance.mH{mass:.0f}.root'
                     ).format(out_dir=out_dir,xsec=sig_xsec,label=q,mass=mass)
         f_signif = ROOT.TFile(f_signif_name, "READ")
         res1 = f_signif.Get("limit")
         res1.GetEntry(0)
         pvalue_sb = res1.limit   
         fields=[mass,nPars_QCD[iq],pvalue_sb,options.mjj_min,options.mjj_max]
         csv_filename=os.path.join(out_dir,'pvalues.csv')
         with open(csv_filename, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fields)
         f_limit_name = ('{out_dir}/higgsCombine_limits_{mass:.1f}_xsec{xsec}_{label}.AsymptoticLimits.mH{mass:.0f}.root').format(out_dir=out_dir,xsec=sig_xsec,label=q,mass=mass)
         f_limit = ROOT.TFile(f_limit_name, "READ")
         res1 = f_limit.Get("limit")
         res1.GetEntry(0)
         
         fields=[signal_name,mass,tag_eff[q]]
         new_fields=[signal_name,mass]
        
         
         fields+=[deff,peff,meff,eff,eff*tag_eff[q]]
         for ii in range((int)(res1.GetEntries())):
            res1.GetEntry(ii)
            #new_fields.append(res1.limit*1680./(137.2*eff*tag_eff[q]))
            new_fields.append(res1.limit)
         
         #new_fields.append(res1.limitErr*1680./(137.2*eff*tag_eff[q]))
         new_fields.append(res1.limitErr)
         
         ## COLUMNS: signal name, mass,tag_eff[q],d_eta_eff,presel_eff,mass_eff,eff(product of d*p*m effs.),total_eff, L2.5,L16,L50,L84,L97.5,L_obs,L_obs_error
         
         csv_filename=os.path.join(out_dir,'xsec_limits.csv')
         with open(csv_filename, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_fields)
         
   if q=='total':
         f_limit_name = ('{out_dir}/higgsCombine_limits_{mass:.1f}_xsec{xsec}_{label}.AsymptoticLimits.mH{mass:.0f}.root').format(out_dir=out_dir,xsec=sig_xsec,label=q,mass=mass)
         f_limit = ROOT.TFile(f_limit_name, "READ")
         res1 = f_limit.Get("limit")
         res1.GetEntry(0)
         
         fields=[signal_name,mass,tag_eff[q]]
         new_fields=[signal_name,mass]
        
         fields+=[deff,peff,meff,eff,eff*tag_eff[q]]
         #new_fields+=[deff,peff,meff,eff,eff*tag_eff[q]]
         for ii in range((int)(res1.GetEntries())):
            res1.GetEntry(ii)
            #fields.append(res1.limit*1680.) # Convert to upper limit on signal
            #new_fields.append(res1.limit*1680./(137.2*eff*tag_eff[q]))
            new_fields.append(res1.limit)
         
         #fields.append(1680.*res1.limitErr) # LimitErr for last entry which is the observed limit
         #new_fields.append(res1.limitErr*1680./(137.2*eff*tag_eff[q]))
         new_fields.append(res1.limitErr)
         
         ## COLUMNS: signal name, mass,tag_eff[q],d_eta_eff,presel_eff,mass_eff,eff(product of d*p*m effs.),total_eff, L2.5,L16,L50,L84,L97.5,L_obs,L_obs_error
         csv_filename=os.path.join(out_dir,'inclusive_event_limits.csv')
         with open(csv_filename, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(fields)
         csv_filename=os.path.join(out_dir,'inclusive_xsec_limits.csv')
         with open(csv_filename, 'a+') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(new_fields)
         
   if not options.best_fitrange:
      sys.exit(0)
   else:
      print("BEST FITTING RANGE FOUND. WILL CONTINUE")
   print("------------------------------------------------- F-TEST result -------------------------------------------------")
   for iq,q in enumerate(quantiles):
      print(" for quantile ",q," chosen ", nPars_QCD[iq]," parameter function")
   print("--------------------------------------------------------------------------------------------------")      

   print("############ MAKE N-CATEGORY DATACARD AND WORKSPACE AND RUN COMBINE #############")
   #The difference here is that the background shape comes from one specific quantile (rather than from its own as above)

   cmdCombine = 'cd {out_dir} && combineCards.py '.format(out_dir=out_dir)

    # Setting the function of quantiles to be equal to the background enriched region --> to be changed: implement partial correlation if N parameters different
   print(qcd_fname)
   print(nPars_QCD)
   if options.correlateB == True:
      for i,p in enumerate(nPars_QCD):
         if p != nPars_QCD[0]:
            print("Names before replacing",qcd_fname[i], qcd_fname[0].split("_")[-1],qcd_fname[i].split("_")[-1])
            if not 'total' in qcd_fname[i]: qcd_fname[i] =  qcd_fname[0].replace( qcd_fname[0].split("_")[-1], qcd_fname[i].split("_")[-1])
            print("Names after replacing",qcd_fname[i])

   #combined_tag_eff=0.
   for iq,q in enumerate(quantiles):  
   
      if q == 'total': continue
      #combined_tag_eff+=tag_eff[q]
      card=DataCardMaker(q+"_4combo", out_dir) # 4combo = for combination, intermediate datacards that will be combined in the final one. 

      if dcb: card.addSignalShapeDCB('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0},{'CMS_res_j':1.0})
      #else: card.addSignalShape('model_signal_mjj','mjj', os.path.join(out_dir, 'sig_fit_%s.root'%q), {'CMS_scale_j':1.0},{'CMS_res_j':1.0})
      #if dcb: card.addSignalShapeDCB('model_signal_mjj','mjj', './XToYY_CASE_shapes/case_interpolation_M%s.0.root'%(str(mass_label)), {'CMS_scale_j':1.0}, {'CMS_res_j':1.0})

      if sig_xsec==0: constant = lumi*eff*tag_eff[q] # 1680.0 # might give too large yields for combine to converge ???
      else: constant = sig_scale  #scaling factor to signal histo integral used by addFixedYieldFromFile --> should return r=1 in FitDiagnostics
      card.addFixedYieldFromFile('model_signal_mjj',0, os.path.join(out_dir, 'sig_fit_%s.root'%q), "mjj_sig_%s"%q, norm=constant) #JEN CHANGE BACK TO: histos_sig[iq].GetName()
      card.addSystematic("CMS_scale_j","param",[0.0,0.01])
      card.addSystematic("CMS_res_j","param",[0.0,0.035])    
      for uc in uncertainties_UD:    
         card.addSystematic(uc,"lnN",{"model_signal_mjj":'%0.04f/%0.04f'%(lnn[q][uc+'_up'],lnn[q][uc+'_down'])})
      
      if options.correlateB == True:
         #TAKE BACKGROUND SHAPE COMES FROM BACKGROUND-ENRICHED QUANTILE SLICE --> WHICH ONE? TRY THE Q0 SLICE!
         print("============================== IMPLEMENTING FULL CORRELATION AMONG CATEGORIES !!!")
         card.addQCDShapeNoTag('model_qcd_mjj','mjj', os.path.join(out_dir, qcd_fname[0]), nPars_QCD[0])
         if q=='q0': card.addFloatingYield('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), histos_qcd[iq].GetName())
         else: card.addFloatingYieldCorr('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), "mjj_qcd_%s"%q,fractions[iq], constant=histos_qcd[0].GetEntries(),mini=0,maxi=1e+8)
         for i in range(1,nPars_QCD[0]+1): card.addSystematic("CMS_JJ_p%i"%i,"flatParam",[])
         card.addSystematic("shapeBkg_model_qcd_mjj_JJ_q0__norm","flatParam",[])
      else:   
         card.addQCDShape('model_qcd_mjj','mjj', os.path.join(out_dir, qcd_fname[iq]), nPars_QCD[iq])
         card.addFloatingYield('model_qcd_mjj',1, os.path.join(out_dir, qcd_fname[iq]), histos_qcd[iq].GetName(),0,1e+8)
         for i in range(1,nPars_QCD[iq]+1): card.addSystematic("CMS_JJ_p%i_JJ_"%i+q+"_4combo","flatParam",[])
         card.addSystematic("shapeBkg_model_qcd_mjj_JJ_%s__norm"%q,"flatParam",[])

      card.importBinnedData(os.path.join(out_dir, 'sb_fit_%s.root'%q), 'mjj_generate_tot_%s'%q,["mjj"],'data_obs',1.0)
      card.makeCard()
      card.delete()
      
      cmdCombine += "JJ_{quantile}=datacard_JJ_{label}.txt ".format(quantile=q,label=q+"_4combo",xsec=sig_xsec)
   
   #MAKE FINAL DATACARD (needs some cosmetics as below) 
   cmdCombine += '&> datacard_{xsec}_final.txt'.format(xsec=sig_xsec)
   print(cmdCombine)
   os.system(cmdCombine)
  
   d = open(os.path.join(out_dir, 'datacard_tmp.txt'),'w')
   dorig = open('{out_dir}/datacard_{xsec}_final.txt'.format(out_dir=out_dir, xsec=sig_xsec),'r')
   for l in dorig.readlines(): d.write(l)
   d.close()
   dorig.close()
   if options.run_combine == True:
          cmd = 'cd {out_dir} && ' \
          'mv datacard_tmp.txt datacard_{xsec}_final.txt && ' \
          'text2workspace.py datacard_{xsec}_final.txt -o workspace_JJ_{xsec}_final.root && ' \
          'combine -M FitDiagnostics workspace_JJ_{xsec}_final.root -m {mass} -n _M{mass}_xsec{xsec} && ' \
          'combine -M AsymptoticLimits workspace_JJ_{xsec}_final.root -m {mass} -n _limits_{mass}_xsec{xsec}_final && ' \
          'combine -M Significance workspace_JJ_{xsec}_final.root -m {mass} -n significance_{xsec} && '\
          'combine -M Significance workspace_JJ_{xsec}_final.root -m {mass} --pvalue -n pvalue_{xsec}'.format(out_dir=out_dir, mass=mass,xsec=sig_xsec, SEED=seed)
   else:
          cmd = 'cd {out_dir} && ' \
          'mv datacard_tmp.txt datacard_{xsec}_final.txt && ' \
          'text2workspace.py datacard_{xsec}_final.txt -o workspace_JJ_{xsec}_final.root'.format(out_dir=out_dir, mass=mass,xsec=sig_xsec)
   print(cmd)
   os.system(cmd)
   
   f_limit_name = ('{out_dir}/higgsCombine_limits_{mass:.1f}_xsec{xsec}_final.AsymptoticLimits.mH{mass:.0f}.root').format(out_dir=out_dir,xsec=sig_xsec,mass=mass)
   f_limit = ROOT.TFile(f_limit_name, "READ")
   res1 = f_limit.Get("limit")
   res1.GetEntry(0)
   
   new_fields=[signal_name,mass]
   
   for ii in range((int)(res1.GetEntries())):
      res1.GetEntry(ii)
      #new_fields.append(res1.limit*1680./(137.2*eff*combined_tag_eff))
      new_fields.append(res1.limit)
   
   #new_fields.append(res1.limitErr*1680./(137.2*eff**combined_tag_eff))
   new_fields.append(res1.limitErr)
   #new_fields.append(combined_tag_eff)   
   
   ## COLUMNS: signal name, mass,tag_eff[q],d_eta_eff,presel_eff,mass_eff,eff(product of d*p*m effs.),total_eff, L2.5,L16,L50,L84,L97.5,L_obs,L_obs_error
   csv_filename=os.path.join(out_dir,'final_xsec_limits.csv')
   with open(csv_filename, 'a+') as csv_file:
      writer = csv.writer(csv_file)
      writer.writerow(new_fields)
   filetypes=["root","png","C"]
   for ft in filetypes:
      cmd = 'find ./*.'+ft+' -mtime 0 -exec mv {} '+out_dir+' \;'
      os.system(cmd)
   print(" DONE! ")

   for iq,q in enumerate(quantiles): 
      if q == 'total': continue
      checkSBFitFinal(out_dir+'/workspace_JJ_{xsec}_{label}.root'.format(xsec=sig_xsec,label='final'),q,roobins,'M{mass}_xsec{xsec}_{q}_final.root'.format(mass=mass,xsec=sig_xsec,q=q),nPars_QCD[0],out_dir)
