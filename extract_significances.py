import ROOT
import csv
quantiles = ['q0', 'q30', 'q50', 'q70', 'q90', 'total']
sig_xsec=0.0
for mass in range(1800,5900,100):
    fields=[mass]        
    out_dir='unblinding_complete/sig_mass{mass:.0f}'.format(mass=mass)
    for q in quantiles:
        f_signif_name = ('{out_dir}/higgsCombinepvalue_{xsec}_{label}.'
                            + 'Significance.mH{mass:.0f}.root'
                            ).format(out_dir=out_dir,xsec=sig_xsec,label=q,mass=mass)
        f_signif = ROOT.TFile(f_signif_name, "READ")
        res1 = f_signif.Get("limit")
        res1.GetEntry(0)
        pvalue = res1.limit
        f_signif.Close()
        fields.append(pvalue)   
    with open('csv_files/pvalues_unblinded_allQ.csv', 'a+') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(fields)
#import pdb; pdb.set_trace()