
import csv,sys,collections
t={}
for r in csv.DictReader(open(sys.argv[1])): t[int(r['launch'])]=float(r['gpu_us'])
agg=collections.defaultdict(lambda:[0.0,0]); tot=0; n=0; miss=0
for r in csv.DictReader(open(sys.argv[2])):
    if r['phase']!='prefill': continue
    L=int(r['launch'])
    if L not in t: miss+=1; continue
    a=agg[r['kernel']]; a[0]+=t[L]; a[1]+=1; tot+=t[L]; n+=1
print('prefill launches timed %d, missing %d, total %.1f ms'%(n,miss,tot/1e3))
for k,(us,c) in sorted(agg.items(),key=lambda x:-x[1][0])[:25]:
    print('%-48s %9.1f ms %5.1f%% %6d launches'%(k,us/1e3,100*us/tot,c))
