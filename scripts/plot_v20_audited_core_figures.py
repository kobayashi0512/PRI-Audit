"""Render six claim-frozen, journal-width main figures for the v14 paper."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from sklearn.metrics import roc_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/singlecell_v10"
V12 = ROOT / "results/paper_v12"
V14 = ROOT / "results/paper_v14"
OUT = ROOT / "figures/paper_v20/main"
OUT.mkdir(parents=True, exist_ok=True)

CHARCOAL = "#3B3B3B"
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
PURPLE = "#CC79A7"
GRAY = "#BDBDBD"
LIGHT = "#F2F2F2"
RED = "#B64342"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 6.4,
    "axes.titlesize": 7.2, "axes.labelsize": 6.5,
    "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
    "legend.fontsize": 5.6, "axes.spines.top": False,
    "axes.spines.right": False, "pdf.fonttype": 42, "svg.fonttype": "none",
})


def js(path):
    return json.loads(Path(path).read_text())


def panel(ax, label):
    ax.text(-.10, 1.06, label, transform=ax.transAxes, fontsize=8, fontweight="bold", va="top")


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=450, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def clean(ax):
    ax.grid(axis="y", color="#E5E5E5", lw=.45, zorder=0)
    ax.set_axisbelow(True)


strict_all = {row["dataset"]: row for row in js(RES / "strict_multiresolution_benchmarks.json")}
gains = pd.read_csv(V12 / "incremental_gain_bootstrap.csv")


# Figure 1 — framework and study design.
fig = plt.figure(figsize=(7.2, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.12], wspace=.42, hspace=.48)
ax = fig.add_subplot(gs[0, 0]); ax.axis("off"); panel(ax, "a")
ax.set_title("Independent-unit hierarchy", loc="left", pad=3)
boxes = [(0.08,.68,.84,.20,"Patient / donor",BLUE),(0.18,.39,.64,.18,"Sample / biopsy",CHARCOAL),(0.28,.10,.44,.18,"Cell / spot",GRAY)]
for x,y,w,h,t,c in boxes:
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.02",facecolor="white",edgecolor=c,lw=1.2,transform=ax.transAxes)); ax.text(x+w/2,y+h/2,t,ha="center",va="center",transform=ax.transAxes,fontweight="bold",color=c)
for y0,y1 in ((.68,.57),(.39,.28)):
    ax.add_patch(FancyArrowPatch((.5,y0),(.5,y1),transform=ax.transAxes,arrowstyle="-|>",mutation_scale=8,color=CHARCOAL,lw=.8))
ax.text(.5,.005,"Inference + resampling\nat patient level",transform=ax.transAxes,ha="center",va="bottom",color=ORANGE,fontsize=5.1)

ax = fig.add_subplot(gs[0, 1:]); panel(ax, "b"); ax.set_title("Eleven public datasets span four analytical roles", loc="left", pad=3)
ds = pd.read_csv(V14 / "main_table1_public_datasets.csv")
mods = ["bulk", "scRNA-seq", "spatial", "perturbation"]
role_colors = {"cross-cohort ceiling":CHARCOAL,"development/strict LOO":BLUE,"external validation":ORANGE,"spatial validation":GREEN,"frozen trigger prior":PURPLE}
for i, row in ds.reset_index(drop=True).iterrows():
    x = mods.index(row.modality); units = pd.to_numeric(pd.Series([row.independent_units_after_filtering]), errors="coerce").iloc[0]; size = 18 if pd.isna(units) else 18 + 1.6*np.sqrt(units)
    ax.scatter(x, i, s=size, color=role_colors[row.role], edgecolor="white", linewidth=.4)
ax.set_xticks(range(4), ["Bulk", "Single-cell", "Spatial", "Perturbation"]); ax.set_yticks(range(len(ds)), ds.dataset); ax.invert_yaxis(); ax.set_xlim(-.5,3.5); ax.grid(axis="x",color="#E5E5E5",lw=.5)

ax = fig.add_subplot(gs[1, 0]); ax.axis("off"); panel(ax, "c"); ax.set_title("Strict outer-patient fitting", loc="left", pad=3)
for j,(y,t,c) in enumerate([(0.78,"Hold out one patient",BLUE),(0.51,"Fit HVG / scaling /\nrepresentation",CHARCOAL),(0.24,"Predict held-out\nsamples",GREEN)]):
    ax.add_patch(FancyBboxPatch((.08,y-.08),.84,.16,boxstyle="round,pad=.015",facecolor="white",edgecolor=c,lw=1.1,transform=ax.transAxes)); ax.text(.5,y,t,ha="center",va="center",transform=ax.transAxes,color=c,fontweight="bold",fontsize=5.5)
    if j<2: ax.annotate("",xy=(.5,y-.17),xytext=(.5,y-.09),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",lw=.8,color=CHARCOAL))
ax.text(.5,.015,"Held-out patient never enters\nrepresentation fitting",ha="center",va="bottom",transform=ax.transAxes,color=ORANGE,fontsize=5.0)

ax = fig.add_subplot(gs[1, 1]); ax.axis("off"); panel(ax, "d"); ax.set_title("Matched incremental value", loc="left", pad=3)
ax.text(.5,.69,r"$\Delta_m = \mathrm{AUROC}_m - \mathrm{AUROC}_{matched\ expression}$",ha="center",va="center",transform=ax.transAxes,fontsize=8,color=BLUE)
ax.text(.5,.45,"same patients • same outer folds • same gene budget",ha="center",transform=ax.transAxes,color=CHARCOAL)
ax.axhline(.25,xmin=.12,xmax=.88,color=GRAY,lw=.7)
ax.text(.5,.12,"Decision threshold: clinically meaningful gain ≥ 0.05",ha="center",transform=ax.transAxes,color=ORANGE,fontsize=5.6)

ax = fig.add_subplot(gs[1, 2]); ax.axis("off"); panel(ax, "e"); ax.set_title("Resolution-specific falsification", loc="left", pad=3)
items=[("Labels",.17,BLUE),("Cells",.50,ORANGE),("Coordinates",.83,GREEN)]
for label,x,c in items:
    ax.add_patch(Circle((x,.55),.09,transform=ax.transAxes,facecolor="white",edgecolor=c,lw=1.2)); ax.text(x,.55,"↻",ha="center",va="center",transform=ax.transAxes,color=c,fontsize=11); ax.text(x,.32,label,ha="center",transform=ax.transAxes,fontweight="bold",color=c,fontsize=5.4)
ax.text(.5,.06,"Signal should change only when\nits information source is broken",ha="center",transform=ax.transAxes,fontsize=5.0,color=CHARCOAL)
save(fig,"fig1_pri_audit_framework")


# Figure 2 — expression ceiling and paired increments.
fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), gridspec_kw={"height_ratios":[1,1.15]})
bulk = pd.read_csv(ROOT / "results/trigger_v9/v9_tuned_pooled_results.csv")
pairs = pd.read_csv(V14 / "main_table1_public_datasets.csv").set_index("dataset").independent_units_after_filtering
y = np.arange(len(bulk)); axes[0,0].scatter(bulk.held_gene_cosine,y,s=24,color=CHARCOAL); axes[0,0].set_yticks(y,bulk.cohort); axes[0,0].invert_yaxis(); axes[0,0].set_xlabel("Held-cohort gene cosine"); axes[0,0].set_xlim(.4,.75); axes[0,0].set_title("Bulk disease-direction transfer",loc="left")
for i,row in bulk.iterrows(): axes[0,0].text(row.held_gene_cosine+.008,i,f"n={int(pairs[row.cohort])}",va="center",fontsize=5.2)
panel(axes[0,0],"a"); clean(axes[0,0])

names=["GSE151177","GSE173706","GSE206391"]; vals=[]; cis=[]
for name in names:
    m=strict_all[name]["methods"]["pseudobulk"]; vals.append(m["auroc"]); cis.append(m["group_bootstrap_ci95"])
vals=np.asarray(vals); cis=np.asarray(cis); x=np.arange(3); axes[0,1].errorbar(x,vals,yerr=[vals-cis[:,0],cis[:,1]-vals],fmt="o",color=CHARCOAL,capsize=3,lw=.8); axes[0,1].axhline(.5,color=GRAY,ls="--",lw=.7); axes[0,1].set_xticks(x,names,rotation=18); axes[0,1].set_ylim(.45,1.03); axes[0,1].set_ylabel("Strict AUROC"); axes[0,1].set_title("Patient-level expression ceiling",loc="left"); panel(axes[0,1],"b"); clean(axes[0,1])

low=gains[gains.candidate.isin(["strict PCA","strict NMF"])].copy(); low["label"]=low.dataset+" · "+low.candidate.str.replace("strict ","",regex=False); low=low.iloc[::-1].reset_index(drop=True); yy=np.arange(len(low)); axes[1,0].errorbar(low.delta_auroc,yy,xerr=[low.delta_auroc-low.delta_ci95_low,low.delta_ci95_high-low.delta_auroc],fmt="none",ecolor=CHARCOAL,lw=.8,capsize=2); axes[1,0].scatter(low.delta_auroc,yy,s=22,color=BLUE,zorder=3); axes[1,0].axvline(0,color=CHARCOAL,lw=.7); axes[1,0].axvline(.05,color=GRAY,lw=.7,ls="--"); axes[1,0].set_yticks(yy,low.label); axes[1,0].set_xlabel("Paired ΔAUROC"); axes[1,0].set_title("Low-rank representations",loc="left"); panel(axes[1,0],"c"); clean(axes[1,0])

allg=gains.copy(); allg["family"]=np.where(allg.candidate.str.contains("PCA|NMF"),"Low-rank",np.where(allg.candidate.str.contains("neighborhood"),"Cell neighborhood","Spatial")); fam=allg.groupby("family",sort=False).delta_auroc.agg(["min","max","mean"]).loc[["Low-rank","Cell neighborhood","Spatial"]]; yy=np.arange(3); axes[1,1].hlines(yy,fam["min"],fam["max"],color=GRAY,lw=3); axes[1,1].scatter(fam["mean"],yy,color=[BLUE,ORANGE,GREEN],s=28,zorder=3); axes[1,1].axvline(0,color=CHARCOAL,lw=.7); axes[1,1].axvline(.05,color=GRAY,lw=.7,ls="--"); axes[1,1].set_yticks(yy,fam.index); axes[1,1].set_xlabel("Range and mean paired ΔAUROC"); axes[1,1].set_title("No family reaches reproducible +0.05",loc="left"); panel(axes[1,1],"d"); clean(axes[1,1])
fig.subplots_adjust(wspace=.52,hspace=.48); save(fig,"fig2_expression_ceiling")


# Figure 3 — global-fit optimism.
fig = plt.figure(figsize=(7.2, 4.5)); gs=fig.add_gridspec(2,3,width_ratios=[.9,1,1.12],hspace=.52,wspace=.48)
ax=fig.add_subplot(gs[:,0]); ax.axis("off"); panel(ax,"a"); ax.set_title("Representation boundary",loc="left")
for y0,t,c in [(.78,"Global PCA / KMeans",ORANGE),(.50,"Held-out patient\nparticipates",ORANGE),(.18,"Strict fold-wise refit",BLUE)]:
    ax.add_patch(FancyBboxPatch((.10,y0-.07),.80,.14,boxstyle="round,pad=.015",facecolor="white",edgecolor=c,lw=1.1,transform=ax.transAxes)); ax.text(.5,y0,t,ha="center",va="center",transform=ax.transAxes,color=c,fontweight="bold",fontsize=5.6)
ax.annotate("optimistic estimate",xy=(.5,.57),xytext=(.5,.66),xycoords="axes fraction",ha="center",color=ORANGE,fontsize=5.1,arrowprops=dict(arrowstyle="-|>",color=ORANGE,lw=.8)); ax.annotate("valid estimate",xy=(.5,.25),xytext=(.5,.34),xycoords="axes fraction",ha="center",color=BLUE,fontsize=5.1,arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=.8))

opt=pd.read_csv(RES/"gse151177_neighborhood_optimism_bootstrap.csv"); k=opt.clusters.to_numpy(); ax=fig.add_subplot(gs[0,1]); panel(ax,"b");
for i,row in opt.iterrows(): ax.plot([0,1],[row.global_auroc,row.strict_auroc],"-",color=GRAY,lw=1); ax.scatter([0,1],[row.global_auroc,row.strict_auroc],color=[ORANGE,BLUE],s=22); ax.text(1.04,row.strict_auroc,f"K={int(row.clusters)}",va="center",fontsize=5.2)
ax.set_xticks([0,1],["Global","Strict"]); ax.set_ylim(.65,1); ax.set_ylabel("AUROC"); ax.set_title("Global versus strict",loc="left"); clean(ax)

ax=fig.add_subplot(gs[0,2]); panel(ax,"c"); yy=np.arange(len(opt)); ax.errorbar(opt.optimism_gap_global_minus_strict,yy,xerr=[opt.optimism_gap_global_minus_strict-opt.patient_bootstrap_ci95_low,opt.patient_bootstrap_ci95_high-opt.optimism_gap_global_minus_strict],fmt="none",ecolor=CHARCOAL,capsize=2,lw=.8); ax.scatter(opt.optimism_gap_global_minus_strict,yy,color=ORANGE,s=24); ax.axvline(0,color=CHARCOAL,lw=.7); ax.axvline(.05,color=GRAY,lw=.7,ls="--"); ax.set_yticks(yy,[f"K={x}" for x in k]); ax.set_xlabel("Global − strict AUROC"); ax.set_title("Independent-unit bootstrap optimism",loc="left"); clean(ax)

abl=pd.read_csv(RES/"patient_balanced_neighborhood_ablation.csv"); ax=fig.add_subplot(gs[1,1]); panel(ax,"d");
for flag,label,c in [(True,"Equal patient mass",BLUE),(False,"Cell-count weighted",GRAY)]:
    q=abl[abl.patient_balanced==flag].sort_values("clusters"); ax.plot(q.clusters,q.loo_auroc,"o-",color=c,lw=1,label=label,ms=3)
ax.set_xticks([8,16,32]); ax.set_xlabel("Neighborhoods"); ax.set_ylabel("Global-fit AUROC"); ax.set_title("Weighting sensitivity",loc="left"); ax.legend(frameon=False); clean(ax)

z=np.load(RES/"cell_cross_patient_shuffle_null_distribution.npz"); null=z["null_aurocs"]; obs=float(z["observed"][0]); ax=fig.add_subplot(gs[1,2]); panel(ax,"e"); ax.hist(null,bins=24,color=GRAY,edgecolor="white"); ax.axvline(obs,color=ORANGE,lw=1.4); ax.set_xlabel("AUROC after cell shuffle"); ax.set_ylabel("Count"); ax.set_title("Cross-patient cell null",loc="left"); ax.text(.04,.92,f"observed={obs:.3f}\np=0.006",transform=ax.transAxes,va="top"); clean(ax)
fig.subplots_adjust(left=.07,right=.98,bottom=.10,top=.95); save(fig,"fig3_unsupervised_optimism")


# Figure 4 — external transfer, calibration and increment.
fig,axes=plt.subplots(2,3,figsize=(7.2,4.9));
ax=axes[0,0]; ax.axis("off"); panel(ax,"a"); ax.set_title("Frozen source-to-target transfer",loc="left")
for x,t,c in [(.03,"GSE151177\nPCA + KMeans",BLUE),(.63,"GSE173706\nno refitting",ORANGE)]: ax.add_patch(FancyBboxPatch((x,.35),.34,.31,boxstyle="round,pad=.02",facecolor="white",edgecolor=c,lw=1.2,transform=ax.transAxes)); ax.text(x+.17,.505,t,ha="center",va="center",transform=ax.transAxes,color=c,fontweight="bold",fontsize=5.6)
ax.annotate("",xy=(.61,.505),xytext=(.39,.505),xycoords="axes fraction",arrowprops=dict(arrowstyle="-|>",color=CHARCOAL,lw=.9)); ax.text(.50,.57,"fixed map",transform=ax.transAxes,ha="center",fontsize=5.1,color=CHARCOAL)

d=np.load(RES/"gse173706_neighborhood_transfer_arrays.npz",allow_pickle=True); y=d["labels"].astype(int); p=d["probabilities"].astype(float); donors=d["donors"].astype(str); fpr,tpr,_=roc_curve(y,p); ax=axes[0,1]; panel(ax,"b"); ax.plot(fpr,tpr,color=BLUE,lw=1.5); ax.plot([0,1],[0,1],"--",color=GRAY,lw=.7); ax.set_xlabel("False-positive rate"); ax.set_ylabel("True-positive rate"); ax.set_title(f"External ranking · AUROC {roc_auc_score(y,p):.3f}",loc="left"); clean(ax)

ax=axes[0,2]; panel(ax,"c"); pairs=pd.DataFrame({"donor":donors,"label":y,"p":p}).pivot(index="donor",columns="label",values="p");
for _,row in pairs.iterrows(): ax.plot([0,1],[row[0],row[1]],"-o",color=GRAY,lw=.7,ms=2.4,alpha=.9)
ax.set_xticks([0,1],["PN","PP"]); ax.set_ylabel("Transferred probability"); ax.set_title("Paired donor scores",loc="left"); clean(ax)

cal=js(RES/"gse173706_neighborhood_transfer_calibration.json"); co=cal["observed"]; ax=axes[1,0]; panel(ax,"d"); ax.plot([0,1],[0,1],"--",color=GRAY,lw=.7); ax.plot(co["reliability_predicted"],co["reliability_observed"],"o-",color=ORANGE,lw=1,ms=3); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency"); ax.set_title("External reliability",loc="left"); ax.text(.04,.93,f"slope={co['calibration_slope']:.1f}\nintercept={co['calibration_intercept']:.1f}\nBrier={co['brier_score']:.3f}",transform=ax.transAxes,va="top",fontsize=5.3); clean(ax)

trgain=gains[(gains.dataset=="GSE173706")&(gains.candidate=="transferred neighborhood")].iloc[0]; ax=axes[1,1]; panel(ax,"e"); ax.errorbar([trgain.delta_auroc],[0],xerr=[[trgain.delta_auroc-trgain.delta_ci95_low],[trgain.delta_ci95_high-trgain.delta_auroc]],fmt="o",color=BLUE,capsize=3); ax.axvline(0,color=CHARCOAL,lw=.7); ax.axvline(.05,color=GRAY,ls="--",lw=.7); ax.set_yticks([0],["Transferred neighborhood"]); ax.set_xlabel("ΔAUROC vs strict expression"); ax.set_title("External incremental value",loc="left"); clean(ax)

z=np.load(RES/"gse173706_neighborhood_transfer_null_distributions.npz"); null=z["permutation_aurocs"]; obs=float(z["observed"][0]); ax=axes[1,2]; panel(ax,"f"); ax.hist(null,bins=24,color=GRAY,edgecolor="white"); ax.axvline(obs,color=ORANGE,lw=1.4); ax.set_xlabel("Paired-label null AUROC"); ax.set_ylabel("Count"); ax.set_title("External label falsification",loc="left"); ax.text(.04,.92,"p=0.0055",transform=ax.transAxes,va="top"); clean(ax)
fig.subplots_adjust(wspace=.52,hspace=.52,left=.07,right=.98,bottom=.10,top=.95); save(fig,"fig4_external_transfer")


# Figure 5 — strict spatial value audit.
fig=plt.figure(figsize=(7.2,4.8)); gs=fig.add_gridspec(2,4,width_ratios=[.85,1,1,.95],wspace=.58,hspace=.52)
ax=fig.add_subplot(gs[0,0]); ax.axis("off"); panel(ax,"a"); ax.set_title("Explicit graph statistics",loc="left")
coords=np.array([[.2,.5],[.42,.72],[.55,.45],[.40,.25],[.72,.65],[.75,.30],[.15,.18]])
for j,(x0,y0) in enumerate(coords): ax.add_patch(Circle((x0,y0),.035,transform=ax.transAxes,facecolor=GREEN if j==2 else "white",edgecolor=GREEN,lw=1));
for j in [0,1,3,4,5]: ax.plot([coords[2,0],coords[j,0]],[coords[2,1],coords[j,1]],color=GRAY,lw=.8,transform=ax.transAxes,zorder=0)
ax.text(.5,.08,"6-NN Moran-like agreement\n+ local contrast",ha="center",transform=ax.transAxes,color=GREEN,fontweight="bold")

ax=fig.add_subplot(gs[0,1:3]); panel(ax,"b"); img=mpimg.imread(RES/"gse206391_representative_LS_NL_spatial_map.png"); rgb=img[...,:3]; mask=np.min(rgb,axis=2)<.985; yy0,xx0=np.where(mask); img=img[max(0,yy0.min()-10):min(img.shape[0],yy0.max()+11),max(0,xx0.min()-10):min(img.shape[1],xx0.max()+11)]; ax.imshow(img); ax.axis("off"); ax.set_title("Predefined median-effect LS/NL pair",loc="left")

sg=js(RES/"gse206391_strict_spatial_graph_benchmark.json")["methods"]; keys=["expression_500","spatial_graph","expression_plus_spatial"]; labels=["Expression-500","Spatial graph","Expression + graph"]; vals=np.array([sg[k]["auroc"] for k in keys]); ci=np.array([sg[k]["group_bootstrap_ci95"] for k in keys]); ax=fig.add_subplot(gs[0,3]); panel(ax,"c"); x=np.arange(3); ax.errorbar(x,vals,yerr=[vals-ci[:,0],ci[:,1]-vals],fmt="o",color=CHARCOAL,capsize=2); ax.axhline(.5,color=GRAY,ls="--",lw=.7); ax.set_xticks(x,labels,rotation=25,ha="right"); ax.set_ylim(.65,1.03); ax.set_ylabel("Strict patient LODO AUROC"); ax.set_title("Matched-gene comparison",loc="left"); clean(ax)

pred=pd.read_csv(RES/"gse206391_strict_spatial_graph_predictions.csv"); methods={"Expression":"expression_500_probability","Spatial":"spatial_graph_probability","Combined":"expression_plus_spatial_probability"}; rows=[]
for name,col in methods.items():
    q=pred.groupby(["patient","true"])[col].mean().unstack(); diff=q[1]-q[0]
    for patient,value in diff.items(): rows.append((name,patient,value))
dd=pd.DataFrame(rows,columns=["method","patient","difference"]); ax=fig.add_subplot(gs[1,0:2]); panel(ax,"d");
for j,name in enumerate(methods):
    q=dd[dd.method==name]; ax.scatter(np.full(len(q),j),q.difference,s=10,facecolor="white",edgecolor=[CHARCOAL, GREEN, BLUE][j],lw=.6); ax.scatter(j,q.difference.mean(),marker="_",s=80,color=[CHARCOAL,GREEN,BLUE][j])
ax.axhline(0,color=CHARCOAL,lw=.7); ax.set_xticks(range(3),methods.keys(),rotation=20); ax.set_ylabel("Patient mean LS − NL score"); ax.set_title("Within-patient separation",loc="left"); clean(ax)

spg=gains[(gains.dataset=="GSE206391") & gains.candidate.isin(["spatial graph","expression + spatial graph"])].iloc[::-1]; ax=fig.add_subplot(gs[1,2]); panel(ax,"e"); yy=np.arange(len(spg)); ax.errorbar(spg.delta_auroc,yy,xerr=[spg.delta_auroc-spg.delta_ci95_low,spg.delta_ci95_high-spg.delta_auroc],fmt="none",ecolor=CHARCOAL,capsize=2,lw=.8); ax.scatter(spg.delta_auroc,yy,color=GREEN,s=22); ax.axvline(0,color=CHARCOAL,lw=.7); ax.axvline(.05,color=GRAY,ls="--",lw=.7); ax.set_yticks(yy,["Expression + graph","Spatial graph"]); ax.set_xlabel("ΔAUROC vs expression-500"); ax.set_title("No spatial increment",loc="left"); clean(ax)

z=np.load(RES/"gse206391_strict_spatial_coordinate_shuffle_distributions.npz"); null=z["null_spatial"]; obs=float(z["observed_spatial"][0]); ax=fig.add_subplot(gs[1,3]); panel(ax,"f"); ax.hist(null,bins=20,color=GRAY,edgecolor="white"); ax.axvline(obs,color=GREEN,lw=1.4); ax.set_xlabel("Coordinate-null AUROC"); ax.set_ylabel("Count"); ax.set_title("Strict coordinate falsification",loc="left"); ax.text(.04,.92,"p=0.446",transform=ax.transAxes,va="top"); clean(ax)
fig.subplots_adjust(left=.07,right=.98,bottom=.10,top=.95); save(fig,"fig5_strict_spatial_value")


# Figure 6 — evidence synthesis and frozen conclusion.
fig=plt.figure(figsize=(7.2,5.2)); gs=fig.add_gridspec(2,2,width_ratios=[1.12,1],height_ratios=[1.2,.8],wspace=.48,hspace=.48)
ax=fig.add_subplot(gs[:,0]); panel(ax,"a"); q=gains.copy(); q["label"]=q.dataset+" · "+q.candidate.str.replace("strict ","",regex=False); q=q.iloc[::-1].reset_index(drop=True); yy=np.arange(len(q)); colors=[BLUE if "PCA" in x or "NMF" in x else (ORANGE if "neighborhood" in x else GREEN) for x in q.candidate]; ax.errorbar(q.delta_auroc,yy,xerr=[q.delta_auroc-q.delta_ci95_low,q.delta_ci95_high-q.delta_auroc],fmt="none",ecolor=CHARCOAL,lw=.7,capsize=2); ax.scatter(q.delta_auroc,yy,c=colors,s=20,zorder=3); ax.axvline(0,color=CHARCOAL,lw=.7); ax.axvline(.05,color=GRAY,ls="--",lw=.8); ax.set_yticks(yy,q.label); ax.set_xlabel("Paired bootstrap ΔAUROC"); ax.set_title("All candidate representations versus matched expression",loc="left"); clean(ax)

ax=fig.add_subplot(gs[0,1]); panel(ax,"b"); cell=js(RES/"cell_cross_patient_shuffle.json"); spatial_label=sg["spatial_graph"]; coord=js(RES/"gse206391_strict_spatial_coordinate_shuffle.json")["null"]["spatial_graph"]; transfer=js(RES/"gse173706_neighborhood_transfer_stability.json"); eff=pd.DataFrame({"test":["Cell shuffle","External labels","Spatial labels","Coordinates"],"effect":[cell["observed_auroc"]-cell["shuffle_mean"],transfer["observed_auroc"]-transfer["permutation_mean"],spatial_label["auroc"]-spatial_label["permutation_mean"],sg["spatial_graph"]["auroc"]-coord["mean"]],"color":[ORANGE,BLUE,BLUE,GREEN]}); x=np.arange(len(eff)); ax.scatter(x,eff.effect,c=eff.color,s=28); ax.vlines(x,0,eff.effect,color=eff.color,lw=1); ax.axhline(0,color=CHARCOAL,lw=.7); ax.set_xticks(x,eff.test,rotation=25,ha="right"); ax.set_ylabel("Observed − null AUROC"); ax.set_title("Falsification distinguishes information sources",loc="left"); clean(ax)

ax=fig.add_subplot(gs[1,1]); panel(ax,"c"); rows=["Strict expression","Strict low-rank","Global neighborhood","Strict neighborhood","External transfer","Trigger OT","Fused GW","Strict spatial graph"]; cols=["Patient","Outer fit","External","Δ≥.05","Null","Calibrated"]; mat=np.array([[1,1,1,np.nan,1,0],[1,1,0,0,1,np.nan],[1,0,0,0,1,np.nan],[1,1,0,0,1,np.nan],[1,1,1,0,1,0],[1,1,0,0,1,np.nan],[1,0,0,0,0,np.nan],[1,1,0,0,1,np.nan]],float); display=np.nan_to_num(mat,nan=-1); cmap=ListedColormap([GRAY,"#F4C7C3","#B7DFC5"]); ax.imshow(display,aspect="auto",cmap=cmap,vmin=-1,vmax=1); ax.set_xticks(range(len(cols)),cols,rotation=35,ha="right"); ax.set_yticks(range(len(rows)),rows)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]): ax.text(j,i,"NA" if np.isnan(mat[i,j]) else ("✓" if mat[i,j]==1 else "×"),ha="center",va="center",fontsize=5.4,color=CHARCOAL)
ax.set_title("Evidence-grade matrix",loc="left"); ax.spines[:].set_visible(False)

box=fig.add_axes([.55,.005,.42,.10]); box.axis("off"); box.add_patch(FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.02",facecolor=LIGHT,edgecolor=CHARCOAL,lw=.8)); box.text(.5,.63,"Disease signal: supported",ha="center",fontweight="bold",color=BLUE); box.text(.5,.28,"Resolution-specific incremental value: not demonstrated",ha="center",fontweight="bold",color=ORANGE)
fig.subplots_adjust(left=.18,right=.98,bottom=.19,top=.95); save(fig,"fig6_falsification_evidence_map")

print(json.dumps({"output":str(OUT),"figures":6,"files":len(list(OUT.glob("*")))},indent=2))
