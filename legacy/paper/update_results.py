"""Render manuscript tables and scientific figures from recorded runs, without training."""
from collections import defaultdict
import os
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import json
from pathlib import Path
import argparse
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import gaussian_filter

PAPER=Path(__file__).resolve().parent
ROOT=PAPER.parent
OUT=ROOT/"outputs/paper_icml2026"
RUNS=OUT/"controlled"
DATA=OUT/"datasets"
GEN=PAPER/"generated"
FIG=PAPER/"figures"
REAL=["volcano","earthquake","flood","fire","general","glycine","proline","prepro","rna"]
LABEL={"volcano":"Volcano","earthquake":"Earthquake","flood":"Flood","fire":"Fire",
       "general":"General","glycine":"Glycine","proline":"Proline","prepro":"Pre-Pro","rna":"RNA (7D)"}
METHOD={"mgm_gmm":"R-MGM mixture (ours)","mgm_mlp":"R-MGM MLP (ours)",
        "mmd_mlp":"MMD MLP control","rfm":"Controlled RFM"}
METRICS=["mmd2_biased","symmetric_nn","coverage","constraint_mean","gpu_sample_seconds",
         "cpu_sample_seconds_mean","train_seconds","target_n"]
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8,"pdf.fonttype":42,
                     "ps.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))


def collect():
    groups=defaultdict(lambda:defaultdict(list))
    for run in RUNS.glob("*/*/seed*"):
        if not (run/"completed.json").exists():continue
        for path in run.glob("evaluation_step*.json"):
            for r in read(path):
                key=(r["data"],r["method"],r["step"],r["sampler_nfe"],r["split"])
                groups[key][r["training_seed"]].append(r)
    stats={}
    for key,seeds in groups.items():
        entry={"n_seeds":len(seeds),"seed_values":{}}
        for metric in METRICS:
            values=[float(np.mean([r[metric] for r in rows])) for _,rows in sorted(seeds.items())
                    if all(metric in r for r in rows)]
            if values:
                entry[metric]=(float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.)
                entry["seed_values"][metric]=values
        stats[key]=entry
    return stats


def stat(s,data,method,step,nfe=1,split="test"):
    return s.get((data,method,step,nfe,split))


def fmt(entry,metric="mmd2_biased",scale=1000,precision=2):
    if not entry or entry["n_seeds"]<3 or metric not in entry:return "---"
    mean,sd=entry[metric]
    return f"${mean*scale:.{precision}f}\\,\\pm\\,{sd*scale:.{precision}f}$"


def write(name,lines):
    GEN.mkdir(exist_ok=True)
    (GEN/name).write_text("\n".join(lines)+"\n",encoding="utf-8")


def main_table(s):
    lines=[r"\begin{table*}[t]",r"\caption{\textbf{Controlled real-data sample quality.} Biased $\widehat{\mathrm{MMD}}^2\times10^3$ (lower is better), mean $\pm$ sample SD over three training seeds, after averaging three draws per seed. Mixture initialization uses training data only. RFM uses 128 field calls at inference. Budgets are updates/outer iterations, not equal FLOPs; the smaller controlled RFM is not the original author-budget model.}",
           r"\label{tab:main}",r"\centering\small",r"\begin{tabular}{@{}lrrrrr@{}}",r"\toprule",
           r"Dataset & Test $n$ & Mixture init. & R-MGM mixture (ours) & RFM & RFM (longer) \\",
           r" & & 0 updates & 2,000 iterations & 2,000 updates & 10,000 updates \\",r"\midrule"]
    for data in REAL:
        x=stat(s,data,"mgm_gmm",2000)
        n=int(x["target_n"][0]) if x else 0
        cells=[LABEL[data],str(n) if n else "---",fmt(stat(s,data,"mgm_gmm",0)),fmt(x),
               fmt(stat(s,data,"rfm",2000,128)),fmt(stat(s,data,"rfm",10000,128))]
        lines.append(" & ".join(cells)+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table*}"])
    write("main_results.tex",lines)
    complete=[d for d in REAL if all(stat(s,d,m,t,n) and stat(s,d,m,t,n)["n_seeds"]==3
               for m,t,n in [("mgm_gmm",0,1),("mgm_gmm",2000,1),("rfm",2000,128),("rfm",10000,128)])]
    if complete:
        beats=lambda m,t,n:sum(stat(s,d,"mgm_gmm",2000)["mmd2_biased"][0]<stat(s,d,m,t,n)["mmd2_biased"][0] for d in complete)
        text=(f"Table~\\ref{{tab:main}} reports fixed-checkpoint outcomes. Across the {len(complete)} fully evaluated tasks, "
              f"midpoint-trained mixtures have lower mean MMD than their initialization on {beats('mgm_gmm',0,1)}, "
              f"the 2,000-update flow on {beats('rfm',2000,128)}, and the longer 10,000-update flow on {beats('rfm',10000,128)}. "
              "These are descriptive counts, not significance tests or a ranking against author-budget models. "
              f"Training worsens the mean initialization score on {len(complete)-beats('mgm_gmm',0,1)} tasks at these settings. "
              "The initialization column attributes fit present before training to the restricted mixture model. The degradations expose failures of the implemented finite-critic training procedure, but do not isolate a unique cause or support universal improvement. "
              "Figure~\\ref{fig:controlled} shows fixed-seed sample visualizations; full-dimensional numerical evaluation is retained for RNA.")
    else:text="The machine-readable controlled experiment ledger is being completed."
    reps=[d for d in ["fire","general","rna"] if stat(s,d,"mmd_mlp",2000) and stat(s,d,"mgm_mlp",2000)
          and stat(s,d,"mmd_mlp",2000)["n_seeds"]==3 and stat(s,d,"mgm_mlp",2000)["n_seeds"]==3]
    if reps:
        nbetter=sum(stat(s,d,"mmd_mlp",2000)["mmd2_biased"][0]<stat(s,d,"mgm_mlp",2000)["mmd2_biased"][0] for d in reps)
        text+=f" The matched MMD-trained MLP has lower mean MMD than the midpoint-trained MLP on {nbetter} of {len(reps)} representative tasks (Table~\\ref{{tab:ablation}}); these runs do not establish an advantage of the midpoint objective over this direct loss."
        nn=sum(stat(s,d,"mmd_mlp",2000)["symmetric_nn"][0]<stat(s,d,"mgm_mlp",2000)["symmetric_nn"][0] for d in reps)
        coverage=sum(stat(s,d,"mmd_mlp",2000)["coverage"][0]>stat(s,d,"mgm_mlp",2000)["coverage"][0] for d in reps)
        text+=f" It also has lower mean nearest-neighbor distance on {nn} and higher coverage on {coverage} of these tasks (supplemental Table~\\ref{{tab:mlp-metrics}}), without implying a universal objective ranking."
    text+="\n\n\\input{generated/ablation_results.tex}\n"
    write("result_observations.tex",[text])


def ablations(s):
    choices=[("Init. GMM","mgm_gmm",0,1),("R-MGM GMM","mgm_gmm",2000,1),
             ("R-MGM MLP","mgm_mlp",2000,1),("MMD MLP","mmd_mlp",2000,1),
             ("RFM (1)","rfm",2000,1),("RFM (128)","rfm",2000,128)]
    lines=[r"\begin{table}[t]",r"\caption{\textbf{Initialization and objective controls.} Biased MMD$^2\times10^3$, mean $\pm$ SD across three training seeds. All trained rows use 2,000 iterations/updates. MLP rows share architecture and output constraints; mixture rows have a different restricted parameterization.}",r"\label{tab:ablation}",r"\centering\small",r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",r"Method & Fire & General & RNA \\",r"\midrule"]
    lines.insert(lines.index(r"\begin{tabular}{@{}lrrr@{}}"),r"\setlength{\tabcolsep}{4pt}")
    for label,m,t,n in choices:lines.append(" & ".join([label]+[fmt(stat(s,d,m,t,n),precision=1) for d in ["fire","general","rna"]])+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    write("ablation_results.tex",lines)
    lines=[r"\begin{table}[t]",r"\caption{\textbf{Sampling latency on Fire.} Milliseconds per batch of 1,024, mean $\pm$ SD over training-seed means; CUDA-synchronized GPU and single-thread CPU measurements exclude warmup. Calls count a generator or a field evaluation; the mixture map is not a neural network. RNG and constraints are included.}",r"\label{tab:latency}",r"\centering\small",r"\begin{tabular}{@{}lrrr@{}}",r"\toprule",r"Method & Calls & GPU ms & CPU ms \\",r"\midrule"]
    for label,m,t,n in choices[1:]:
        row=stat(s,"fire",m,t,n)
        lines.append(" & ".join([label,str(n),fmt(row,"gpu_sample_seconds",1000),fmt(row,"cpu_sample_seconds_mean",1000)])+r" \\")
    lines.extend([r"\bottomrule",r"\end{tabular}",r"\end{table}"])
    write("latency_results.tex",lines)


def protocol(s):
    lines=[r"\subsection{Fixed Primary Protocol}",
           "Model seeds are 0, 1, 2 and the fixed data-split seed is 0. MLPs have two hidden layers of width 128 (three affine layers), SiLU activations, and a 16-dimensional standard-normal latent. Critics and flow fields use eight Fourier time frequencies and raw wrapped spatial angles on tori. The kernel embedding used for evaluation is a separate circular embedding; it is not the critic input.",
           "All runs use batch 256, float32, AdamW learning rate $2\\times10^{-4}$ and weight decay zero. There is one critic update and one generator update per midpoint iteration. No neural-flow teacher is used. The mixture initialization is a train-fitted statistical starting model, not a pretrained flow. Torus mixtures use 64 components for General and 32 for other torsion datasets; sphere mixtures use 12 for Volcano and 24 otherwise. K-means initialization uses at most 8,192 training samples. Hard component selection leaves mixture-weight logits fixed during pathwise training; only centers and scales receive gradients.",
           "The MMD control has the same constrained MLP output map and architecture as the MLP midpoint model and optimizes the biased batch MMD with the fixed training-reference kernel. It is an objective control, not an implementation of a kernel drifting algorithm. Every primary model is evaluated at prespecified steps; validation and test metrics are both saved but neither selects checkpoints or settings. Earlier test exposure is disclosed, so new metrics describe a locked evaluation on an already explored benchmark.",
           r"\subsection{Data Counts and Evaluation Size}",r"Table~\ref{tab:counts} records the local data counts and fixed evaluation size.",r"\begin{table}[t]\caption{Local preprocessing counts and fixed split sizes. The RNA count is 9,473; it is not asserted to equal every published preprocessing release.}\label{tab:counts}\centering\small\begin{tabular}{@{}lrrrrr@{}}\toprule",r"Data & Total & Train & Validation & Test & Evaluated $n$ \\",r"\midrule"]
    for data in REAL:
        path=DATA/data/"manifest.json"
        if path.exists():
            c=read(path)["counts"];n=min(1024,c["test"])
            lines.append(" & ".join([LABEL[data],str(sum(c.values())),str(c["train"]),str(c["val"]),str(c["test"]),str(n)])+r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
          "All splits are by row. Molecular-source identifiers are not used for group splitting, and geographical rows can be dependent or duplicated. The conclusions concern the specified density-learning protocol, not unseen proteins, independent events, or temporal forecasting.",
          r"\subsection{Metrics}",
          r"For matched-size samples $x_{1:n}$ and $y_{1:n}$, the reported statistic is",
          r"\begin{equation}\widehat{\mathrm{MMD}}_V^2=\frac{1}{n^2}\sum_{i,j=1}^n\big[k(x_i,x_j)+k(y_i,y_j)-2k(x_i,y_j)\big].\end{equation}",
          "The kernel denominator $h^2$ is the median strictly positive squared embedded distance among a fixed random training reference of at most 1,024 points. It is not fitted to generated or test samples. The diagonal terms are included, making this a biased V-statistic with a finite-sample floor; it is not an unbiased estimate. Test and validation targets are sampled without replacement using fixed independent target-selection seeds and reused across models and training seeds.",
          "Nearest-neighbor distances use spherical angles and wrapped Euclidean angle distance on a flat torus. Spherical vectors are normalized in float64 and their angle is computed as $2\\operatorname{atan2}(\\|\\hat x-\\hat y\\|_2,\\|\\hat x+\\hat y\\|_2)$, avoiding false positive distances between identical float32 directions. Symmetric NN is one half the sum of the two directed mean nearest distances. Coverage is the fraction of fixed target points within radius $r$ of a generated point; $r$ is the median positive leave-one-out nearest distance on the fixed training reference. Identical normalized directions are explicitly excluded from the sphere reference calibration. Mesh distances below are ambient Euclidean, not intrinsic geodesic distances.",
          "Three generated draws are averaged within each training seed, followed by mean and sample standard deviation across the three independent training seeds. Draws are Monte Carlo replicates, not extra training seeds. Sphere validity is mean absolute unit-norm error. Torus validity is the fraction outside the half-open angle interval $[0,2\\pi)$.",
          r"\subsection{Sampling and Compute}",
          "The controlled RFM uses projected/exponential Euler with midpoint-in-time evaluation and one actual vector-field call per step. Both 1- and 128-NFE cases are saved at each prespecified checkpoint. Forward hooks verify counts. CUDA is synchronized around timing; GPU warmups and metric calculations are excluded. Single-thread CPU timing uses three repeated batches. Data preparation and mixture initialization are separately recorded, and training duration excludes evaluation pauses. This is not a claim of identical training FLOPs or total wall-clock cost across algorithms."])
    alljobs=list(RUNS.glob("*/*/seed*/completed.json"))
    if alljobs:
        seconds=sum(read(p).get("train_seconds",0) for p in alljobs)
        lines.append(f"The final controlled bundle contains {len(alljobs)} completed training jobs, with {seconds/60:.2f} minutes of summed synchronized training time. This sum excludes evaluation, setup, discarded preliminary reruns, exploratory historical jobs, and incomplete attempts; it is not total project GPU time. Hardware: one NVIDIA GeForce RTX 4090, PyTorch 2.5.1 with CUDA 12.1. Other GPU processes were observed during part of this bundle; synchronized durations are actual wall times, not isolated hardware costs or proof of a speedup. Individual runs record their own durations.")
    if RUNS.parent.name=="controlled_independent":
        lines.extend([r"\subsection{Randomness and Metric Audit}",
            "A timing-helper audit exposed evaluation-induced resetting of a global CUDA stream in the preliminary GMM/RFM runs. The final bundle replaces all 54 affected runs using frozen sources and separate target, latent, time/flip, and flow-base RNG streams derived from model seed. It copies the 18 unaffected MLP objective-control jobs with file hashes. Evaluation is asserted to preserve all training streams at every checkpoint. Rerun checkpoints save optimizer and CPU/CUDA/private RNG states; the copied earlier MLP checkpoints are suitable for reevaluation but lack full continuation state. GPU reductions are not guaranteed bitwise identical across devices.",
            "A separate analysis-only correction recomputes sphere angles and coverage from saved arrays with the normalized float64 metric above. Raw metric JSONs are archived with correction hashes, and training sources and generated arrays are unchanged. The final-table marker requires the corrected metric version. Both audits precede the final aggregation; neither selects models by their test scores."])
    lines.extend([r"\subsection{Reproduction Commands}",
          r"From the repository root with the provided conda environment, the following commands check and render the final recorded bundle. The experiment README gives the ordered training, immutable-snapshot rerun, metric-correction, and validation-only commands for a clean reconstruction.",
          r"\begingroup\small\begin{verbatim}",
          r"python paper/experiments/verify_independent.py",
          r"python paper/check_geometry.py",
          r"python paper/make_illustrations.py",
          r"python paper/update_results.py --run-root controlled_independent --require-complete",
          r"powershell -File paper/build.ps1",
          r"\end{verbatim}\endgroup",
          "Data/split hashes and executed-code hashes accompany the final result ledger. Natural Earth public-domain land outlines are used only as geographic context in maps; density color fields are computed from saved samples."])
    lines.append("Display KDEs are separate from the evaluation MMD kernel. Spherical plots use a normalized von Mises--Fisher kernel with concentration 45 (density per steradian); torus plots smooth a 120-by-120 periodic histogram using Gaussian sigma 2.4 bins and divide by angular bin area (density per squared radian). Comparisons use one logarithmic color scale and clip values below $10^{-3}$ of the maximum within that comparison. Horizontal geographic and angle domains span $[-180,180]$ degrees; only the first two RNA angles are displayed.")
    write("protocol_details.tex",lines)


def kde(ax,arr,data,vmax=None):
    if data in REAL[:4]:
        lo,la=np.meshgrid(np.linspace(-180,180,145),np.linspace(-89,89,73))
        lon,lat=np.deg2rad(lo),np.deg2rad(la)
        grid=np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],-1)
        q=grid.reshape(-1,3)
        dens=np.zeros(len(q))
        for start in range(0,len(arr),256):dens+=np.exp(45*(q@arr[start:start+256].T-1)).sum(1)
        dens=dens.reshape(lo.shape)/max(1,len(arr))*45/(2*np.pi*(1-np.exp(-90)))
        if vmax is None:return dens.max()
        ax.pcolormesh(lo,la,np.maximum(dens,vmax*1e-3),cmap="YlGnBu",norm=LogNorm(vmin=vmax*1e-3,vmax=vmax),shading="auto",rasterized=True)
        coast=read(PAPER/"vendor/ne_110m_land.geojson")
        for feature in coast["features"]:
            geom=feature["geometry"]
            polys=geom["coordinates"] if geom["type"]=="MultiPolygon" else [geom["coordinates"]]
            for poly in polys:
                p=np.array(poly[0]); ax.plot(p[:,0],p[:,1],color="#777777",lw=.28,alpha=.8)
        ax.set(xlim=(-180,180),ylim=(-90,90),xticks=[-120,0,120],yticks=[-60,0,60])
    else:
        xy=(np.rad2deg(arr[:,:2])+180)%360-180
        h,xe,ye=np.histogram2d(xy[:,0],xy[:,1],bins=120,range=[[-180,180],[-180,180]])
        h=gaussian_filter(h,sigma=2.4,mode="wrap")/max(1,len(arr))/(2*np.pi/120)**2
        if vmax is None:return h.max()
        ax.pcolormesh(xe,ye,np.maximum(h.T,vmax*1e-3),cmap="YlGnBu",norm=LogNorm(vmin=vmax*1e-3,vmax=vmax),shading="flat",rasterized=True)
        ax.set(xlim=(-180,180),ylim=(-180,180),xticks=[-120,0,120],yticks=[-120,0,120])
    ax.tick_params(labelsize=6,length=2)
    ax.set_aspect("auto")
    return None


def sample_array(data,method,step,nfe=1):
    p=RUNS/data/method/"seed0"/f"samples_step{step}_nfe{nfe}.npz"
    return np.load(p)["generated_draw_0"] if p.exists() else None


def plots():
    cases=["fire","general","rna"]
    cols=[("Held-out data",None,0,1),("R-MGM\nmixture","mgm_gmm",2000,1),
          ("R-MGM\nMLP","mgm_mlp",2000,1),("RFM (10k)\n128 calls","rfm",10000,128)]
    fig,axes=plt.subplots(3,4,figsize=(7,4.25))
    for i,data in enumerate(cases):
        targetpath=DATA/data/"split_indices_and_targets.npz"
        if not targetpath.exists():continue
        target=np.load(targetpath)["test_target"]
        arrays=[target if m is None else sample_array(data,m,t,n) for _,m,t,n in cols]
        v=max(kde(None,a,data) for a in arrays if a is not None)
        for j,((label,_,_,_),arr) in enumerate(zip(cols,arrays)):
            ax=axes[i,j]
            if arr is None:ax.axis("off");continue
            kde(ax,arr,data,v)
            if i==0:ax.set_title(label,fontsize=8)
            if j==0:ax.set_ylabel(LABEL[data]+("\nLatitude" if data=="fire" else "\nAngle 2"),fontsize=8)
            if i==2:ax.set_xlabel("Angle 1 (degrees)",fontsize=7)
            if j:ax.set_yticklabels([])
    fig.subplots_adjust(left=.10,right=.92,bottom=.10,top=.91,wspace=.13,hspace=.24)
    for i,data in enumerate(cases):
        target=np.load(DATA/data/"split_indices_and_targets.npz")["test_target"]
        arrays=[target if m is None else sample_array(data,m,t,n) for _,m,t,n in cols]
        vmax=max(kde(None,a,data) for a in arrays if a is not None)
        pos=axes[i,-1].get_position()
        cax=fig.add_axes([.94,pos.y0,.012,pos.height])
        cb=fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmax*1e-3,vmax),cmap="YlGnBu"),cax=cax)
        cb.ax.tick_params(labelsize=5,length=2)
    fig.savefig(FIG/"controlled_samples.pdf",bbox_inches="tight",dpi=260)
    fig.savefig(FIG/"controlled_samples.png",bbox_inches="tight",dpi=350)
    plt.close(fig)
    for data in REAL:
        targetpath=DATA/data/"split_indices_and_targets.npz"
        if not targetpath.exists():continue
        choices=[("Held-out data",None,0,1),("Mixture init.","mgm_gmm",0,1),
                 ("R-MGM mixture (ours)","mgm_gmm",2000,1),("RFM 2k / 128 NFE","rfm",2000,128),
                 ("RFM 10k / 128 NFE","rfm",10000,128)]
        arrays=[np.load(targetpath)["test_target"] if m is None else sample_array(data,m,t,n) for _,m,t,n in choices]
        if any(x is None for x in arrays):continue
        vmax=max(kde(None,a,data) for a in arrays)
        fig,axes=plt.subplots(1,5,figsize=(14,3))
        for ax,(title,_,_,_),arr in zip(axes,choices,arrays):
            kde(ax,arr,data,vmax);ax.set_title(title,fontsize=11)
        fig.subplots_adjust(left=.04,right=.99,bottom=.14,top=.85,wspace=.12)
        fig.savefig(FIG/f"{data}_comparison.png",bbox_inches="tight",dpi=320)
        plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(7,3.5))
        for ax,(title,_,_,_),arr in zip(axes.flat,choices,arrays):
            kde(ax,arr,data,vmax);ax.set_title(title,fontsize=8)
        axes.flat[-1].axis("off")
        fig.subplots_adjust(left=.06,right=.99,bottom=.08,top=.91,wspace=.20,hspace=.4)
        fig.savefig(FIG/f"{data}_comparison.pdf",bbox_inches="tight",dpi=260)
        plt.close(fig)


def objective_plots():
    columns=[("Held-out data",None),("Midpoint MLP","mgm_mlp"),("Direct-MMD MLP","mmd_mlp")]
    fig,axes=plt.subplots(3,3,figsize=(7,4.25))
    scales=[]
    for i,data in enumerate(["fire","general","rna"]):
        target=np.load(DATA/data/"split_indices_and_targets.npz")["test_target"]
        arrays=[target]+[sample_array(data,m,2000) for _,m in columns[1:]]
        assert all(a is not None for a in arrays)
        vmax=max(kde(None,a,data) for a in arrays);scales.append(vmax)
        for j,(ax,arr) in enumerate(zip(axes[i],arrays)):
            kde(ax,arr,data,vmax)
            if i==0:ax.set_title(columns[j][0],fontsize=8)
            if j==0:ax.set_ylabel(LABEL[data]+("\nLatitude" if data=="fire" else "\nAngle 2"),fontsize=8)
            else:ax.set_yticklabels([])
            if i==2:ax.set_xlabel("Angle 1 (degrees)",fontsize=7)
    fig.subplots_adjust(left=.10,right=.92,bottom=.10,top=.92,wspace=.14,hspace=.24)
    for i,vmax in enumerate(scales):
        pos=axes[i,-1].get_position()
        cax=fig.add_axes([.94,pos.y0,.012,pos.height])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=LogNorm(vmin=vmax*1e-3,vmax=vmax),cmap="YlGnBu"),cax=cax)
        bar.ax.tick_params(labelsize=6,length=2)
    fig.savefig(FIG/"objective_controls.pdf",bbox_inches="tight",dpi=260)
    fig.savefig(FIG/"objective_controls.png",bbox_inches="tight",dpi=350)
    plt.close(fig)


def mesh_graphics():
    import torch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    variants=[("target","Target"),("projection","Projected mixture"),("sphere_map","Soft sphere map"),
              ("sphere_map_gmm","Sphere-map mixture")]
    datasets=["bunny10","bunny50","bunny100","spot10","spot50","spot100"]
    entries={}
    for data in datasets:
        base=OUT/"legacy_mesh"/data
        if not (base/"target.npz").exists() or not all((base/f"{v}_samples.npz").exists() for v,_ in variants[1:]):continue
        arrays={"target":np.load(base/"target.npz")["target"]}
        arrays.update({v:np.load(base/f"{v}_samples.npz")["generated_draw_0"] for v,_ in variants[1:]})
        checkpoint=next((ROOT/"outputs/rmgm_mesh_bunny_spot_pilot/projection"/data/"rmgm").glob("seed0_*/checkpoints/last.pt"))
        state=torch.load(checkpoint,map_location="cpu",weights_only=True)["generator"]
        entries[data]=(state["manifold.v"].numpy(),state["manifold.f"].numpy(),arrays)

    def draw(ax,data,variant,title=True):
        v,f,arrays=entries[data]
        surface=Poly3DCollection(v[f],facecolor="#d0d4d8",edgecolor="none",alpha=.15,rasterized=True)
        ax.add_collection3d(surface)
        p=arrays[variant]
        color={"target":"#33383d","projection":"#13836d","sphere_map":"#bb463d","sphere_map_gmm":"#2766a2"}[variant]
        ax.scatter(*p.T,s=1.9,color=color,alpha=.65,linewidths=0,depthshade=False,rasterized=True)
        lo,hi=v.min(0),v.max(0);c=(lo+hi)/2;r=(hi-lo).max()/2*1.02
        ax.set(xlim=(c[0]-r,c[0]+r),ylim=(c[1]-r,c[1]+r),zlim=(c[2]-r,c[2]+r))
        ax.set_box_aspect((1,1,1));ax.view_init(18,-62);ax.set_axis_off()
        if title:ax.set_title(dict(variants)[variant],fontsize=8,pad=0)

    for data in entries:
        fig=plt.figure(figsize=(7,2.2))
        for j,(variant,title) in enumerate(variants):
            ax=fig.add_subplot(1,4,j+1,projection="3d");draw(ax,data,variant)
        fig.subplots_adjust(left=0,right=1,bottom=0,top=.85,wspace=0)
        fig.savefig(FIG/f"{data}_surface.pdf",bbox_inches="tight",dpi=260)
        fig.set_size_inches(14,4.4)
        fig.savefig(FIG/f"{data}_surface.png",bbox_inches="tight",dpi=320)
        plt.close(fig)
    selected=[d for d in ["bunny50","spot50"] if d in entries]
    if len(selected)==2:
        fig=plt.figure(figsize=(7,3.25))
        for i,data in enumerate(selected):
            for j,(variant,_) in enumerate(variants):
                ax=fig.add_subplot(2,4,i*4+j+1,projection="3d");draw(ax,data,variant,i==0)
                if j==0:ax.text2D(-.02,.5,data.replace("50"," $k=50$"),transform=ax.transAxes,rotation=90,va="center",fontsize=8)
        fig.subplots_adjust(left=.025,right=1,bottom=0,top=.91,wspace=0,hspace=.04)
        fig.savefig(FIG/"mesh_selected.pdf",bbox_inches="tight",dpi=280)
        fig.savefig(FIG/"mesh_selected.png",bbox_inches="tight",dpi=350)
        plt.close(fig)


def stability():
    records=[]
    for base in sorted(OUT.glob("stabilization_general*")):
        path=base/"results.json"
        if not path.exists():continue
        for r in read(path):
            if "final_validation_mmd2_mean" not in r:continue
            cfgpath=base/r["trial"]/"config.json"
            cfg=read(cfgpath) if cfgpath.exists() else {}
            records.append((base,r,cfg))
    lines=["This post hoc diagnostic was motivated by the primary failures. It uses General training data and its fixed validation target only, model seed 0, and three generated draws per probe. No new test score is obtained or used to choose a candidate. Trials are reported together rather than replacing the unsuccessful primary setting. These exploratory results do not establish cross-seed robustness.",
           r"Table~\ref{tab:stability} records the initial and final validation outcomes.",
           r"\begin{table}[t]\caption{Validation-only General optimization study. All values are means of three Monte Carlo draws from one seed-0 model, not independent training replicates. MMD columns are multiplied by $10^3$; optimizer counts include critic warmup.}\label{tab:stability}\centering\small",
           r"\begin{tabular}{@{}llrrrrr@{}}\toprule",
           r"Configuration & Generator LR & Gen./field steps & Total updates & Initial val. & Final val. & Train s \\",r"\midrule"]
    fig,ax=plt.subplots(figsize=(7,2.7))
    count=0
    for base,r,cfg in records:
        p=cfg.get("protocol",{});t=cfg.get("trial",{})
        isrfm=t.get("method")=="rfm" or r["trial"]=="periodic_rfm"
        periodic=t.get("periodic",True)
        init=t.get("initialization","")
        initname={"random_mlp":"random MLP","random_gmm":"random GMM","train_kmeans_gmm":"train-fit GMM"}.get(init,"RFM" if isrfm else r["trial"].replace("_"," "))
        label=("Periodic " if periodic else "Raw ")+initname
        cs=t.get("critic_steps",0)
        if not isrfm:label+=f", C={cs}"
        lr=p.get("lr_rfm",2e-4) if isrfm else p.get("lr_g",0.)
        cells=[label,f"{lr:.0e}",str(r["steps"]),str(r["total_optimizer_updates"]),
               f"{r['initial_validation_mmd2_mean']*1000:.2f}",f"{r['final_validation_mmd2_mean']*1000:.2f}",f"{r['train_seconds']:.1f}"]
        lines.append(" & ".join(cells)+r" \\")
        if r["trial"] not in {"periodic_kmeans_gmm_c5","periodic_random_mlp_c5","periodic_random_mlp_c5_lr2e4","periodic_random_gmm_c5_lr2e4_g10000","periodic_rfm"}:continue
        curve=[]
        for path in (base/r["trial"]).glob("*_metrics.json"):
            rr=[q for q in read(path) if q.get("split")=="val"]
            if not rr:continue
            q=rr[0]
            x=q["step"] if isrfm else (0 if q["phase"]=="initial" else p.get("critic_warmup",0)+(cs+1)*q["step"])
            curve.append((x,np.mean([v["mmd2_biased"] for v in rr])))
        curve.sort()
        if curve:
            a=np.array(curve);ax.semilogy(a[:,0],a[:,1],"o-",lw=1.3,markersize=3,label=label+f", lr={lr:.0e}");count+=1
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
           "The periodic critic/field uses sine and cosine spatial harmonics 1--4, eight Fourier time frequencies, width 128, and three affine layers. The study adds 500 critic-only warmup updates, five or ten critic updates per generator iteration, and slower generator learning rates, with the higher-rate refinement recorded separately. Its flow control uses the same periodic field architecture and fixed update counts; it must not be confused with the raw-angle controlled RFM in the primary table."])
    if count:
        ax.set(xlabel="Total optimizer updates (critic + generator, or flow)",ylabel="Validation MMD$^2$")
        ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,ncol=2)
        fig.tight_layout();fig.savefig(FIG/"validation_stability.pdf",bbox_inches="tight")
        fig.savefig(FIG/"validation_stability.png",bbox_inches="tight",dpi=320)
        lines.extend([r"Figure~\ref{fig:stability} shows illustrative configurations from this post hoc study.",
              r"\begin{figure}[t]\centering\includegraphics[width=\textwidth]{figures/validation_stability.pdf}",
              r"\caption{General validation diagnostics from one training seed. Lines join fixed probe checkpoints; they are not interpolated test estimates. Curves illustrate retention, random initialization, higher-rate refinement, and periodic RFM; all completed trial outcomes are listed in Table~\ref{tab:stability}. Update counts are not equal FLOPs; measured training times appear in the table.}\label{fig:stability}\end{figure}"])
    plt.close(fig)
    write("stabilization_results.tex",lines)


def noise_diagnostics():
    base=OUT/"noisy_torus_study"
    path=base/"results.json"
    if not path.exists():
        write("noisy_torus_results.tex",["The randomized path in Appendix~\\ref{app:noisy-torus} is a mathematical construction; no optimization benefit is assumed from its existence."])
        return
    records=read(path)
    assert len(records)==3,"Do not render a partial noise diagnostic"
    lines=[r"\subsection{True-Velocity Randomized Paths}",
        r"We separately probe the reversible randomized torus path in \cref{eq:noisy-torus} on General, with $\varepsilon\in\{0,0.3,0.7\}$. The actual derivative includes the noise-amplitude derivative. The experiment uses a randomly initialized MLP, the periodic critic, seed 0, batch 512, 10,000 generator updates, five critic updates per generator step, 500 critic warmup updates, and generator/critic learning rates $2\times10^{-4}$. All three amplitudes and fixed checkpoints are specified before this diagnostic. It is post hoc validation-only evidence, not a replacement for the primary table or independent-seed confirmation. Evaluation preserves the training RNG state.",
        r"Table~\ref{tab:noise} lists every amplitude. Figure~\ref{fig:noise} shows the fixed-checkpoint validation curves. The $\varepsilon=0$ row is a separately executed deterministic control with the same code path and random draws, not a copied earlier result.",
        r"\begin{table}[t]\caption{Randomized-torus validation diagnostic. One seed-0 model per amplitude; MMD$^2\times10^3$ means average three sampling draws. The final SD is Monte Carlo variation only. No test metrics are evaluated.}\label{tab:noise}\centering\small\begin{tabular}{@{}rrrrr@{}}\toprule",
        r"Amplitude $\varepsilon$ & Initial val. & Final val. & Final MC SD & Train s \\",r"\midrule"]
    fig,ax=plt.subplots(figsize=(7,2.65))
    for r in records:
        lines.append(f"{r['noise_amplitude']:g} & {r['initial_validation_mmd2_mean']*1000:.2f} & {r['final_validation_mmd2_mean']*1000:.2f} & {r['final_validation_mmd2_sd_mc']*1000:.2f} & {r['train_seconds']:.1f}"+r" \\")
        curve=[]
        for p in (base/r["trial"]).glob("*_metrics.json"):
            rows=[q for q in read(p) if q["split"]=="val"]
            if rows:curve.append((rows[0]["step"],np.mean([q["mmd2_biased"] for q in rows])))
        curve.sort();a=np.array(curve)
        ax.semilogy(a[:,0],a[:,1],"o-",lw=1.3,markersize=3,label=f"$\\varepsilon={r['noise_amplitude']:g}$")
    lines.append(r"\bottomrule\end{tabular}\end{table}")
    lines.append("Noise improves the final validation MMD relative to this deterministic control in the single-seed sweep, but all three models remain above the periodic RFM reference (approximately $1.91\\times10^{-3}$). Their 60,500 total optimizer updates exceed that reference's 22,500 field updates; neither update counts nor FLOPs are matched. This diagnostic does not establish cross-seed efficacy or an advantage over the flow.")
    ax.set(xlabel="Generator updates",ylabel="Validation MMD$^2$")
    ax.grid(alpha=.2);ax.legend(frameon=False,ncol=3)
    fig.tight_layout();fig.savefig(FIG/"validation_noise.pdf",bbox_inches="tight")
    fig.savefig(FIG/"validation_noise.png",bbox_inches="tight",dpi=320);plt.close(fig)
    lines.extend([r"\begin{figure}[t]\centering\includegraphics[width=\textwidth]{figures/validation_noise.pdf}",
        r"\caption{Validation-only true-velocity noise diagnostic. Lines join prespecified checkpoints for three seed-0 models. Different amplitudes define different path discrepancies; these curves do not estimate the population divergence or establish a generally optimal noise level.}\label{fig:noise}\end{figure}"])
    write("noisy_torus_results.tex",lines)


def extra(s):
    lines=[r"\subsection{Full Sample Metrics}",r"Table~\ref{tab:all-metrics} supplements Table~\ref{tab:main} with distances, coverage, and validity.",
           r"\begin{table}[t]\caption{Additional controlled test metrics, mean $\pm$ training-seed SD for NN and coverage. Validity reports the mean error defined in the text.}\label{tab:all-metrics}\centering\small\begin{tabular}{@{}llrrr@{}}\toprule",
           r"Data & Method / updates & Symmetric NN $\downarrow$ & Coverage $\uparrow$ & Validity error $\downarrow$ \\",r"\midrule"]
    for data in REAL:
        for method,step,nfe,label in [("mgm_gmm",2000,1,"R-MGM mixture / 2k"),("rfm",2000,128,"RFM / 2k"),("rfm",10000,128,"RFM / 10k")]:
            x=stat(s,data,method,step,nfe)
            if not x:continue
            validity=x.get("constraint_mean",(0,0))[0]
            lines.append(" & ".join([LABEL[data],label,fmt(x,"symmetric_nn",1,3),fmt(x,"coverage",1,3),f"${validity:.1e}$"])+r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
          "Spherical NN units are radians. Torus NN uses the Euclidean norm of wrapped angular differences in radians. The coverage threshold is fixed separately per dataset from training data. Validity means unit-norm error on a sphere and out-of-range fraction on a torus; these different quantities must not be averaged across domains.",
           r"\subsection{Matched MLP Objective Controls}",
           r"Figure~\ref{fig:objective-controls} visualizes the parameterization-matched models from training seed 0.",
           r"\begin{figure}[t]\centering\includegraphics[width=\textwidth]{figures/objective_controls.pdf}",
           r"\caption{Matched constrained MLPs after 2,000 updates: the same fixed test targets and training seed 0, with midpoint and direct-MMD objectives. Each row uses one logarithmic KDE scale and the plotting convention of Figure~\ref{fig:controlled}. Only the first two RNA angles are displayed. These are sample densities, not model likelihoods.}\label{fig:objective-controls}\end{figure}",
           r"Table~\ref{tab:mlp-metrics} supplements the matched-architecture objective comparison with non-MMD metrics. The MMD control directly optimizes the evaluation kernel, so its MMD advantage alone would not establish superiority on all distributional criteria.",
           r"\begin{table}[t]\caption{Matched MLP controls at 2,000 generator updates. Mean and sample SD across three training seeds, after averaging three draws per seed. Smaller symmetric NN and larger coverage are better.}\label{tab:mlp-metrics}\centering\small\begin{tabular}{@{}llrr@{}}\toprule",
           r"Data & Objective & Symmetric NN & Coverage \\",r"\midrule"])
    for data in ["fire","general","rna"]:
        for method,label in [("mgm_mlp","Midpoint"),("mmd_mlp","Direct MMD")]:
            x=stat(s,data,method,2000)
            lines.append(" & ".join([LABEL[data],label,fmt(x,"symmetric_nn",1,3),fmt(x,"coverage",1,3)])+r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",r"\subsection{Numerical Geometry Audits}"])
    audit=OUT/"geometry_audit.json"
    if audit.exists():
        lines.extend([r"Table~\ref{tab:geometry-audit} reports the interpolation residuals.",r"\begin{table}[t]\caption{Maximum numerical residuals over 64 float64 endpoint pairs per geometry. Position and derivative-comparison norms are ambient; velocity reversal uses the manifold metric.}\label{tab:geometry-audit}\centering\small\begin{tabular}{@{}lrrr@{}}\toprule",
             r"Geometry & Position reversal & Velocity reversal & Velocity vs. finite difference \\",r"\midrule"])
        for r in read(audit)["geometric_audits"]:
            vals=[r["geometry"]]+[f"${r[k]:.2e}$" for k in ["max_point_reversal_ambient","max_velocity_reversal_metric","max_velocity_vs_fd_ambient"]]
            lines.append(" & ".join(vals)+r" \\")
        lines.extend([r"\bottomrule\end{tabular}\end{table}","The hyperbolic target itself is approximate. The audited endpoint distribution is not an exhaustive numerical guarantee."])
    calibration=RUNS.parent/"calibration_summary.json"
    if calibration.exists():
        lines.extend([r"\subsection{Empirical Training-Resampling Reference}",
            r"Table~\ref{tab:resampling} calibrates the sample estimator using direct resampling from the empirical training distribution. It is a memorization reference, not a learned generator or a universal lower bound on MMD.",
            r"\begin{table}[t]\caption{Empirical training resampling against the fixed test targets. Mean and Monte Carlo SD over three draws; there are zero training seeds in these reference rows.}\label{tab:resampling}\centering\small\begin{tabular}{@{}lrrr@{}}\toprule",r"Data & MMD$^2\times10^3$ & Symmetric NN & Coverage \\",r"\midrule"])
        for r in read(calibration):
            if r["split"]!="test":continue
            vals=[]
            for key,scale in [("mmd2_biased",1000),("symmetric_nn",1),("coverage",1)]:
                a=r[key];vals.append(f"${a['mean']*scale:.3f}\\pm{a['sd_across_mc_draws']*scale:.3f}$")
            lines.append(" & ".join([LABEL[r["data"]]]+vals)+r" \\")
        lines.append(r"\bottomrule\end{tabular}\end{table}")
    lines.extend([r"\begin{figure}[t]\centering\includegraphics[width=.48\textwidth]{figures/transport_bounds.pdf}",r"\caption{Exact two-point example from Appendix~\ref{app:proofs}. The midpoint discrepancy controls $W_1^2$ but need not control $W_2^2$ by a universal linear constant. Curves are analytic, not learned-model estimates.}\label{fig:bounds}\end{figure}",
          r"\subsection{Historical Additional-Geometry Pilots}",
          "The following historical runs use the legacy metric estimator and one training seed. Its kernel was fitted from generated and target points, and test metrics were monitored during development. We report final steps only, but historical tuning exposure remains. Values are exploratory and must not be pooled with the new fixed-kernel, three-seed tables. Paired-source tasks were adapted to target-only generation by retaining $x_1$; therefore the maze and hyperbolic rows do not reproduce the original source-to-target transport experiment.",
          r"Table~\ref{tab:legacy-extra} records all matching historical final-step runs, including repeated configurations rather than selecting the best observed result.",r"\begin{table}[t]\caption{Historical additional-geometry pilots with their original estimator. Repeated task/method rows are separate configurations; these values are not comparable to the new fixed-kernel table.}\label{tab:legacy-extra}\centering\small\begin{tabular}{@{}llrrrr@{}}\toprule",r"Task & Method & Dimension & Steps & Legacy MMD$^2$ & Legacy sym. NN \\",r"\midrule"])
    legacy=read(ROOT/"outputs/rmgm_summary/final_reproduction_status.json")["runs"]
    for r in legacy:
        if r["family"] not in ("extra_pilot","scaling_pilot"):continue
        vals=[r["data"].replace("_",r"\_"),"R-MGM" if r["method"]=="rmgm" else "Light RFM",str(r["dim"]),str(r["steps"]),f"{r['final_mmd2']:.4f}",f"{r['final_symmetric_nn']:.4f}"]
        lines.append(" & ".join(vals)+r" \\")
    lines.extend([r"\bottomrule\end{tabular}\end{table}",
          "Wrapped-SPD vector dimensions 3, 6, 10 correspond to matrix sizes 2, 3, 4. They are synthetic data, not EEG covariance models. The high-dimensional torus pilots cover dimensions 5, 10 and 20 only; they are not the full author scaling sweep."])
    for data in REAL:
        if (FIG/f"{data}_comparison.pdf").exists():
            lines.extend([f"Figure~\\ref{{fig:extra-{data}}} shows the fixed-seed comparison for {LABEL[data]}.",r"\begin{figure*}[t]\centering",f"\\includegraphics[width=\\textwidth]{{figures/{data}_comparison.pdf}}",
                   f"\\caption{{{LABEL[data]}: fixed test target and training seed 0. Columns are labeled with their generator or flow training budget. Common-bandwidth sample KDEs use the same scale within this figure; these are not likelihood plots.}}\\label{{fig:extra-{data}}}\\end{{figure*}}"])
    mesh=OUT/"legacy_mesh_results.json"
    if mesh.exists():
        grouped=defaultdict(list)
        for r in read(mesh):grouped[(r["data"],r["variant"])].append(r)
        lines.extend([r"\subsection{Final-Checkpoint Exploratory Mesh Results}",
             "Each row is a single seed-0 model trained for 800 iterations. Values average three generated draws against the same 1,024 held-out targets. The plus/minus below is Monte Carlo sample SD, not training-seed variation. All kernels and coverage radii use a fixed training reference. The targets retain historical test exposure; final-checkpoint reevaluation does not erase it.",
             r"Table~\ref{tab:mesh} gives the final-checkpoint surface results.",r"\begin{table}[t]\caption{Exploratory mesh models: means and Monte Carlo SD over three draws from a single trained model per row. These are ambient sample metrics, not intrinsic likelihoods.}\label{tab:mesh}\centering\small\begin{tabular}{@{}llrrrr@{}}\toprule",r"Target & Generator & MMD$^2\times10^3$ & Ambient sym. NN & Coverage & Mean feasibility \\",r"\midrule"])
        names={"projection":"Projected mixture","sphere_map":"Soft sphere map","sphere_map_gmm":"Sphere-map mixture"}
        for (d,v),rr in sorted(grouped.items()):
            values=[]
            for k,scale in [("mmd2_biased",1000),("symmetric_nn",1),("coverage",1)]:
                a=np.array([x[k] for x in rr])*scale
                values.append(f"${a.mean():.3f}\\pm{a.std(ddof=1):.3f}$")
            error=np.mean([x["constraint_mean"] for x in rr])
            lines.append(" & ".join([d,names[v]]+values+[f"${error:.1e}$"])+r" \\")
        lines.extend([r"\bottomrule\end{tabular}\end{table}","Surface feasibility measures mean distance to a second closest-point projection. It is not an intrinsic density metric or a certificate of bijective parameterization. The soft map interpolates radii from mesh-vertex directions and then projects; it is not the separate ray-intersection utility."])
    for data in ["bunny10","bunny50","bunny100","spot10","spot50","spot100"]:
        if (FIG/f"{data}_surface.pdf").exists():
            lines.extend([f"Figure~\\ref{{fig:surface-{data}}} gives the full parameterization comparison for {data}.",r"\begin{figure*}[t]\centering",f"\\includegraphics[width=\\textwidth]{{figures/{data}_surface.pdf}}",
               f"\\caption{{{data}: one fixed seed-0 final checkpoint per variant, common cameras and 1,024 points. The loss is a local projection surrogate, not the spectral-path construction.}}\\label{{fig:surface-{data}}}\\end{{figure*}}"])
    write("additional_results.tex",lines)


def main():
    global RUNS,DATA
    parser=argparse.ArgumentParser();parser.add_argument("--require-complete",action="store_true")
    parser.add_argument("--run-root",choices=["controlled","controlled_independent"],default="controlled")
    parser.add_argument("--skip-sample-plots",action="store_true")
    args=parser.parse_args();RUNS=OUT/args.run_root
    if (RUNS/"controlled").exists():RUNS=RUNS/"controlled"
    DATA=RUNS.parent/"datasets" if (RUNS.parent/"datasets").exists() else OUT/"datasets"
    if args.run_root=="controlled_independent":
        ready=read(RUNS.parent/"FINAL_TABLES_READY.json")
        assert ready.get("sphere_metric_version")=="sphere_normalized_float64_atan2_v2",ready
        assert ready.get("all_primary_jobs_completed") and ready.get("rng_correction_verified") and ready.get("metric_provenance_verified"),ready
    FIG.mkdir(exist_ok=True);GEN.mkdir(exist_ok=True)
    s=collect()
    if args.require_complete:
        for d in REAL:
            for m,t,n in [("mgm_gmm",0,1),("mgm_gmm",2000,1),("rfm",2000,128),("rfm",10000,128)]:
                assert stat(s,d,m,t,n) and stat(s,d,m,t,n)["n_seeds"]==3,(d,m,t,n)
        for d in ["fire","general","rna"]:
            for m in ["mgm_mlp","mmd_mlp"]:
                assert stat(s,d,m,2000) and stat(s,d,m,2000)["n_seeds"]==3,(d,m)
    main_table(s);ablations(s);protocol(s)
    if not args.skip_sample_plots:plots();objective_plots();mesh_graphics()
    stability();noise_diagnostics();extra(s)
    serial=[dict(data=k[0],method=k[1],step=k[2],nfe=k[3],split=k[4],**v) for k,v in sorted(s.items())]
    (GEN/"presentation_statistics.json").write_text(json.dumps(serial,indent=2),encoding="utf-8")
    print(json.dumps({"groups":len(s),"complete_primary_datasets":sum(bool(stat(s,d,'rfm',10000,128)) for d in REAL)}))


if __name__=="__main__":main()
