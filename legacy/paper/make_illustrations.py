"""Generate geometric method illustrations; all curves use actual sphere geodesics."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, RED, TEAL, GREY = "#2766a2", "#bb463d", "#13836d", "#d8dce0"


def sphere(ax):
    u, v = np.meshgrid(np.linspace(0, 2*np.pi, 65), np.linspace(0, np.pi, 33))
    ax.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v),
                    color=GREY, alpha=.18, linewidth=0, shade=False, rasterized=True)
    ax.set(xlim=(-1.3,1.3), ylim=(-1.3,1.3), zlim=(-1.15,1.15))
    ax.set_box_aspect((1,1,1))
    ax.view_init(25,-65)
    ax.set_axis_off()


def arc(a,b,t):
    angle=np.arccos(np.clip(a@b,-1,1))
    return (np.sin((1-t)*angle)[...,None]*a+np.sin(t*angle)[...,None]*b)/np.sin(angle)


def main():
    a=np.array([-.9,-.25,.36]); a/=np.linalg.norm(a)
    b=np.array([.85,-.12,.51]); b/=np.linalg.norm(b)
    t=np.linspace(0,1,150)
    path=arc(a,b,t)
    fig=plt.figure(figsize=(7.0,2.65))
    for idx,reverse in enumerate((False,True)):
        ax=fig.add_subplot(1,3,idx+1,projection="3d")
        sphere(ax)
        ax.plot(*path.T,color=BLUE,lw=2.4)
        ax.scatter(*a,s=30,color=BLUE,depthshade=False)
        ax.scatter(*b,s=30,color=RED,depthshade=False)
        ax.text(*(a*1.16),"$x_0$",color=BLUE)
        ax.text(*(b*1.16),"$x_1$",color=RED)
        for q in (.25,.75):
            x=arc(a,b,np.array(q))
            eps=1e-4
            vel=(arc(a,b,np.array(q+eps))-arc(a,b,np.array(q-eps)))/(2*eps)
            if reverse: vel=-vel
            ax.scatter(*x,s=23,color=TEAL,depthshade=False)
            ax.quiver(*x,*(vel*.17),color=TEAL,linewidth=1.7,arrow_length_ratio=.3)
        ax.text2D(.5,.93,"$(x_0,x_1,B)$" if not reverse else "$(x_1,x_0,1-B)$",
                  transform=ax.transAxes,ha="center")
        ax.text2D(.5,.02,"Same observation" if not reverse else "Opposite tangent target",
                  transform=ax.transAxes,ha="center",fontsize=8)
    ax=fig.add_subplot(1,3,3)
    ax.axis("off")
    ax.text(.06,.84,"Hidden orientation",weight="bold",fontsize=10)
    ax.text(.06,.61,"$\\widetilde x=\\gamma_{x_0,x_1}(s)$\n$s\\in\\{t,1-t\\}$",linespacing=1.8)
    ax.text(.06,.29,"$a=\\partial_s\\gamma_{x_0,x_1}(s)$\n$f_\\psi(t,\\widetilde x)\\in T_{\\widetilde x}\\mathcal{M}$",linespacing=1.8)
    fig.subplots_adjust(left=0,right=1,bottom=.02,top=.98,wspace=.03)
    fig.savefig(OUT/"method_symmetry.pdf",bbox_inches="tight",dpi=260)
    fig.savefig(OUT/"method_symmetry.png",bbox_inches="tight",dpi=260)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(7.0,2.25))
    for ax in axes: ax.set_aspect("equal");ax.axis("off")
    th=np.linspace(.05,np.pi-.05,200)
    for ax in axes: ax.plot(np.cos(th),np.sin(th),color=GREY,lw=13,solid_capstyle="round")
    ax=axes[0]
    for theta,c,lab in [(2.7,BLUE,"$G_\\theta(z)$"),(.4,RED,"$x\\sim\\nu$")]:
        x=np.array([np.cos(theta),np.sin(theta)])
        ax.scatter(*x,color=c,s=45,zorder=5);ax.text(x[0],x[1]-.19,lab,ha="center",fontsize=9)
    q=np.linspace(.4,2.7,120)
    ax.plot(np.cos(q),np.sin(q),color=TEAL,lw=2)
    ax.text(0,1.26,"Geometric midpoint training",ha="center",fontsize=10)
    ax.text(0,-.05,"Tangent regression + variational generator update",ha="center",fontsize=7.7)
    ax=axes[1]
    rng=np.random.default_rng(17)
    theta=np.r_[rng.normal(.5,.09,90),rng.normal(2.45,.12,90)]
    ax.scatter(np.cos(theta),np.sin(theta),s=9,color=TEAL,alpha=.65)
    ax.text(0,1.26,"One-step sampling",ha="center",fontsize=10)
    ax.text(0,.35,"$z\\longmapsto G_\\theta(z)\\in\\mathcal{M}$",ha="center",fontsize=12)
    ax.text(0,-.05,"No learned flow integration at inference",ha="center",fontsize=8)
    for ax in axes: ax.set_xlim(-1.3,1.3);ax.set_ylim(-.22,1.42)
    fig.subplots_adjust(left=.02,right=.98,bottom=.02,top=.98,wspace=.20)
    fig.savefig(OUT/"teaser.pdf",bbox_inches="tight")
    fig.savefig(OUT/"teaser.png",bbox_inches="tight",dpi=300)
    plt.close(fig)

    a=np.geomspace(.001,.99,150)
    divergence=2*a*a/(1+a*a)
    action=(1+a*a)/2
    assert np.all(a*a <= divergence+1e-14)
    assert np.all(divergence <= action+1e-14)
    fig,ax=plt.subplots(figsize=(3.25,2.25))
    ax.loglog(a,a*a,label="$W_1^2$",color=BLUE,lw=1.7)
    ax.loglog(a,divergence,label="$\\mathcal{D}$",color=TEAL,lw=1.7)
    ax.loglog(a,a,label="$W_2^2$",color=RED,lw=1.7,ls="--")
    ax.set(xlabel="Mass imbalance $a$",ylabel="Discrepancy ($L=1$)")
    ax.legend(frameon=False,fontsize=8,loc="lower right")
    ax.grid(alpha=.16,which="major")
    fig.tight_layout()
    fig.savefig(OUT/"transport_bounds.pdf",bbox_inches="tight")
    fig.savefig(OUT/"transport_bounds.png",bbox_inches="tight",dpi=300)
    plt.close(fig)


if __name__=="__main__": main()
