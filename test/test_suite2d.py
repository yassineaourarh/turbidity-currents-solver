# ============================================================================
#  test_suite2d.py - Banc d'essai 2-D pour le systeme de Ripa
# ============================================================================
#  Ce fichier orchestre les simulations 2-D du modele de Ripa :
#    1) Transport passif d'un paquet gaussien de sediments
#    2) Rupture de barrage rectangulaire  (Touma 2015, section 3.2.1)
#    3) Rupture de barrage circulaire     (Touma 2015, section 3.2.2)
#    4) Perturbation gaussienne sur lit irregulier (section 3.2.3)
#
#  Les figures de reference (Fig. 11 et Fig. 12 de l'article) sont
#  reproduites avec les fonctions fig11_rectangular_profiles() et
#  fig12_circular_surface().
#
#  Lancer tous les tests :  python test_suite2d.py
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from plotting import imshow_fixed, surface_fixed

from config   import g, h_min, CFL, Tmax, Lx
from solver2d import step_2d
from initial_conditions import (
    rect_dambreak,
    circular_dambreak,
    steady_gauss_perturb,
)

# ============================================================================
#  Parametres par defaut pour le cas << transport gaussien >>
# ============================================================================
Nx = 200              # Nombre de mailles en x
Ny = 100              # Nombre de mailles en y
Ly = 0.5 * Lx         # Longueur du domaine en y (moitie de Lx)
dx = Lx / Nx           # Pas d'espace en x
dy = Ly / Ny           # Pas d'espace en y
dt_snap = 5.0          # Intervalle entre deux snapshots (s)


# ============================================================================
#  Conditions initiales : transport gaussien de sediments
# ============================================================================

def initial_conditions_gaussian():
    """
    Construit les conditions initiales pour le test de transport gaussien.

    Ecoulement uniforme : h = 1 m, u = 0.5 m/s, v = 0.
    Un paquet gaussien de concentration theta est place au tiers gauche
    du domaine. Le courant uniforme le transporte vers la droite.

    Retourne h, qx, qy, s, b  (tableaux 2-D).
    """
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y)

    h  = np.ones((Ny, Nx))           # hauteur uniforme = 1 m
    u0 = 0.5                          # vitesse uniforme en x
    qx = h * u0                      # debit en x = h * u
    qy = np.zeros_like(qx)           # pas de courant en y

    # Paquet gaussien de concentration theta
    x0, y0 = 0.3 * Lx, 0.5 * Ly      # centre du paquet (30% de Lx, milieu en y)
    sigma  = 0.05 * Lx               # largeur du paquet
    # s = h * theta, avec theta = exp(-r^2 / 2*sigma^2) * 0.05
    s = h * np.exp(-((X-x0)**2 + (Y-y0)**2)/(2*sigma**2)) * 0.05

    b = np.zeros_like(h)              # fond plat
    return h, qx, qy, s, b


# ============================================================================
#  Utilitaires
# ============================================================================

def _ensure_dir(dirname: str):
    """Cree un dossier s'il n'existe pas deja."""
    os.makedirs(dirname, exist_ok=True)


def save_figure(field: np.ndarray, t: float, name: str, vmin=None, vmax=None):
    """
    Sauvegarde une carte 2-D (imshow) dans le dossier outputs2d/.

    Parametres
    ----------
    field : np.ndarray 2-D  - champ a afficher
    t : float               - instant courant (pour le titre)
    name : str              - nom du champ (pour le titre et le fichier)
    vmin, vmax : float      - bornes de l'echelle couleur (optionnel)
    """
    _ensure_dir("outputs2d")
    plt.figure(figsize=(6, 3))
    im = plt.imshow(field, origin="lower", extent=[0, Lx, 0, Ly], aspect="auto",
                    vmin=vmin, vmax=vmax)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f"{name} -- t = {t:.1f} s")
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(f"outputs2d/{name.replace(' ','_').lower()}_{int(t):04d}.png")
    plt.close()


# ============================================================================
#  TEST 1 : Transport passif d'un paquet gaussien
# ============================================================================

def run_gaussian_transport():
    """
    Simule le transport passif d'un paquet gaussien de sediments
    par un courant uniforme (u = 0.5 m/s, v = 0).

    Le paquet theta gaussien doit se deplacer vers la droite sans
    deformation (le schema est d'ordre 1, donc il y aura de la diffusion).
    Des snapshots de h et theta sont sauvegardes dans outputs2d/.
    """
    print("==> Test 2D : Transport passif d'un paquet gaussien de sediments")

    h, qx, qy, s, b = initial_conditions_gaussian()
    t = 0.0
    next_snap = 0.0
    s_max0 = s.max()    # valeur max initiale de s (pour l'echelle couleur)

    while t < Tmax:
        # Calcul du pas de temps CFL 2-D
        h_safe = np.maximum(h, h_min)
        u = qx / h_safe                            # vitesse en x
        v = qy / h_safe                            # vitesse en y
        theta = s / h_safe                          # temperature theta
        celer = np.sqrt(g * theta * h_safe)         # celerite
        max_speed = np.max(np.sqrt(u**2 + v**2) + celer)  # vitesse d'onde max
        dt = CFL * min(dx, dy) / max_speed          # condition CFL 2-D
        dt = min(dt, Tmax - t)

        # Avance d'un pas de temps
        h, qx, qy, s = step_2d(h, qx, qy, s, b, dx, dy, dt)
        t += dt

        # Sauvegarde periodique des figures
        if t >= next_snap or abs(t - Tmax) < 1e-9:
            print(f"  . snapshot t = {t:.1f} s")
            save_figure(h + b, t, "Surface libre eta (m)")
            save_figure(s / np.maximum(h, h_min), t, "Concentration theta",
                        vmin=0.0, vmax=s_max0/h.min())
            next_snap += dt_snap

    print("[INFO] Figures enregistrees sous outputs2d/.")


# ============================================================================
#  TEST 2 : Runner generique pour les cas-tests 2-D
# ============================================================================

def run_case(name, init_func, L=1.0, Nx=201, Ny=201,
             Tfin=0.2, dump_every=0.05):
    """
    Runner generique pour les tests 2-D sur un domaine [-L, L]^2.

    Parametres
    ----------
    name : str         - nom du test (titre + base des fichiers)
    init_func          - fonction qui retourne (h, qx, qy, s, Z) a partir de (x, y)
    L : float          - demi-taille du domaine carre
    Nx, Ny : int       - nombre de mailles en x et y
    Tfin : float       - duree de la simulation (s)
    dump_every : float - intervalle entre deux snapshots (s)
    """
    # Maillage
    x = np.linspace(-L, L, Nx)
    y = np.linspace(-L, L, Ny)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Conditions initiales
    h, qx, qy, s, Z = init_func(x, y)
    theta = s / np.maximum(h, h_min)

    # Boucle en temps
    t = 0.0
    next_dump = 0.0
    snap_id = 0
    os.makedirs("Outputs", exist_ok=True)

    while t < Tfin - 1e-12:
        # CFL 2-D
        c = np.sqrt(g * theta * h)                      # celerite
        umax = np.max(np.abs(qx / np.maximum(h, h_min)) + c)  # vitesse max en x
        vmax = np.max(np.abs(qy / np.maximum(h, h_min)) + c)  # vitesse max en y
        dt = CFL * min(dx / max(umax, 1e-12), dy / max(vmax, 1e-12))
        if t + dt > Tfin:
            dt = Tfin - t

        # Avance
        h, qx, qy, s = step_2d(h, qx, qy, s, Z, dx, dy, dt)
        theta = s / np.maximum(h, h_min)       # recalcul de theta apres le pas
        t += dt

        # Sauvegarde periodique
        if t >= next_dump - 1e-12 or t >= Tfin - 1e-12:
            plt.figure(figsize=(4, 3))
            plt.imshow(h, extent=[-L, L, -L, L], origin="lower", cmap="viridis")
            plt.colorbar(label="h (m)")
            plt.title(f"{name} -- t = {t:0.2f} s")
            plt.tight_layout()
            plt.savefig(f"Outputs/{name}_{snap_id:03d}.png", dpi=120)
            plt.close()
            next_dump += dump_every
            snap_id += 1

    print(f"{name}: termine jusqu'a t = {Tfin}s, {snap_id} images ecrites.")


# ============================================================================
#  FIG. 11 : Profils 1-D (coupe au centre) de la rupture rectangulaire
# ============================================================================

def fig11_rectangular_profiles():
    """
    Reproduit la Figure 11 de l'article (Touma & Klingenberg 2015).

    On calcule la rupture de barrage rectangulaire a t = 0.2 s
    pour 3 resolutions differentes (50^2, 100^2, 200^2),
    puis on trace la coupe en y = 0 (profil de h au centre).

    Le graphe montre la convergence du schema quand on raffine le maillage.
    """
    L = 1.0
    Tfin = 0.2
    Nlist = [50, 100, 200]          # trois resolutions a comparer
    profiles = []

    for Nx in Nlist:
        Ny = Nx
        # Maillage et pas
        x = np.linspace(-L, L, Nx)
        y = np.linspace(-L, L, Ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        # Conditions initiales
        h, qx, qy, s, Z = rect_dambreak(x, y)
        print(f"[IC] Nx={Nx}  centre={h[Ny//2, Nx//2]:.3f}  coin={h[0,0]:.3f}")
        theta = s / np.maximum(h, h_min)

        # Integration en temps jusqu'a Tfin
        t = 0.0
        while t < Tfin - 1e-12:
            # Calcul du pas de temps
            c  = np.sqrt(g * theta * h)          # celerite
            u  = qx / np.maximum(h, h_min)       # vitesse x
            v  = qy / np.maximum(h, h_min)       # vitesse y
            ws = np.sqrt(u*u + v*v) + c           # vitesse d'onde
            den = np.max(ws)
            dt  = Tfin - t if den < 1e-14 else min(CFL * min(dx, dy) / den, Tfin - t)

            # Avance d'un pas
            h, qx, qy, s = step_2d(h, qx, qy, s, Z, dx, dy, dt)
            theta = s / np.maximum(h, h_min)
            t += dt

        # Extraction de la coupe au centre (ligne y = 0)
        profiles.append((x.copy(), h[Ny//2, :]))

    # Trace des 3 profils superposes
    plt.figure(figsize=(5, 4))
    for (xv, prof), N, mk in zip(profiles, Nlist, ["+", "x", "--"]):
        plt.plot(xv, prof, mk, label=f"{N}^2")
    plt.xlim(-L, L); plt.ylim(0.8, 2.05)
    plt.xlabel("x (m)"); plt.ylabel("h (m)")
    plt.title("Rectangular dam-break -- t = 0.2 s")
    plt.legend(); plt.tight_layout()
    plt.savefig("Fig11_rectangular_profiles.png", dpi=150); plt.close()
    print("[INFO] Fig. 11 exportee.")


# ============================================================================
#  FIG. 12 : Vue 3-D de la rupture de barrage circulaire
# ============================================================================

def fig12_circular_surface(N=100, t_end=0.2, CFL=0.45, L=1.0):
    """
    Reproduit la Figure 12 de l'article (Touma & Klingenberg 2015).

    Rupture de barrage circulaire sur le domaine [-1, 1]^2.
    A l'interieur du cercle r <= 0.25 : h = 2 m, theta = 1
    A l'exterieur               : h = 1 m, theta = 1.5

    La figure montre la surface 3-D de h a t = 0.2 s, avec echelle
    couleur fixee entre 0.8 et 2.0 m (comme dans l'article).

    Retourne x, y, h (pour analyse ulterieure si besoin).
    """
    # Maillage
    x = np.linspace(-L, L, N)
    y = np.linspace(-L, L, N)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    X, Y = np.meshgrid(x, y, indexing="xy")

    # Conditions initiales (Touma 2015, section 3.2.2)
    r2 = X**2 + Y**2
    h = np.where(r2 <= 0.25, 2.0, 1.0)         # h = 2 dans le cercle
    theta = np.where(r2 <= 0.25, 1.0, 1.5)     # theta = 1 dans le cercle
    u = np.zeros_like(h)                         # repos
    v = np.zeros_like(h)
    qx = h * u
    qy = h * v
    s = h * theta                                # s = h * theta
    Z = np.zeros_like(h)                         # fond plat

    # Boucle en temps
    t = 0.0
    tiny = 1e-14
    while t < t_end - 1e-14:
        # Vitesses et celerites
        u = np.where(h > tiny, qx / h, 0.0)
        v = np.where(h > tiny, qy / h, 0.0)
        c = np.sqrt(g * np.maximum(h, tiny) * theta)   # celerite Ripa
        a = max(np.max(np.abs(u) + c), np.max(np.abs(v) + c))
        dt = CFL * min(dx, dy) / a
        if t + dt > t_end:
            dt = t_end - t

        # Avance d'un pas
        h, qx, qy, s = step_2d(h, qx, qy, s, Z, dx, dy, dt)
        theta = np.where(h > tiny, s / h, theta)   # recalcul de theta
        t += dt

    # Trace de la surface 3-D (avec echelle fixee comme dans l'article)
    fig = plt.figure(figsize=(7.5, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X, Y, h, linewidth=0, antialiased=True, cmap="viridis",
        vmin=0.8, vmax=2.0
    )
    cb = fig.colorbar(surf, ax=ax, shrink=0.8, pad=0.12)
    cb.set_label("h (m)")
    ax.set_xlim(-L, L); ax.set_ylim(-L, L); ax.set_zlim(0.8, 2.0)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("h (m)")
    ax.view_init(elev=25, azim=-35)

    hmin = float(h.min()); hmax = float(h.max())
    ax.set_title(f"Fig. 12 -- Circular dam-break ; t = {t_end} s\n"
                 f"hmin={hmin:.3f}, hmax={hmax:.3f}")
    plt.tight_layout()
    plt.show()

    return x, y, h


# ============================================================================
#  POINT D'ENTREE PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    # ---- Fig. 11 : Profils 1-D de la rupture rectangulaire ----
    fig11_rectangular_profiles()

    # ---- Fig. 12 : Surface 3-D de la rupture circulaire ----
    fig12_circular_surface()

    # ---- Transport gaussien ----
    run_gaussian_transport()

    # ---- Tests generiques 2-D ----
    # Rupture de barrage rectangulaire (domaine [-1,1]^2, t_fin = 0.2 s)
    run_case("Rect_dambreak", rect_dambreak, L=1.0, Tfin=0.2)

    # Rupture de barrage circulaire (domaine [-1,1]^2, t_fin = 0.2 s)
    run_case("Circular_dambreak", circular_dambreak, L=1.0, Tfin=0.2)

    # Perturbation gaussienne sur lit irregulier (domaine [-2,2]^2)
    run_case("Steady_perturb", steady_gauss_perturb,
             L=2.0, Nx=401, Ny=401, Tfin=1.0, dump_every=0.2)
