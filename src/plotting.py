# ============================================================================
#  plotting.py - Affichage et sauvegarde des resultats de simulation
# ============================================================================
#  Ce module fournit les fonctions de visualisation pour :
#    1) Les simulations 1-D : tracee de eta (surface libre), u (vitesse),
#       theta (temperature) et P (pression optionnelle) en sous-graphes.
#    2) Les simulations 2-D : cartes couleur (imshow) et surfaces 3D.
#
#  Les figures sont soit affichees a l'ecran, soit sauvegardees en PNG
#  dans un sous-dossier outputs/<nom_du_test>/.
# ============================================================================

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D   # necessaire pour les vues 3D


# ============================================================================
#  Outils internes
# ============================================================================

def _safe(name: str) -> str:
    """
    Nettoie un titre pour l'utiliser comme nom de dossier/fichier.
    Supprime les caracteres speciaux (deux-points, tirets longs)
    et remplace les espaces par des underscores.
    """
    return (name
            .replace(":", "")
            .replace("--", "")
            .replace("---", "")
            .replace(" ", "_")
            .strip("_"))


# ============================================================================
#  Fonction principale : tracee 1-D
# ============================================================================

def plot_fields(x: np.ndarray,
                eta_snap: list,
                u_snap:  list,
                theta_snap: list,
                t_list:   list,
                P_snap:   list | None = None,
                *,
                Z: np.ndarray | None = None,
                title: str = "Test",
                save: bool = True,
                dpi: int = 120):
    """
    Affiche (ou sauvegarde) une serie de champs 1-D sous forme de graphiques.

    Pour chaque instant t dans t_list, on trace 3 ou 4 sous-graphes :
      1) eta(x) = surface libre (h + Z)  --  avec le fond Z en tirets si fourni
      2) u(x)   = vitesse
      3) theta(x) = temperature potentielle
      4) P(x)   = pression (optionnel, si P_snap est fourni)

    Parametres
    ----------
    x           : np.ndarray - abscisses (1-D, taille Nx)
    eta_snap    : list       - listes de eta(x) a chaque instant
    u_snap      : list       - listes de u(x) a chaque instant
    theta_snap  : list       - listes de theta(x) a chaque instant
    t_list      : list       - instants correspondants (s)
    P_snap      : list|None  - listes de P(x), optionnel
    Z           : np.ndarray - bathymetrie (tracee en tirets noirs si fournie)
    title       : str        - titre global et nom du sous-dossier de sortie
    save        : bool       - True = sauvegarde PNG ; False = affichage ecran
    dpi         : int        - resolution des images sauvegardees
    """
    nframes = len(t_list)
    # Verification : toutes les listes doivent avoir la meme longueur
    assert len(eta_snap) == nframes == len(u_snap) == len(theta_snap), \
        "Les listes de snapshots n'ont pas la meme longueur"

    # Creation du dossier de sortie si sauvegarde demandee
    tag = _safe(title)
    out_dir = os.path.join("outputs", tag)
    if save:
        os.makedirs(out_dir, exist_ok=True)

    # Nombre de sous-graphes : 3 par defaut, 4 si pression fournie
    nrows = 4 if P_snap is not None else 3

    for k in range(nframes):
        fig, (ax_eta, ax_u, ax_theta, *rest) = plt.subplots(
            nrows, 1, figsize=(10, 10), sharex=True
        )

        # ---------- Sous-graphe 1 : Surface libre eta ----------
        ax_eta.plot(x, eta_snap[k], label="eta")
        if Z is not None:
            # Trace la bathymetrie en tirets noirs pour reference
            ax_eta.plot(x, Z, "--k", lw=0.8, label="Z (fond)")
            ax_eta.legend(loc="upper right")
        ax_eta.set_ylabel("eta (m)")

        # ---------- Sous-graphe 2 : Vitesse u ----------
        ax_u.plot(x, u_snap[k])
        ax_u.set_ylabel("u (m/s)")

        # ---------- Sous-graphe 3 : Temperature theta ----------
        ax_theta.plot(x, theta_snap[k])
        ax_theta.set_ylabel("theta")

        # ---------- Sous-graphe 4 : Pression (si fournie) ----------
        if P_snap is not None:
            ax_P = rest[0]
            ax_P.plot(x, P_snap[k])
            ax_P.set_ylabel("P")

        # ---------- Habillage commun ----------
        # L'axe x est sur le dernier sous-graphe
        (rest[0] if rest else ax_theta).set_xlabel("x (m)")
        fig.suptitle(f"{title} -- t = {t_list[k]:.2f} s")
        fig.tight_layout()

        # ---------- Sauvegarde ou affichage ----------
        if save:
            fname = os.path.join(out_dir, f"frame_{k:04d}.png")
            fig.savefig(fname, dpi=dpi)
            print(f"[INFO] {fname} sauvegarde")
            plt.close(fig)
        else:
            plt.show()


# ============================================================================
#  Visualisation 2-D : carte couleur (imshow) avec echelle fixee
# ============================================================================

def imshow_fixed(field, extent, title, fname, vmin, vmax, cmap="viridis"):
    """
    Trace une carte 2-D en couleurs avec echelle fixee et sauvegarde en PNG.

    Parametres
    ----------
    field   : np.ndarray 2D - champ a afficher
    extent  : list [xmin, xmax, ymin, ymax] - etendue spatiale
    title   : str  - titre de la figure
    fname   : str  - chemin du fichier PNG de sortie
    vmin    : float - valeur minimale de l'echelle couleur
    vmax    : float - valeur maximale de l'echelle couleur
    cmap    : str  - palette de couleurs (par defaut "viridis")
    """
    plt.figure(figsize=(5, 4))
    im = plt.imshow(field, origin="lower", extent=extent,
                    vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
    plt.colorbar(im, fraction=.046, pad=.04, label="h (m)")
    plt.title(title)
    plt.xlabel("x (m)"); plt.ylabel("y (m)")
    plt.tight_layout(); plt.savefig(fname, dpi=150); plt.close()


# ============================================================================
#  Visualisation 2-D : surface 3D avec echelle fixee
# ============================================================================

def surface_fixed(X, Y, H, title, fname, vmin, vmax, cmap="viridis"):
    """
    Trace une vue 3D (surface) avec echelle z et couleur fixees.

    Parametres
    ----------
    X, Y : np.ndarray 2D - grilles de coordonnees (meshgrid)
    H    : np.ndarray 2D - champ a afficher en hauteur
    title : str  - titre de la figure
    fname : str  - chemin du fichier PNG de sortie
    vmin, vmax : float - bornes de l'echelle z et couleur
    cmap  : str  - palette de couleurs
    """
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X, Y, H, cmap=cmap,
                           vmin=vmin, vmax=vmax, rstride=1, cstride=1,
                           linewidth=0)
    fig.colorbar(surf, shrink=.55, aspect=14, label="h (m)")
    ax.set_zlim(vmin, vmax)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("h (m)")
    ax.set_title(title)
    plt.tight_layout(); fig.savefig(fname, dpi=150); plt.close()
