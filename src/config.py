# ============================================================================
#  config.py - Parametres globaux de la simulation
# ============================================================================
#  Ce fichier centralise TOUS les parametres partages par les differents
#  solveurs et fichiers du projet :
#    - Dimensions du domaine et maillage spatial
#    - Constantes physiques (gravite, seuil mouille/sec)
#    - Contraintes temporelles (CFL, duree)
#    - Parametres de frottement (loi de Manning)
#    - Fonctions utilitaires (limiteur minmod)
#
#  Le systeme resolu est le **modele de Ripa** (Saint-Venant avec
#  temperature/densite variable theta) :
#    dh/dt  + d(q)/dx                            = 0
#    dq/dt  + d(q^2/h + 0.5 g theta h^2)/dx      = -g theta h dZ/dx  (+frottement)
#    ds/dt  + d(q theta)/dx                       = 0
#  ou  s = h*theta  est la variable transportee, theta = temperature potentielle.
# ============================================================================

import numpy as np

# ===================== Domaine spatial & maillage ========================
Lx = 25.0           # Longueur du domaine en x (m)
Nx = 501            # Nombre de points de maillage en x (=> dx ~ 0.05 m)
Ny = Nx             # Nombre de points en y (utilise pour les cas 2-D)
dx = Lx / (Nx - 1)  # Pas d'espace (m) - maillage regulier

# ===================== Constantes physiques ==============================
g     = 1.0          # Acceleration de la gravite (m/s^2)
                     # Souvent normalisee a 1 dans les cas-tests academiques
h_min = 1.0e-6       # Seuil minimal de hauteur d'eau : en dessous la cellule
                     # est consideree << seche >> (evite les divisions par zero)

# ===================== Parametres temporels ==============================
CFL   = 0.4          # Nombre de Courant-Friedrichs-Lewy : controle la stabilite
                     # du schema explicite (dt = CFL * dx / max(|u| + c))
Tmax  = 20.0         # Duree totale de la simulation (s)

# ===================== Frottement de Manning =============================
# La loi de Manning modelise la resistance exercee par le fond :
#   dq/dt = -g n^2 |q| q / h^(7/3)
# Elle est integree analytiquement (cf. Delestre 2010, chap. 2) pour
# garantir la stabilite meme avec de petites hauteurs d'eau.
FRICTION_LAW = "manning"   # "none" -> pas de frottement ; "manning" -> active
n_manning    = 0.03         # Coefficient de rugosite de Manning (s/m^(1/3))
                            # Peut etre surcharge dans les cas-tests

# ===================== Divers ============================================
theta0 = 0.01        # Valeur par defaut de theta (temperature potentielle)


def minmod(a, b):
    """
    Limiteur minmod vectoriel.

    Le limiteur minmod(a, b) vaut :
      - 0  si a et b sont de signes opposes (discontinuite detectee),
      - le plus petit en valeur absolue de a et b sinon (pente limitee).

    Utilise dans la reconstruction MUSCL pour limiter les pentes
    et eviter les oscillations parasites (propriete TVD du schema).

    Parametres
    ----------
    a, b : np.ndarray  - pentes gauche et droite (meme forme)

    Retourne
    --------
    np.ndarray - pente limitee element par element
    """
    res   = np.zeros_like(a)
    # On ne garde une pente que si les deux gradients vont dans le meme sens
    mask  = (a * b) > 0
    # Parmi les deux pentes de meme signe, on choisit la plus petite (valeur absolue)
    res[mask] = np.where(np.abs(a[mask]) < np.abs(b[mask]), a[mask], b[mask])
    return res
