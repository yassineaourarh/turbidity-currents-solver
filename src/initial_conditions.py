# ============================================================================
#  initial_conditions.py - Conditions initiales pour les cas-tests 1D et 2D
# ============================================================================
#  Ce module rassemble toutes les fonctions qui definissent les conditions
#  initiales (CI) du systeme de Ripa : hauteur d'eau h, debit q, variable
#  transportee s = h*theta, et bathymetrie Z (quand applicable).
#
#  Chaque fonction retourne les champs necessaires au lancement d'une
#  simulation. Les cas-tests couvrent :
#    - Equilibres stationnaires (lac au repos, equilibre Ripa)
#    - Ruptures de barrage (mouille/mouille, mouille/sec, sur bosse)
#    - Ecoulements transcritiques (avec/sans choc)
#    - Cas 2D (rupture rectangulaire, circulaire, perturbation gaussienne)
# ============================================================================

import numpy as np
from config import h_min, g


# ============================================================================
# 1) EQUILIBRE STATIONNAIRE  u = 0, h^2 * theta = C  (eq. 5 de l'article)
# ============================================================================

def ripa_equilibrium(x, *, a=0.0, b=1.0, C=1.0):
    """
    Construit un etat d'equilibre exact du systeme de Ripa.

    L'equilibre verifie u = 0 et h^2 * theta = C (constante).
    On choisit theta(x) = a*x + b  (profil lineaire, doit rester > 0),
    puis  h(x) = sqrt(C / theta(x)).

    Parametres
    ----------
    x : np.ndarray - positions des cellules
    a : float      - pente du profil de theta (0 = theta constant)
    b : float      - valeur de theta en x = 0
    C : float      - constante de l'equilibre h^2 * theta

    Retourne
    --------
    h, q, s : np.ndarray - hauteur, debit (=0), et s = h * theta
    """
    theta = a * x + b
    assert np.all(theta > 0.0), "theta doit rester positif partout"
    h = np.sqrt(C / theta)         # h tel que h^2 * theta = C
    q = np.zeros_like(x)           # repos : u = 0
    s = h * theta                  # s = h * theta
    return h, q, s


def perturb_height(h, q, s, *, eps=1e-3, k=1):
    """
    Ajoute une petite perturbation sinusoidale a la hauteur d'eau.

    La perturbation est de la forme :  delta_h = eps * sin(2*pi*k*x/L).
    La variable s est recalculee pour conserver theta coherent :
      s_perturbe = h_perturbe * theta_original.

    Parametres
    ----------
    h, q, s : np.ndarray - etat initial a perturber
    eps : float  - amplitude de la perturbation (m)
    k : int      - nombre d'onde de la perturbation

    Retourne
    --------
    h_pert, q, s_pert : np.ndarray - etat perturbe
    """
    N = h.size
    x = np.linspace(0.0, 1.0, N, endpoint=False)
    delta = eps * np.sin(2 * np.pi * k * x)
    h_pert = h + delta
    # Recuperer theta original pour recalculer s
    theta  = np.where(h > h_min, s / h, 0.0)
    s_pert = h_pert * theta          # garantit h^2*theta ~ C a O(eps)
    return h_pert, q.copy(), s_pert


# ============================================================================
# 2) BATHYMETRIE PLATE
# ============================================================================

def flat_bathymetry(x):
    """Retourne une bathymetrie plate Z(x) = 0 partout."""
    return np.zeros_like(x)


# ============================================================================
# 3) RUPTURE DE BARRAGE - MOUILLE/MOUILLE  (dam-break wet/wet)
# ============================================================================

def dambreak_wetwet(x):
    """
    Rupture de barrage classique mouille/mouille.

    A gauche du barrage (x < 500) : h = 10 m (haute)
    A droite (x >= 500)          : h = 1 m  (basse)
    theta = 0.05 partout, u = 0.

    Retourne h, q, s.
    """
    h = np.where(x < 500, 10.0, 1.0)     # discontinuite de hauteur
    q = np.zeros_like(x)                  # repos initial
    s = h * 0.05                          # theta = 0.05 uniforme
    return h, q, s


# ============================================================================
# 4) RUPTURE DE BARRAGE - FOND SEC A DROITE  (dam-break dry bed)
# ============================================================================

def dambreak_dryright(x, *, theta0: float = 1.0):
    """
    Rupture de barrage 1-D : gauche mouille, droite seche.

    A gauche du milieu du domaine : h = 1 m
    A droite                      : h = h_min (quasi sec)
    theta est uniforme = theta0.

    C'est un test classique pour verifier la robustesse du schema
    au front de mouillage (interface mouille/sec).

    Retourne h, q, s.
    """
    h = np.where(x < x.max() / 2, 1.0, h_min)   # sec a droite
    q = np.zeros_like(x)
    s = h * theta0                                # theta uniforme
    return h, q, s


# ============================================================================
# 5) BOSSE GAUSSIENNE (bathymetrie de la these)
# ============================================================================

def bump_bathymetry(x):
    """
    Bathymetrie en forme de bosse gaussienne centree en x = 10 m.
    Z(x) = 0.2 * exp(-((x - 10) / 2)^2)
    Hauteur maximale = 0.2 m, largeur caracteristique = 2 m.
    """
    return 0.2 * np.exp(-((x - 10.0) / 2.0) ** 2)


# ============================================================================
# 6) ECOULEMENT FLUVIAL STATIONNAIRE SUR BOSSE
# ============================================================================

def stationary_fluvial_bump(x):
    """
    Ecoulement fluvial stationnaire au-dessus d'une bosse gaussienne.

    La surface libre est plate : eta = h + Z = 2.0 m.
    Le debit est constant : q = 4.42 m^2/s.
    theta = 1 (s = h).

    Ce cas-test verifie que le schema preserve un ecoulement
    stationnaire au-dessus d'une topographie non-triviale.

    Retourne h, q, s, b (bathymetrie incluse).
    """
    b   = bump_bathymetry(x)                # bosse gaussienne
    eta = 2.0                               # surface libre constante
    h   = np.maximum(eta - b, h_min)        # h = eta - Z (>= h_min)
    q   = 4.42 * np.ones_like(x)           # debit constant
    s   = h.copy()                          # theta = 1 => s = h
    return h, q, s, b


# ============================================================================
# 7) LAC AU REPOS SUR BATHYMETRIE (classique Saint-Venant)
# ============================================================================

def lake_at_rest(x, b):
    """
    Lac au repos : eta = h + Z = 2.0 m, u = 0.

    C'est le test fondamental de la propriete << bien-balance >> :
    un schema WB doit preserver cet etat exactement (a la precision
    machine) malgre la bathymetrie non plate.

    Retourne h, q, s (s = 0 car theta = 0 ici).
    """
    eta = 2.0
    h   = np.maximum(eta - b, h_min)       # h = eta - Z
    q   = np.zeros_like(x)                 # repos
    s   = np.zeros_like(x)                 # theta = 0
    return h, q, s


# ============================================================================
# 8) ECOULEMENT TRANSCRITIQUE SANS CHOC
# ============================================================================

def transcritique_sans_choc(x):
    """
    Ecoulement transcritique (passe de fluvial a torrentiel) SANS choc
    au-dessus d'une bosse gaussienne.

    Le debit q = 1.53 m^2/s est impose. On calcule la hauteur critique
    hc = (q^2/g)^(1/3), puis la hauteur amont h_up par Newton sur
    l'equation de Bernoulli.

    Ce test verifie la capacite du schema a capturer la transition
    entre regime fluvial (amont) et torrentiel (aval).

    Retourne h, q, s, b.
    """
    b = bump_bathymetry(x)
    q = 1.53
    # Hauteur critique : h ou le nombre de Froude = 1
    hc = (q*q/g)**(1/3)
    # Energie specifique en amont (loin de la bosse, Z = 0.2 au sommet)
    E_up = 1.5*hc + 0.2

    # Resolution de l'equation de Bernoulli h + q^2/(2gh^2) = E_up
    # par la methode de Newton (8 iterations suffisent)
    h_up = 1.0
    for _ in range(8):
        f  = h_up + q*q/(2*g*h_up**2) - E_up        # residu
        df = 1.0 - q*q/(g*h_up**3)                   # derivee
        h_up -= f/df
    h_up = float(h_up)

    # Assemblage du profil de hauteur par zones
    h = np.empty_like(x)
    h[x <  8.0] = h_up          # zone subcritique (amont de la bosse)
    h[x > 12.0] = 0.66          # zone supercritique (aval)
    mask = (x >= 8.0) & (x <= 12.0)
    h[mask] = hc                 # zone de transition (sur la bosse)
    s = h.copy()                 # theta = 1
    q0 = q * np.ones_like(x)
    return h, q0, s, b


# ============================================================================
# 9) ECOULEMENT TRANSCRITIQUE AVEC CHOC
# ============================================================================

def transcritique_avec_choc(x):
    """
    Ecoulement transcritique avec choc hydraulique.

    Le profil initial est un ecoulement au-dessus d'une bosse
    parabolique (support [8, 12] m). Un choc se forme en aval
    de la bosse quand l'ecoulement torrentiel rencontre l'eau
    au repos en aval.

    Retourne h, q, s, z (bathymetrie).
    """
    # Bathymetrie parabolique (support [8, 12])
    z = np.zeros_like(x)
    mask = (x>8.0) & (x<12.0)
    z[mask] = 0.2 - 0.005*(x[mask]-10.0)**2   # bosse parabolique

    H = 0.33                                   # niveau aval
    h = np.maximum(H - z, h_min)               # h = max(H - Z, 0)
    q = np.zeros_like(x)                       # repos initial
    s = h.copy()                               # theta = 1
    return h, q, s, z


# ============================================================================
# 10) LAC AU REPOS PERTURBE  (Touma 2015, section 3.1.2)
# ============================================================================

def lake_at_rest_perturbation(x: np.ndarray, *,
                              H: float = 6.0,
                              theta0: float = 4.0):
    """
    Construit un VERITABLE lac au repos pour le systeme de Ripa :

        eta = h + Z = H  (constante imposee)
        u = 0             (repos absolu)
        theta = theta0    (temperature uniforme)

    La bathymetrie Z(x) est composee de deux << bosses >> cosinus de
    classe C1, centrees sur x = -0.9 et x = +0.4, dans le domaine [-1, 1].

    Ce test verifie la propriete << bien-balance >> du schema :
    l'equilibre exact doit etre preserve a la precision machine.

    Retourne h, q, s, Z.
    """
    Z = np.zeros_like(x)

    # Bosse 1 (C1) - support [-1.0, -0.8], centree en x = -0.9
    mask1 = (-1.0 <= x) & (x <= -0.8)
    xi1 = 10 * np.pi * (x[mask1] + 0.9)          # variable reduite dans [-pi, pi]
    Z[mask1] = 0.85 * (np.cos(xi1) + 1.0)        # hauteur max = 1.70 m

    # Bosse 2 - support [0.3, 0.5], centree en x = +0.4
    mask2 = (0.3 <= x) & (x <= 0.5)
    xi2 = 10 * np.pi * (x[mask2] - 0.4)
    Z[mask2] = 1.25 * (np.cos(xi2) + 1.0)        # hauteur max = 2.50 m

    h = np.maximum(H - Z, h_min)                  # h = eta - Z >= h_min
    q = np.zeros_like(x)                          # repos
    s = h * theta0                                # s = h * theta (theta constant)
    return h, q, s, Z


# ============================================================================
# 11) RUPTURE DE BARRAGE SUR BOSSE RECTANGULAIRE  (Touma 2015, section 3.1.3)
# ============================================================================

def dam_break_rect_bump(x,
                        *,
                        H_L=20.0,       # niveau amont hors bosse (m)
                        H_R=15.0,       # niveau aval hors bosse (m)
                        theta_L=10.0,   # temperature a gauche
                        theta_R=5.0,    # temperature a droite
                        Z0=8.0,         # hauteur de la bosse (m)
                        xb=300.0,       # centre de la bosse (m)
                        half_width=75.0):  # demi-largeur de la bosse (m)
    """
    Rupture de barrage avec bosse rectangulaire (Touma 2015, section 3.1.3).

    Domaine = [0, 600] m.
    La bosse rectangulaire est centree en xb, de largeur 2*half_width,
    et de hauteur Z0.

    Conditions initiales :
      h(x, 0) = H_L - Z(x)  si x <= xb    (niveau amont)
      h(x, 0) = H_R - Z(x)  sinon          (niveau aval)
      u = 0 partout
      theta = theta_L a gauche, theta_R a droite

    Ce test combine une discontinuite de hauteur, de temperature ET
    de bathymetrie : c'est un cas tres exigeant pour le solveur.

    Retourne h, q, s, Z.
    """
    # Construction de la bathymetrie rectangulaire
    Z = np.zeros_like(x)
    mask_bump = np.abs(x - xb) < half_width
    Z[mask_bump] = Z0

    # Hauteur d'eau initiale (discontinue au centre)
    h = np.where(x <= xb, H_L - Z, H_R - Z)
    q = np.zeros_like(x)                         # repos
    # Temperature theta discontinue (gauche/droite)
    theta = np.where(x <= xb, theta_L, theta_R)
    s = h * theta                                 # s = h * theta
    return h, q, s, Z


# ============================================================================
#  CAS-TESTS 2D
# ============================================================================

# ============================================================================
# 12) RUPTURE DE BARRAGE RECTANGULAIRE (Touma 2015, section 3.2.1)
#     Domaine [-1, 1]^2, fond plat Z = 0
# ============================================================================

def rect_dambreak(x, y):
    """
    Rupture de barrage rectangulaire en 2D.

    A l'interieur d'un ruban vertical (|x| <= 0.5) :
      h = 2 m,  theta = 1
    A l'exterieur :
      h = 1 m,  theta = 1.5

    u = v = 0 partout, fond plat.

    Retourne h, qx, qy, s, Z.
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    inside = (np.abs(X) <= 0.5)                   # ruban vertical |x| <= 0.5
    h = np.where(inside, 2.0, 1.0)               # hauteur : 2 m dedans, 1 m dehors
    theta = np.where(inside, 1.0, 1.5)           # theta : 1 dedans, 1.5 dehors
    qx = np.zeros_like(h)                        # repos
    qy = np.zeros_like(h)
    s = h * theta                                 # s = h * theta
    Z = np.zeros_like(h)                          # fond plat
    return h, qx, qy, s, Z


# ============================================================================
# 13) RUPTURE DE BARRAGE CIRCULAIRE (Touma 2015, section 3.2.2)
#     Domaine [-1, 1]^2, fond plat Z = 0
# ============================================================================

def circular_dambreak(x, y):
    """
    Rupture de barrage circulaire en 2D.

    A l'interieur du cercle r <= 0.25 (r = sqrt(x^2 + y^2)) :
      h = 2 m
    A l'exterieur :
      h = 1 m

    u = v = 0 partout, theta = 1, fond plat.
    L'onde de choc se propage radialement de maniere symetrique.

    Retourne h, qx, qy, s, Z.
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.sqrt(X*X + Y*Y)                       # distance au centre
    h = np.ones_like(X)
    h[r <= 0.25] = 2.0                            # interieur du cercle
    qx = np.zeros_like(h)                        # repos
    qy = np.zeros_like(h)
    s  = h                                        # theta = 1 => s = h
    Z  = np.zeros_like(h)                         # fond plat
    return h, qx, qy, s, Z


# ============================================================================
# 14) PERTURBATION GAUSSIENNE SUR LIT IRREGULIER (Touma 2015, section 3.2.3)
# ============================================================================

def steady_gauss_perturb(x, y):
    """
    Etat quasi-stationnaire sur un fond irregulier avec theta constant.

    La bathymetrie Z est composee de deux bosses gaussiennes :
      - une a gauche centree en (-0.5, -0.5)
      - une a droite centree en (+0.5, +0.5)

    L'etat initial est un << lac au repos >> perturbe :
      eta = h + Z = H0 = 3.0 m  (constant)
      u = v = 0
      theta = 4/3

    Ce test verifie que le schema preserve l'equilibre h + Z = cste
    sur une topographie 2D non-triviale.

    Retourne h, qx, qy, s, Z.
    """
    X, Y = np.meshgrid(x, y, indexing="xy")
    # Bathymetrie : deux bosses gaussiennes
    Z = np.where(
        X <= 0.0,
        0.5 * np.exp(-100 * ((X + 0.5) ** 2 + (Y + 0.5) ** 2)),   # bosse gauche
        0.6 * np.exp(-100 * ((X - 0.5) ** 2 + (Y - 0.5) ** 2)),   # bosse droite
    )
    H0 = 3.0                                     # cote d'equilibre h + Z
    h = H0 - Z                                   # hauteur d'eau
    qx = np.zeros_like(h)                        # repos
    qy = np.zeros_like(h)
    theta = np.full_like(h, 4.0 / 3.0)           # theta = 4/3 partout
    s = h * theta
    return h, qx, qy, s, Z
