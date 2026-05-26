# ============================================================================
#  solver_tc.py - Solveur pour ecoulements transcritiques (avec/sans choc)
# ============================================================================
#  Ce module implemente un solveur 1-D pour le systeme de Ripa adapte aux
#  ecoulements transcritiques :
#    - Reconstruction MUSCL (ordre 2 en espace) avec limiteur minmod
#    - Flux numerique HLL (Harten-Lax-van Leer)
#    - Reconstruction hydrostatique (Audusse et al.) pour le well-balancing
#    - Capteur de choc optionnel (desactive la reconstruction MUSCL
#      dans les zones de forte variation)
#    - Conditions aux limites : debit impose en amont, hauteur imposee en aval
#    - Frottement de Manning (integration analytique)
#
#  Deux wrappers publics :
#    step_tc()      : ecoulement transcritique sans choc
#    step_tc_choc() : ecoulement transcritique avec choc (capteur actif)
# ============================================================================

import numpy as np
from config import g, h_min, FRICTION_LAW, n_manning


# ============================================================================
#  Limiteur minmod
# ============================================================================

def minmod(a, b):
    """
    Limiteur minmod vectoriel.
    Retourne 0 si a et b sont de signes opposes,
    sinon le plus petit en valeur absolue.
    Garantit la propriete TVD du schema MUSCL.
    """
    s = np.sign(a) + np.sign(b)      # = 0 si signes opposes, +/-2 si meme signe
    return 0.5 * s * np.minimum(np.abs(a), np.abs(b))


# ============================================================================
#  Flux physiques et vitesses d'onde
# ============================================================================

def phys_flux(h, q, s):
    """
    Calcule les flux physiques F(U) du systeme de Ripa.

    F = ( q,
          q^2/h + 0.5*g*theta*h^2,
          s*u )
    ou theta = s/h et u = q/h.

    Retourne un array de shape (3, N).
    """
    hs    = np.maximum(h, h_min)       # securite division par zero
    u     = q / hs                     # vitesse
    theta = s / hs                     # temperature potentielle
    return np.array([
        q,                                      # flux de masse
        q * u + 0.5 * g * theta * hs**2,        # flux de quantite de mouvement
        s * u                                   # flux de temperature (advection)
    ])


def wave_speeds(hL, qL, sL, hR, qR, sR):
    """
    Calcule les vitesses d'onde minimale (SL) et maximale (SR)
    pour le solveur HLL, a chaque interface.

    Les vitesses sont estimees par :
        SL = min(uL - cL, uR - cR)   (onde la plus rapide vers la gauche)
        SR = max(uL + cL, uR + cR)   (onde la plus rapide vers la droite)

    ou c = sqrt(g * theta * h) est la celerite des ondes de gravite Ripa.
    """
    hsL = np.maximum(hL, h_min); hsR = np.maximum(hR, h_min)
    uL  = qL / hsL;              uR  = qR / hsR
    # Celerites des ondes de gravite (modele Ripa : c^2 = g*theta*h)
    thL = sL / hsL;              thR = sR / hsR
    cL  = np.sqrt(g * thL * hsL); cR = np.sqrt(g * thR * hsR)
    # Bornes gauche et droite du cone d'influence
    SL = np.minimum(uL - cL, uR - cR)
    SR = np.maximum(uL + cL, uR + cR)
    return SL, SR


# ============================================================================
#  Flux numerique HLL
# ============================================================================

def hll_flux(UL, UR):
    """
    Flux numerique HLL (Harten-Lax-van Leer) entre les etats UL et UR.

    Le flux HLL est donne par :
      - F = FL                                    si SL >= 0 (ecoulement vers la droite)
      - F = FR                                    si SR <= 0 (ecoulement vers la gauche)
      - F = (SR*FL - SL*FR + SL*SR*(UR-UL))      sinon (cas intermediaire)
            / (SR - SL)

    C'est un solveur de Riemann approche qui capture correctement
    les chocs et les rarefactions.

    Parametres
    ----------
    UL, UR : np.ndarray, shape (3, N_interfaces)
    """
    hL, qL, sL = UL
    hR, qR, sR = UR
    # Flux physiques a gauche et a droite
    FL = phys_flux(hL, qL, sL)
    FR = phys_flux(hR, qR, sR)
    # Vitesses d'onde min/max
    SL, SR = wave_speeds(hL, qL, sL, hR, qR, sR)

    F = np.zeros_like(FL)
    # Cas 1 : toutes les ondes vont vers la droite => flux = FL
    maskL = SL >= 0
    # Cas 2 : toutes les ondes vont vers la gauche => flux = FR
    maskR = SR <= 0
    # Cas 3 : les ondes encadrent l'interface => formule HLL
    maskM = ~(maskL | maskR)

    F[:, maskL] = FL[:, maskL]
    F[:, maskR] = FR[:, maskR]

    denom = SR[maskM] - SL[maskM]
    F[:, maskM] = (
        SR[maskM] * FL[:, maskM]
        - SL[maskM] * FR[:, maskM]
        + SL[maskM] * SR[maskM] * (UR[:, maskM] - UL[:, maskM])
    ) / denom
    return F


# ============================================================================
#  Reconstruction MUSCL (ordre 2)
# ============================================================================

def muscl_slopes(arr_ext):
    """
    Calcule les pentes limitees MUSCL avec le limiteur minmod.

    Pour chaque cellule i, la pente limitee est :
      sigma_i = minmod(arr[i] - arr[i-1], arr[i+1] - arr[i])

    Parametres
    ----------
    arr_ext : np.ndarray - tableau etendu (avec ghost cells), taille N+2
                           (indices 0=ghost_L, 1..N=cellules, N+1=ghost_R)

    Retourne
    --------
    np.ndarray, taille N - pentes limitees pour les cellules internes
    """
    dL = arr_ext[1:-1] - arr_ext[:-2]     # delta gauche : arr[i] - arr[i-1]
    dR = arr_ext[2:]   - arr_ext[1:-1]    # delta droite : arr[i+1] - arr[i]
    return minmod(dL, dR)


# ============================================================================
#  Reconstruction hydrostatique (Audusse et al.)
# ============================================================================

def hydrostatic_reconstruction(hL, qL, sL, bL,
                               hR, qR, sR, bR):
    """
    Reconstruction hydrostatique d'Audusse pour le well-balancing.

    L'idee est de reconstruire les etats a l'interface de maniere a
    preserver exactement l'equilibre du lac au repos (h + Z = cste, u = 0).

    On eleve la bathymetrie au max des deux cotes :
      b* = max(bL, bR)
    et on ajuste les hauteurs d'eau :
      hL* = max(0, etaL - b*)    ou etaL = hL + bL
      hR* = max(0, etaR - b*)    ou etaR = hR + bR

    Les vitesses u et temperatures theta sont conservees de l'etat original,
    mais appliquees sur les hauteurs reconstruites.

    Retourne hL*, qL*, sL*, hR*, qR*, sR*.
    """
    bmax = np.maximum(bL, bR)               # bathymetrie maximale a l'interface
    etaL = hL + bL                           # surface libre gauche
    etaR = hR + bR                           # surface libre droite
    hL_  = np.maximum(0.0, etaL - bmax)     # hauteur reconstruite gauche
    hR_  = np.maximum(0.0, etaR - bmax)     # hauteur reconstruite droite

    # Conservation des grandeurs primitives (u, theta) de l'etat original
    uL   = np.where(hL > h_min, qL/hL, 0.0)
    uR   = np.where(hR > h_min, qR/hR, 0.0)
    thL  = np.where(hL > h_min, sL/hL, 0.0)
    thR  = np.where(hR > h_min, sR/hR, 0.0)

    # Reconstruction des variables conservees sur les hauteurs ajustees
    qL_  = uL  * hL_
    qR_  = uR  * hR_
    sL_  = thL * hL_
    sR_  = thR * hR_
    return hL_, qL_, sL_, hR_, qR_, sR_


# ============================================================================
#  Conditions aux limites
# ============================================================================

def apply_bc(h, q, s, b, *, q_in=None, h_up=None, h_down=None):
    """
    Applique les conditions aux limites via des ghost cells (cellules fantomes).

    Les tableaux sont etendus d'une cellule de chaque cote (mode 'edge' = Neumann).
    Ensuite on impose eventuellement :
      - Amont (i=0) : debit q_in et/ou hauteur h_up imposee
      - Aval (i=-1) : hauteur h_down imposee, q transmissif

    La temperature theta de la cellule fantome est heritee de la cellule
    interne adjacente pour preserver l'equilibre.

    Retourne h_ext, q_ext, s_ext, b_ext (tableaux etendus, taille N+2).
    """
    # Extension par copie des valeurs au bord (Neumann / transmissif)
    h_e = np.pad(h, 1, mode='edge')
    q_e = np.pad(q, 1, mode='edge')
    s_e = np.pad(s, 1, mode='edge')
    b_e = np.pad(b, 1, mode='edge')

    # --- Condition amont (ghost cell gauche) ---
    if q_in is not None:
        q_e[0] = q_in              # debit impose en entree
    if h_up is not None:
        h_e[0] = h_up              # hauteur imposee en amont
    # theta du ghost = theta de la 1ere cellule interne
    th = s_e[1] / max(h_e[1], h_min)
    s_e[0] = th * h_e[0]

    # --- Condition aval (ghost cell droite) ---
    if h_down is not None:
        h_e[-1] = h_down           # hauteur imposee en aval
    q_e[-1] = q_e[-2]              # debit transmissif (copie)
    # theta du ghost = theta de la derniere cellule interne
    th = s_e[-2] / max(h_e[-2], h_min)
    s_e[-1] = th * h_e[-1]

    return h_e, q_e, s_e, b_e


# ============================================================================
#  Frottement de Manning (integration analytique)
# ============================================================================

def apply_friction_manning(q, h, dt, n):
    """
    Applique le frottement de Manning par integration analytique.

    L'equation du frottement est :
      dq/dt = -g * n^2 * |q| * q / h^(7/3)

    La solution analytique sur un pas dt donne :
      q^(n+1) = q^n / (1 + dt * g * n^2 * |q^n| / h^(7/3))

    Cette formulation est inconditionnellement stable, meme pour
    de petites hauteurs d'eau.

    Parametres
    ----------
    q : np.ndarray - debit actuel
    h : np.ndarray - hauteur d'eau
    dt : float     - pas de temps
    n : float      - coefficient de Manning

    Retourne
    --------
    np.ndarray - debit apres frottement
    """
    hs   = np.maximum(h, h_min)
    alpha = g * n**2 / (hs**(7.0/3.0))
    return q / (1.0 + dt * alpha * np.abs(q))


# ============================================================================
#  Pas de temps generique (coeur du solveur)
# ============================================================================

def one_step(h, q, s, b, dx, dt,
             q_in=None, h_up=None, h_down=None,
             friction=0.0,
             with_shock=False,
             slope_sensor=0.1,
             du_threshold=0.5):
    """
    Avance la solution d'un pas de temps dt.

    Etapes :
      1) Conditions aux limites (ghost cells)
      2) Reconstruction MUSCL (pentes limitees)
      3) Capteur de choc optionnel (coupe la reconstruction pres des chocs)
      4) Reconstruction des etats aux interfaces
      5) Reconstruction hydrostatique (well-balancing)
      6) Calcul du flux HLL
      7) Mise a jour conservative
      8) Terme source topographique  S_q = -g * theta * h * dZ/dx
      9) Frottement de Manning (si actif)
      10) Controle de positivite (h >= h_min, s >= 0)

    Parametres
    ----------
    h, q, s : np.ndarray - etats courants (taille N)
    b       : np.ndarray - bathymetrie (taille N)
    dx, dt  : float      - pas d'espace et de temps
    q_in    : float|None - debit impose en amont
    h_up    : float|None - hauteur imposee en amont
    h_down  : float|None - hauteur imposee en aval
    friction : float     - coefficient de frottement (0 = aucun)
    with_shock : bool    - activer le capteur de choc
    slope_sensor : float - seuil du capteur (variation relative de h)
    du_threshold : float - seuil du capteur (variation absolue de u)

    Retourne
    --------
    h_new, q_new, s_new : np.ndarray - etats au pas de temps suivant
    """
    N = h.size

    # ---- 1) Extension avec conditions aux limites ----
    h_e, q_e, s_e, b_e = apply_bc(h, q, s, b,
                                  q_in=q_in, h_up=h_up, h_down=h_down)

    # ---- 2) Pentes MUSCL sur les cellules internes ----
    sh = muscl_slopes(h_e)     # pentes de h
    sq = muscl_slopes(q_e)     # pentes de q
    ss = muscl_slopes(s_e)     # pentes de s

    # ---- 3) Capteur de choc (optionnel) ----
    # Detecte les zones de forte variation (chocs) et y coupe la
    # reconstruction MUSCL (retour a l'ordre 1 localement).
    if with_shock:
        u = np.where(h > h_min, q/h, 0.0)
        dh = np.abs(np.diff(h))           # saut de h entre cellules voisines
        du = np.abs(np.diff(u))           # saut de u entre cellules voisines
        h_mid = 0.5 * (h[:-1] + h[1:])   # h moyen a l'interface
        # Detection : variation relative de h trop grande OU variation de u trop grande
        sensor_int = (dh > slope_sensor * np.maximum(h_mid, h_min)) | (du > du_threshold)
        # Propager le capteur aux cellules voisines
        sensor_cells = np.zeros(N, dtype=bool)
        sensor_cells[:-1] |= sensor_int
        sensor_cells[1:]  |= sensor_int
        # Annuler les pentes dans les cellules << choquees >> (retour ordre 1)
        sh[sensor_cells] = 0.0
        sq[sensor_cells] = 0.0
        ss[sensor_cells] = 0.0

    # ---- 4) Extension des pentes pour les ghost cells ----
    sh_full = np.zeros(N+2); sh_full[1:-1] = sh
    sq_full = np.zeros(N+2); sq_full[1:-1] = sq
    ss_full = np.zeros(N+2); ss_full[1:-1] = ss

    # ---- 5) Reconstruction des etats gauche/droite aux interfaces ----
    # Etat gauche : valeur de la cellule i + demi-pente vers la droite
    h_L = h_e[:-1] + 0.5 * sh_full[:-1]
    q_L = q_e[:-1] + 0.5 * sq_full[:-1]
    s_L = s_e[:-1] + 0.5 * ss_full[:-1]
    b_L = b_e[:-1]

    # Etat droite : valeur de la cellule i+1 - demi-pente vers la gauche
    h_R = h_e[1:] - 0.5 * sh_full[1:]
    q_R = q_e[1:] - 0.5 * sq_full[1:]
    s_R = s_e[1:] - 0.5 * ss_full[1:]
    b_R = b_e[1:]

    # ---- 6) Reconstruction hydrostatique (well-balancing) ----
    hL_, qL_, sL_, hR_, qR_, sR_ = hydrostatic_reconstruction(
        h_L, q_L, s_L, b_L,
        h_R, q_R, s_R, b_R
    )
    UL = np.array([hL_, qL_, sL_])
    UR = np.array([hR_, qR_, sR_])

    # ---- 7) Calcul du flux numerique HLL ----
    F = hll_flux(UL, UR)

    # ---- 8) Mise a jour conservative ----
    # U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U      = np.array([h_e, q_e, s_e])
    U_new  = U[:, 1:-1].copy()
    U_new -= dt/dx * (F[:, 1:] - F[:, :-1])

    # ---- 9) Terme source topographique ----
    # S_q = -g * theta * h * dZ/dx  (gradient de pression lie au fond)
    # On utilise des differences centrees pour dZ/dx
    db_dx  = (b_e[2:] - b_e[:-2]) / (2 * dx)
    theta  = np.where(U_new[0] > h_min, U_new[2]/U_new[0], 0.0)
    U_new[1] -= dt * g * theta * U_new[0] * db_dx

    # ---- 10) Frottement de Manning ----
    if (FRICTION_LAW == "manning") and (friction > 0.0 or n_manning > 0.0):
        n_loc = friction if friction > 0.0 else n_manning
        U_new[1] = apply_friction_manning(U_new[1], U_new[0], dt, n_loc)

    # ---- 11) Controle de positivite ----
    h_new = np.maximum(U_new[0], h_min)           # h >= h_min
    q_new = np.where(h_new > h_min, U_new[1], 0.0)  # q = 0 si sec
    s_new = np.maximum(U_new[2], 0.0)             # s >= 0, theta peut etre > 1

    return h_new, q_new, s_new


# ============================================================================
#  Wrappers publics
# ============================================================================

def step_tc(h, q, s, b, dx, dt, *, q_in, h_down, h_up, friction=0.0):
    """
    Un pas de temps pour ecoulement transcritique SANS choc.
    Reconstruction MUSCL active partout (pas de capteur de choc).
    """
    return one_step(h, q, s, b, dx, dt,
                    q_in=q_in, h_up=h_up, h_down=h_down,
                    friction=friction,
                    with_shock=False)


def step_tc_choc(h, q, s, b, dx, dt, *,
                 q_in, h_down, friction=0.0,
                 slope_sensor=0.1, du_threshold=0.5):
    """
    Un pas de temps pour ecoulement transcritique AVEC choc.
    Le capteur de choc desactive la reconstruction MUSCL
    dans les zones de forte variation pour eviter les oscillations.
    """
    return one_step(h, q, s, b, dx, dt,
                    q_in=q_in, h_up=None, h_down=h_down,
                    friction=friction,
                    with_shock=True,
                    slope_sensor=slope_sensor,
                    du_threshold=du_threshold)
