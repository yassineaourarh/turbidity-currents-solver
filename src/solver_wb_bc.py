# ============================================================================
#  solver_wb_bc.py - Solveur 1-D bien-balance avec conditions aux limites
# ============================================================================
#  Version 5 (02 mars 2025)
#
#  Ce solveur est concu pour etre robuste a l'interface mouille/sec.
#  Il combine :
#    - Reconstruction hydrostatique d'Audusse (well-balancing)
#    - Flux de Rusanov (module fluxes.py)
#    - Capteur mouille/sec (coupe les flux aux cellules quasi-seches)
#    - Splitting de Strang pour le frottement de Manning (demi-pas avant
#      et apres la mise a jour conservative)
#    - Conditions aux limites par ghost cells
#
#  Historique v4 -> v5 :
#    - La ghost cell amont herite de la meme temperature theta que la
#      premiere cellule interne => equilibre exact preserve meme si theta != 1
#      (cas << Lake at rest >>).
# ============================================================================

from __future__ import annotations
import numpy as np
from config  import g, h_min, FRICTION_LAW, n_manning
from fluxes  import rusanov_flux


# ============================================================================
#  Frottement de Manning - integration analytique locale
# ============================================================================

def apply_friction_manning(q: np.ndarray,
                           h: np.ndarray,
                           dt: float,
                           n: float) -> np.ndarray:
    """
    Applique le frottement de Manning par integration analytique.

    L'equation du frottement est :
      dq/dt = -g * n^2 * |q| * q / h^(7/3)

    La solution analytique sur un pas dt donne :
      q^{n+1} = q^n / (1 + dt * g * n^2 * |q^n| / h^{7/3})

    Cette formulation est inconditionnellement stable et ne produit
    pas de debit negatif si q > 0 initialement.

    Parametres
    ----------
    q : np.ndarray - debit(s) courant(s)
    h : np.ndarray - hauteur(s) d'eau
    dt : float     - pas de temps (peut etre dt/2 pour le splitting)
    n : float      - coefficient de Manning

    Retourne q apres frottement.
    """
    hs    = np.maximum(h, h_min)
    alpha = g * n**2 / hs**(7.0/3.0)
    return q / (1.0 + dt * alpha * np.abs(q))


# Seuil mouille/sec : en dessous de h_cut, la cellule est traitee
# comme seche (flux annule, debit mis a zero).
h_cut = 2.0e-2          # seuil mouille/sec = 2 cm


# ============================================================================
#  Pas de temps principal
# ============================================================================

def step(h: np.ndarray,
         q: np.ndarray,
         s: np.ndarray,
         b: np.ndarray,
         dx: float,
         dt: float,
         *,
         H_target: float | None = None,   # cote (eta) amont imposee
         H_up:     float | None = None,   # alias retro-compatibilite
         q_in:     float | None = None,   # debit amont optionnel
         friction: float = 0.0            # >0 => surcharge n_manning
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Avance la solution d'un pas de temps avec le schema bien-balance.

    Le schema suit les etapes suivantes :
      1. Ghost-cells et conditions aux limites (Dirichlet / transmissif)
      2. Reconstruction hydrostatique d'Audusse (well-balancing)
      3. Capteur mouille/sec (coupe le flux aux interfaces quasi-seches)
      4. Flux numerique de Rusanov (module fluxes.py)
      5. Splitting Manning (1er demi-pas : dt/2)
      6. Mise a jour conservative  U -= dt/dx * (F_{i+1/2} - F_{i-1/2})
      7. Terme source topographique  S_q = -g*theta*h*dZ/dx
      8. Splitting Manning (2eme demi-pas : dt/2)
      9. Controle de positivite et recoupure des cellules seches

    Parametres
    ----------
    h, q, s : np.ndarray - hauteur, debit, variable transportee (taille N)
    b       : np.ndarray - bathymetrie (taille N)
    dx      : float      - pas d'espace
    dt      : float      - pas de temps (deja limite par CFL)
    H_target: float|None - cote de surface libre imposee en amont (eta = H_target)
    H_up    : float|None - alias pour H_target (ancien nom)
    q_in    : float|None - debit impose en amont (0 par defaut => repos)
    friction: float      - surcharge du coefficient de Manning (0 = valeur config)

    Retourne
    --------
    h_new, q_new, s_new : np.ndarray - etats au pas de temps suivant
    """

    # ===== Etape 1 : Ghost cells (extension du domaine) =====
    # On ajoute une cellule fictive de chaque cote (mode 'edge' = Neumann)
    h_ext = np.pad(h, 1, mode='edge')
    q_ext = np.pad(q, 1, mode='edge')
    s_ext = np.pad(s, 1, mode='edge')
    b_ext = np.pad(b, 1, mode='edge')

    # ---------- Condition AMONT (ghost cell gauche, i=0) ----------
    if H_target is None:
        H_target = H_up                   # compatibilite ancienne interface
    if H_target is not None:
        # Imposer la cote de surface libre eta = H_target
        h_ext[0] = max(H_target - b_ext[0], h_min)

    # Debit impose en amont (si fourni)
    if q_in is not None:
        q_ext[0] = q_in
    else:
        q_ext[0] = 0.0                    # repos par defaut

    # La temperature theta du ghost cell amont = theta de la 1ere cellule interne
    # Ceci est ESSENTIEL pour preserver l'equilibre du lac au repos
    theta_up = s_ext[1] / max(h_ext[1], h_min)
    s_ext[0] = theta_up * h_ext[0]

    # ---------- Condition AVAL (ghost cell droite, i=-1) ----------
    # Condition transmissive : on copie simplement les valeurs de la derniere cellule
    h_ext[-1] = h_ext[-2]
    q_ext[-1] = q_ext[-2]
    s_ext[-1] = s_ext[-2]

    # ===== Etape 2 : Reconstruction hydrostatique d'Audusse =====
    # On calcule les etats aux interfaces en tenant compte de la bathymetrie
    # pour garantir la propriete well-balanced.
    bL, bR = b_ext[:-1], b_ext[1:]         # bathy gauche/droite a chaque interface
    hL, hR = h_ext[:-1], h_ext[1:]         # h gauche/droite
    qL, qR = q_ext[:-1], q_ext[1:]         # q gauche/droite
    sL, sR = s_ext[:-1], s_ext[1:]         # s gauche/droite

    # Elevation de la bathymetrie a l'interface au max des deux cotes
    b_max = np.maximum(bL, bR)
    # Hauteurs d'eau reconstruites  (eta_L - b_max) et (eta_R - b_max)
    hL_s  = np.maximum(0.0, (hL + bL) - b_max)
    hR_s  = np.maximum(0.0, (hR + bR) - b_max)

    # Variables primitives (vitesse u et temperature theta)
    uL  = np.where(hL > h_min, qL / hL, 0.0)
    uR  = np.where(hR > h_min, qR / hR, 0.0)
    thL = np.where(hL > h_min, sL / hL, 0.0)
    thR = np.where(hR > h_min, sR / hR, 0.0)

    # Reconstruction des etats conserves (h*, q* = u*h*, s* = theta*h*)
    UL = np.vstack([hL_s, uL * hL_s, thL * hL_s])
    UR = np.vstack([hR_s, uR * hR_s, thR * hR_s])

    # ===== Etape 3 : Capteur mouille/sec =====
    # On annule les flux aux interfaces ou au moins un cote est quasi-sec
    dry_int = (hL < h_cut) | (hR < h_cut)
    UL[:, dry_int] = 0.0
    UR[:, dry_int] = 0.0

    # ===== Etape 4 : Flux de Rusanov =====
    F = rusanov_flux(UL, UR)
    F[:, dry_int] = 0.0                     # securite supplementaire

    # ===== Etape 5 : Splitting Manning (1er demi-pas : dt/2) =====
    if FRICTION_LAW == "manning" and (friction > 0.0 or n_manning > 0.0):
        n_loc = friction if friction > 0.0 else n_manning
        q_ext[1:-1] = apply_friction_manning(q_ext[1:-1],
                                             h_ext[1:-1],
                                             0.5 * dt, n_loc)

    # ===== Etape 6 : Mise a jour conservative =====
    # U^{n+1} = U^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U = np.array([h_ext[1:-1], q_ext[1:-1], s_ext[1:-1]])
    U -= dt / dx * (F[:, 1:] - F[:, :-1])

    # ===== Etape 7 : Terme source topographique =====
    # S_q = -g * theta * h * dZ/dx  (differences centrees pour dZ/dx)
    db_dx = (b_ext[2:] - b_ext[:-2]) / (2.0 * dx)
    theta = np.where(U[0] > h_min, U[2] / U[0], 0.0)
    U[1] -= dt * g * theta * U[0] * db_dx

    # ===== Etape 8 : Splitting Manning (2eme demi-pas : dt/2) =====
    if FRICTION_LAW == "manning" and (friction > 0.0 or n_manning > 0.0):
        U[1] = apply_friction_manning(U[1], U[0], 0.5 * dt, n_loc)

    # ===== Etape 9 : Controle de positivite et recoupure =====
    h_new = np.maximum(U[0], h_min)           # h >= h_min
    very_dry = h_new < h_cut                   # cellules quasi-seches

    h_new[very_dry] = h_min                    # forcer h_min dans les cellules seches
    q_new = U[1]
    q_new[very_dry] = 0.0                      # annuler le debit dans les cellules seches

    s_new = np.maximum(U[2], 0.0)             # s >= 0 (theta peut etre > 1)
    s_new[very_dry] = h_new[very_dry]         # front sec : theta = 1 (valeur neutre)

    return h_new, q_new, s_new
