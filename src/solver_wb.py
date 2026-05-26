# ============================================================================
#  solver_wb.py - Schema bien-balance (Ripa) 1-D, version simple
# ============================================================================
#  Ce solveur implemente un schema volumes finis d'ordre 1 pour le systeme
#  de Ripa en 1-D. C'est une version plus simple que solver_wb_bc.py :
#    - Pas de reconstruction MUSCL (ordre 1 seulement)
#    - Conditions aux limites Neumann simples (pas de debit impose)
#    - Flux de Rusanov (Local Lax-Friedrichs)
#    - Terme source topographique bien-balance
#    - Frottement lineaire optionnel (pas de Manning ici)
#
#  Ce solveur est utilise pour les tests d'equilibre (Ripa equilibrium)
#  et la rupture de barrage sur bosse rectangulaire.
# ============================================================================

import numpy as np
from fluxes import compute_flux
from config import h_min, g


# ============================================================================
#  Flux de Rusanov (reimplemente localement)
# ============================================================================

def rusanov_flux(UL, UR):
    """
    Flux de Rusanov entre les etats UL et UR.

    Formule :
      F_num = 0.5 * (F(UL) + F(UR)) - 0.5 * a * (UR - UL)

    ou a = max(|u| + c) est la vitesse d'onde maximale.
    La celerite pour le modele Ripa est : c = sqrt(g * theta * h).

    Parametres
    ----------
    UL, UR : np.ndarray, chacun de shape (3, N) - [h, q, s] a gauche et droite
    """
    # Flux physiques des deux cotes
    FL = compute_flux(*UL)                        # flux gauche
    FR = compute_flux(*UR)                        # flux droit

    # Extraction des variables conservees
    hL, qL, sL = UL
    hR, qR, sR = UR

    # Variables primitives (avec securite h_min)
    uL = qL / np.maximum(hL, h_min)              # vitesse gauche
    uR = qR / np.maximum(hR, h_min)              # vitesse droite
    tL = sL / np.maximum(hL, h_min)              # theta gauche
    tR = sR / np.maximum(hR, h_min)              # theta droite

    # Celerites des ondes de gravite (modele Ripa)
    aL = np.abs(uL) + np.sqrt(g * tL * np.maximum(hL, h_min))
    aR = np.abs(uR) + np.sqrt(g * tR * np.maximum(hR, h_min))
    # Vitesse d'onde maximale pour la dissipation
    a  = np.maximum(aL, aR)

    # Formule de Rusanov
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)


# ============================================================================
#  Pas de temps WB-Ripa (schema d'ordre 1)
# ============================================================================

def step(h, q, s, b, dx, dt, *, friction=0.0):
    """
    Un pas de temps du schema bien-balance << Ripa >> (ordre 1).

    Le schema resout le systeme :
        U = (h, q, s)^T
        d_t U + d_x F(U) = S(U, b)

    avec le terme source topographique bien-balance :
        S_q = -g * theta * h * dZ/dx

    Etapes du schema :
      1) Cellules fantomes Neumann (copie des valeurs au bord)
      2) Flux numeriques de Rusanov a chaque interface
      3) Mise a jour conservative U -= dt/dx * (F_{i+1/2} - F_{i-1/2})
      4) Source topographique (differences centrees pour dZ/dx)
      5) Frottement lineaire optionnel
      6) Positivite : h >= h_min, s >= 0

    Parametres
    ----------
    h, q, s : np.ndarray - hauteur, debit, variable transportee (taille N)
    b       : np.ndarray - bathymetrie (taille N)
    dx      : float      - pas d'espace
    dt      : float      - pas de temps (deja limite par CFL)
    friction : float     - coefficient de frottement lineaire (0 = aucun)

    Retourne
    --------
    h_new, q_new, s_new : np.ndarray - solution au pas suivant
    """
    # -- 1) Cellules fantomes (Neumann = copie des bords) --
    h = np.pad(h, 1, mode="edge")
    q = np.pad(q, 1, mode="edge")
    s = np.pad(s, 1, mode="edge")
    b = np.pad(b, 1, mode="edge")

    # -- 2) Flux numeriques de Rusanov sur chaque interface --
    # UL[i] = etat de la cellule i,  UR[i] = etat de la cellule i+1
    UL = np.array([h[:-1], q[:-1], s[:-1]])      # etats gauche (interface i+1/2)
    UR = np.array([h[1:],  q[1:],  s[1:]])       # etats droite (interface i+1/2)
    Fint = rusanov_flux(UL, UR)                   # flux numriques, shape (3, N+1)

    # -- 3) Mise a jour conservative --
    # U_i^{n+1} = U_i^n - dt/dx * (F_{i+1/2} - F_{i-1/2})
    U = np.array([h[1:-1], q[1:-1], s[1:-1]])    # cellules internes seulement
    U -= dt / dx * (Fint[:, 1:] - Fint[:, :-1])

    # -- 4) Source topographique (well-balanced) --
    # S_q = -g * theta * h * dZ/dx
    # On utilise des differences centrees : dZ/dx ~ (Z[i+1] - Z[i-1]) / (2*dx)
    db_dx = (b[2:] - b[:-2]) / (2 * dx)          # gradient de la bathymetrie
    theta = U[2] / np.maximum(U[0], h_min)        # theta = s / h
    U[1]  -= dt * g * theta * U[0] * db_dx        # force hydrostatique du fond

    # -- 5) Frottement lineaire (optionnel) --
    # Terme de frottement simple : dq/dt = -friction * u * h
    if friction > 0.0:
        u = U[1] / np.maximum(U[0], h_min)        # vitesse
        U[1] -= dt * friction * u * np.maximum(U[0], h_min)

    # -- 6) Etats physiquement admissibles --
    h_new = np.maximum(U[0], h_min)               # positivite de h
    q_new = np.where(h_new > h_min, U[1], 0.0)   # q = 0 si sec

    # s >= 0 (theta peut etre > 1, c'est physiquement correct pour Ripa)
    s_new = np.maximum(U[2], 0.0)

    return h_new, q_new, s_new
