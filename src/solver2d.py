# ============================================================================
#  solver2d.py - Solveur 2-D couple pour le systeme de Ripa
# ============================================================================
#  Ce module implemente un solveur 2-D pour les equations de Ripa
#  (Saint-Venant avec temperature/densite variable theta).
#
#  Variables conservees en 2-D :
#    U = (h, qx, qy, s) = (hauteur, debit x, debit y, h*theta)
#
#  Flux en direction x :
#    Fx = ( qx,
#           qx^2/h + 0.5*g*theta*h^2,
#           qx*qy/h,
#           qx*theta )
#
#  Flux en direction y :
#    Fy = ( qy,
#           qx*qy/h,
#           qy^2/h + 0.5*g*theta*h^2,
#           qy*theta )
#
#  Schema numerique :
#    - Rusanov (Local Lax-Friedrichs) dimension par dimension
#    - Ordre 1 en espace (pas de reconstruction MUSCL en 2D)
#    - Correction << surface-gradient >> pour le well-balancing :
#      annule -g*theta*h*grad(Z), preservant l'equilibre lac au repos
#    - Ghost cells Neumann (copie des bords)
#
#  Securisations :
#    - theta = max(s/h, 0) pour eviter les sqrt de valeurs negatives
#    - Apres chaque pas : h >= h_min et s >= 0
# ============================================================================

from __future__ import annotations
import numpy as np
from config import g, h_min, CFL


# ============================================================================
#  Utilitaires
# ============================================================================

def theta_from(h: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Calcule theta = s/h avec securite : force theta >= 0 et h >= h_min.

    C'est la temperature potentielle (ou densite relative) utilisee
    dans le modele de Ripa. Si h est trop petit, on retourne 0.
    """
    return np.maximum(s / np.maximum(h, h_min), 0.0)


# ============================================================================
#  Flux physiques - Direction x
# ============================================================================

def flux_x(h, qx, qy, s):
    """
    Calcule les 4 composantes du flux en direction x :
      Fx = ( qx,
             qx^2/h + 0.5*g*theta*h^2,   (quantite de mouv. x + pression Ripa)
             qx*qy/h,                      (couplage croise des debits)
             qx*theta )                    (transport de la temperature)

    Parametres : h, qx, qy, s  tableaux 2-D de meme forme (Ny, Nx).
    Retourne un tableau de shape (4, Ny, Nx).
    """
    theta = theta_from(h, s)
    inv_h = 1.0 / np.maximum(h, h_min)      # 1/h avec protection
    return np.stack(
        (
            qx,                                           # flux de masse
            qx * qx * inv_h + 0.5 * g * h * h * theta,  # flux qte mouvement x
            qx * qy * inv_h,                             # couplage croise
            qx * theta,                                  # transport de theta
        ),
        axis=0,
    )


# ============================================================================
#  Flux physiques - Direction y
# ============================================================================

def flux_y(h, qx, qy, s):
    """
    Calcule les 4 composantes du flux en direction y :
      Fy = ( qy,
             qx*qy/h,                      (couplage croise)
             qy^2/h + 0.5*g*theta*h^2,     (quantite de mouv. y + pression Ripa)
             qy*theta )                    (transport de la temperature)

    Parametres : h, qx, qy, s  tableaux 2-D de meme forme.
    Retourne un tableau de shape (4, Ny, Nx).
    """
    theta = theta_from(h, s)
    inv_h = 1.0 / np.maximum(h, h_min)
    return np.stack(
        (
            qy,                                           # flux de masse
            qx * qy * inv_h,                             # couplage croise
            qy * qy * inv_h + 0.5 * g * h * h * theta,  # flux qte mouvement y
            qy * theta,                                  # transport de theta
        ),
        axis=0,
    )


# ============================================================================
#  Flux de Rusanov a l'interface (vectorise)
# ============================================================================

def rusanov_interface(UL, UR, FL, FR, a):
    """
    Flux numerique de Rusanov a l'interface, vectorise pour le 2-D.

    Formule :
      F_num = 0.5 * (FL + FR) - 0.5 * a * (UR - UL)

    Parametres
    ----------
    UL, UR : np.ndarray, shape (4, ...) - etats gauche et droite
    FL, FR : np.ndarray, shape (4, ...) - flux physiques gauche et droite
    a      : np.ndarray, shape (...) - vitesse d'onde maximale a l'interface

    Retourne le flux numerique, shape (4, ...).
    """
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)


# ============================================================================
#  Pas de temps complet 2-D (Strang splitting : x puis y)
# ============================================================================

def step_2d(h, qx, qy, s, Z, dx, dy, dt):
    """
    Avance les champs 2-D d'un pas de temps dt.

    Le schema procede en 3 etapes (splitting dimensionnel) :
      1) Sweep en direction x :
         - Calcul des flux Rusanov sur les interfaces verticales (i+1/2, j)
         - Mise a jour U -= dt/dx * (F_{i+1/2} - F_{i-1/2})
      2) Sweep en direction y :
         - Calcul des flux Rusanov sur les interfaces horizontales (i, j+1/2)
         - Mise a jour U -= dt/dy * (G_{j+1/2} - G_{j-1/2})
      3) Source topographique bien-balancee :
         - qx -= dt * g * theta * h * dZ/dx
         - qy -= dt * g * theta * h * dZ/dy

    Parametres
    ----------
    h, qx, qy, s : np.ndarray 2-D  - champs (Ny, Nx)
    Z             : np.ndarray 2-D  - bathymetrie (Ny, Nx)
    dx, dy        : float           - pas d'espace en x et y
    dt            : float           - pas de temps

    Retourne
    --------
    h_new, qx_new, qy_new, s_new : np.ndarray 2-D
    """
    # ===== Extension par ghost cells Neumann (1 cellule autour) =====
    h = np.pad(h,   1, "edge")
    qx = np.pad(qx, 1, "edge")
    qy = np.pad(qy, 1, "edge")
    s  = np.pad(s,  1, "edge")
    Z  = np.pad(Z,  1, "edge")

    # ===== SWEEP en direction X =====
    # Flux physiques aux interfaces verticales (i, i+1) en fixant y (colonnes internes)
    FL = flux_x(h[:-1, 1:-1], qx[:-1, 1:-1], qy[:-1, 1:-1], s[:-1, 1:-1])
    FR = flux_x(h[1:, 1:-1],  qx[1:, 1:-1],  qy[1:, 1:-1],  s[1:, 1:-1])
    # Etats gauche/droite pour la dissipation
    UL = np.stack((h[:-1, 1:-1], qx[:-1, 1:-1], qy[:-1, 1:-1], s[:-1, 1:-1]))
    UR = np.stack((h[1:, 1:-1],  qx[1:, 1:-1],  qy[1:, 1:-1],  s[1:, 1:-1]))

    # Vitesse d'onde maximale |u| + c (pour la dissipation Rusanov)
    uL = qx[:-1, 1:-1] / np.maximum(h[:-1, 1:-1], h_min)
    uR = qx[1:, 1:-1]  / np.maximum(h[1:, 1:-1],  h_min)
    cL = np.sqrt(g * theta_from(h[:-1, 1:-1], s[:-1, 1:-1]) * h[:-1, 1:-1])
    cR = np.sqrt(g * theta_from(h[1:, 1:-1],  s[1:, 1:-1])  * h[1:, 1:-1])
    a  = np.maximum(np.abs(uL) + cL, np.abs(uR) + cR)

    # Flux numerique de Rusanov en x
    Fnum = rusanov_interface(UL, UR, FL, FR, a)

    # Mise a jour conservative en x
    U = np.stack((h[1:-1, 1:-1], qx[1:-1, 1:-1], qy[1:-1, 1:-1], s[1:-1, 1:-1]))
    U -= (dt / dx) * (Fnum[:, 1:, :] - Fnum[:, :-1, :])

    # ===== SWEEP en direction Y =====
    # Flux physiques aux interfaces horizontales (j, j+1) en fixant x (lignes internes)
    Gd = flux_y(h[1:-1, :-1], qx[1:-1, :-1], qy[1:-1, :-1], s[1:-1, :-1])
    Gg = flux_y(h[1:-1, 1:],  qx[1:-1, 1:],  qy[1:-1, 1:],  s[1:-1, 1:])
    UL = np.stack((h[1:-1, :-1], qx[1:-1, :-1], qy[1:-1, :-1], s[1:-1, :-1]))
    UR = np.stack((h[1:-1, 1:],  qx[1:-1, 1:],  qy[1:-1, 1:],  s[1:-1, 1:]))

    # Vitesse d'onde maximale |v| + c (direction y)
    vL = qy[1:-1, :-1] / np.maximum(h[1:-1, :-1], h_min)
    vR = qy[1:-1, 1:]  / np.maximum(h[1:-1, 1:],  h_min)
    cL = np.sqrt(g * theta_from(h[1:-1, :-1], s[1:-1, :-1]) * h[1:-1, :-1])
    cR = np.sqrt(g * theta_from(h[1:-1, 1:],  s[1:-1, 1:])  * h[1:-1, 1:])
    a  = np.maximum(np.abs(vL) + cL, np.abs(vR) + cR)

    # Flux numerique de Rusanov en y
    Gnum = rusanov_interface(UL, UR, Gd, Gg, a)
    # Mise a jour conservative en y
    U -= (dt / dy) * (Gnum[:, :, 1:] - Gnum[:, :, :-1])

    # ===== Source topographique bien-balancee =====
    # Gradient de la bathymetrie (differences centrees)
    h_new, qx_new, qy_new, s_new = U
    dzdx = (Z[2:, 1:-1] - Z[:-2, 1:-1]) / (2 * dx)     # dZ/dx
    dzdy = (Z[1:-1, 2:] - Z[1:-1, :-2]) / (2 * dy)     # dZ/dy
    theta = theta_from(h_new, s_new)

    # Force exercee par le gradient bathymetrique sur chaque composante de debit
    qx_new -= dt * g * theta * h_new * dzdx      # source en x
    qy_new -= dt * g * theta * h_new * dzdy      # source en y

    # ===== Controle physique =====
    h_new = np.maximum(h_new, h_min)                       # h >= h_min
    s_new = np.maximum(s_new, h_new * 1e-12)               # s > 0 (theta > 0)
    qx_new = np.where(h_new > h_min, qx_new, 0.0)        # debit nul si sec
    qy_new = np.where(h_new > h_min, qy_new, 0.0)

    return h_new, qx_new, qy_new, s_new
