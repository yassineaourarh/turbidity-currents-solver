# ============================================================================
#  fluxes.py - Flux physiques et flux numerique de Rusanov
# ============================================================================
#  Ce module definit les flux du systeme de Ripa a 3 composantes (h, q, s) :
#
#    Variables conservees :  U = (h, q, s)
#      h : hauteur d'eau (m)
#      q : debit lineique = h * u  (m^2/s)
#      s : variable transportee = h * theta  (theta = temperature potentielle)
#
#    Flux physiques :
#      F(U) = ( q,
#               q*u + 0.5*g*theta*h^2,
#               q*theta )
#
#    Le flux numerique de Rusanov (ou Local Lax-Friedrichs) est une methode
#    simple et robuste pour approcher le flux aux interfaces entre cellules.
# ============================================================================

import numpy as np
from config import g, h_min


def compute_flux(h: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Calcule les flux physiques F(U) du systeme de Ripa.

    Parametres
    ----------
    h : np.ndarray - hauteur d'eau dans chaque cellule
    q : np.ndarray - debit lineique (h*u) dans chaque cellule
    s : np.ndarray - variable transportee (h*theta) dans chaque cellule

    Retourne
    --------
    np.ndarray de shape (3, N) contenant les 3 composantes du flux :
      [0] Fh = q                              (conservation de la masse)
      [1] Fq = q*u + 0.5*g*theta*h^2          (conservation de la quantite de mouvement)
      [2] Fs = q*theta                         (transport de la temperature)
    """
    # Vitesse u = q/h (avec protection contre h -> 0)
    u = q / np.maximum(h, h_min)
    # Temperature potentielle theta = s/h
    theta = s / np.maximum(h, h_min)
    # Composante masse : flux = debit
    Fh = q
    # Composante quantite de mouvement : flux = pression dynamique + hydrostatique (Ripa)
    Fq = q * u + 0.5 * g * theta * h**2
    # Composante temperature : advection de theta par le debit
    Fs = q * theta
    return np.array([Fh, Fq, Fs])


def wave_speed(h: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """
    Calcule la vitesse d'onde maximale en chaque cellule.

    Pour le systeme de Ripa, la vitesse de propagation des ondes est :
        c = sqrt(g * theta * h)
    et la vitesse d'onde maximale vaut :
        lambda_max = |u| + c

    Cette vitesse sert a calculer le pas de temps CFL et la dissipation
    dans le flux de Rusanov.

    Retourne
    --------
    np.ndarray - |u| + sqrt(g * theta * h) pour chaque cellule
    """
    u = q / np.maximum(h, h_min)
    theta = s / np.maximum(h, h_min)
    return np.abs(u) + np.sqrt(g * theta * np.maximum(h, h_min))


def rusanov_flux(UL: np.ndarray, UR: np.ndarray) -> np.ndarray:
    """
    Flux numerique de Rusanov (Local Lax-Friedrichs) entre deux etats.

    Le flux de Rusanov est une combinaison du flux physique moyen et
    d'un terme de dissipation proportionnel au saut d'etat :

        F_num = 0.5 * (F(UL) + F(UR)) - 0.5 * a * (UR - UL)

    ou a = max(lambda_max(UL), lambda_max(UR)) est la vitesse d'onde
    maximale des deux cotes de l'interface.

    Parametres
    ----------
    UL : np.ndarray, shape (3, N_interfaces) - etats a gauche de chaque interface
    UR : np.ndarray, shape (3, N_interfaces) - etats a droite de chaque interface

    Retourne
    --------
    np.ndarray, shape (3, N_interfaces) - flux numerique a chaque interface
    """
    # Flux physiques des deux cotes
    FL = compute_flux(*UL)
    FR = compute_flux(*UR)
    # Vitesse d'onde maximale (dissipation numerique)
    a  = np.maximum(wave_speed(*UL), wave_speed(*UR))
    # Formule de Rusanov : moyenne des flux + dissipation
    return 0.5 * (FL + FR) - 0.5 * a * (UR - UL)
