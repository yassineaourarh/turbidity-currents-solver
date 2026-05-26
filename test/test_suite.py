# ============================================================================
#  test_suite.py - Banc d'essai 1-D pour le systeme de Ripa (h, q, s)
# ============================================================================
#  Ce fichier orchestre l'execution de tous les cas-tests 1-D :
#    - Ecoulement fluvial stationnaire sur bosse
#    - Ecoulement transcritique sans choc
#    - Ecoulement transcritique avec choc
#    - Rupture de barrage (fond sec) avec frottement Manning
#    - Equilibre Ripa exact (u=0, h^2*theta = C)
#    - Equilibre Ripa perturbe (petite oscillation sinusoidale)
#    - Lac au repos perturbe sur bathymetrie non-plane (Touma 2015, 3.1.2)
#    - Rupture de barrage sur bosse rectangulaire (Touma 2015, 3.1.3)
#
#  Chaque cas-test possede un << runner >> dedie qui :
#    1) Initialise les conditions (via initial_conditions.py)
#    2) Boucle en temps avec le solveur adapte
#    3) Sauvegarde des snapshots a intervalles reguliers
#    4) Produit les figures (via plotting.py)
#
#  Lancer tous les tests :  python test_suite.py
# ============================================================================

import numpy as np
from config       import Lx, Nx, CFL, Tmax, g, h_min, n_manning, dx
from plotting     import plot_fields
from solver_wb_bc import step as step_wb_bc    # solveur WB avec conditions limites
from solver_tc    import step_tc, step_tc_choc # solveur transcritique
from solver_wb    import step as step_wb       # solveur WB simple (Ripa)
step_ripa = step_wb                            # alias pour lisibilite

from initial_conditions import (
    stationary_fluvial_bump,
    transcritique_sans_choc,
    transcritique_avec_choc,
    dambreak_dryright,
    bump_bathymetry,
    ripa_equilibrium, perturb_height,
    lake_at_rest_perturbation,
    dam_break_rect_bump
)


# ============================================================================
#  RUNNER 1 : Tests generiques avec solveur WB-BC  (solver_wb_bc)
# ============================================================================

def run_test(name, h, q, s, b, *, friction=0.0,
             impose_debit=True, return_snapshots=False):
    """
    Execute un cas-test generique avec le solveur bien-balance + conditions limites.

    Le solveur impose :
      - H_target = (h + b)[0]  (cote de surface libre a l'entree)
      - q_in = q[0]            (debit a l'entree, si impose_debit=True)

    Parametres
    ----------
    name : str           - nom du test (utilise pour le titre des figures)
    h, q, s, b          - conditions initiales (taille Nx)
    friction : float     - coefficient de frottement
    impose_debit : bool  - True = imposer le debit amont
    return_snapshots     - True = renvoyer les eta et u snapshots
    """
    x      = np.linspace(0.0, Lx, Nx)
    dx_loc = x[1] - x[0]

    # Conditions aux limites deduites des CI
    H_target = (h + b)[0]                # cote de surface libre initiale en amont
    q_in     = q[0] if impose_debit else None   # debit entrant (ou libre)

    # Listes pour stocker les snapshots
    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    t = 0.0

    # ===== Boucle en temps =====
    while t < Tmax:
        # Calcul du pas de temps CFL
        h_safe = np.maximum(h, h_min)
        theta  = s / h_safe                      # temperature potentielle
        celer  = np.sqrt(g * theta * h_safe)     # celerite des ondes
        u_loc  = q / h_safe                      # vitesse
        dt     = CFL * dx_loc / np.max(np.abs(u_loc) + celer)
        dt     = min(dt, Tmax - t)               # ne pas depasser Tmax

        # Avance d'un pas de temps
        h, q, s = step_wb_bc(
            h, q, s, b, dx_loc, dt,
            H_target=H_target,
            q_in=q_in,
            friction=friction
        )

        t += dt

        # Sauvegarde des snapshots (toutes les 2 secondes environ)
        if int(t) % 2 == 0 and (not t_list or int(t) != int(t_list[-1])):
            h_safe = np.maximum(h, h_min)
            eta_snap.append(h_safe + b)           # surface libre eta = h + Z
            u_snap.append(q / h_safe)             # vitesse
            theta = s / h_safe
            c_snap.append(theta)                  # theta
            P_snap.append(0.5 * h_safe * theta**2)  # pression Ripa
            t_list.append(t)

    # Production des figures
    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=name, save=True)
    if return_snapshots:
        return np.array(eta_snap), np.array(u_snap)


# ============================================================================
#  RUNNER 2 : Ecoulement transcritique SANS choc  (solver_tc)
# ============================================================================

def run_test_tc(name, h, q, s, b, *, q_in, h_down, h_up,
                friction=0.0, return_snapshots=False):
    """
    Execute un cas-test transcritique sans choc.

    Conditions aux limites :
      - q_in  : debit impose en amont
      - h_down : hauteur imposee en aval
      - h_up   : hauteur imposee en amont

    Le solveur utilise step_tc (MUSCL + HLL, sans capteur de choc).
    """
    x      = np.linspace(0.0, Lx, Nx)
    dx_loc = x[1] - x[0]

    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    t = 0.0

    while t < Tmax:
        # Pas de temps CFL
        h_safe = np.maximum(h, h_min)
        theta  = s / h_safe
        celer  = np.sqrt(g * theta * h_safe)
        u_loc  = q / h_safe
        dt     = CFL * dx_loc / np.max(np.abs(u_loc) + celer)
        dt     = min(dt, Tmax - t)

        # Avance avec le solveur transcritique
        h, q, s = step_tc(h, q, s, b, dx_loc, dt,
                          q_in=q_in, h_down=h_down, h_up=h_up,
                          friction=friction)

        t += dt

        # Snapshot toutes les 2 secondes
        if int(t) % 2 == 0 and (not t_list or int(t) != int(t_list[-1])):
            h_safe = np.maximum(h, h_min)
            eta_snap.append(h_safe + b)
            u_snap.append(q / h_safe)
            theta = s / h_safe
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=name, save=True)
    if return_snapshots:
        return np.array(eta_snap), np.array(u_snap)


# ============================================================================
#  RUNNER 3 : Ecoulement transcritique AVEC choc  (solver_tc)
# ============================================================================

def run_test_tc_choc(name, h, q, s, b, *, q_in, h_down,
                     friction=0.0, return_snapshots=False,
                     dump_every=2.0, cfl=CFL):
    """
    Execute un cas-test transcritique avec choc.

    Le capteur de choc est actif : il desactive localement la
    reconstruction MUSCL dans les zones de forte variation pour
    eviter les oscillations pres du choc.

    Les snapshots sont sauvegardes toutes les dump_every secondes,
    plus le snapshot initial (t=0).
    """
    x      = np.linspace(0.0, Lx, Nx)
    dx_loc = x[1] - x[0]

    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    t = 0.0
    next_dump = 0.0

    # Snapshot initial (t = 0)
    h_safe = np.maximum(h, h_min)
    eta_snap.append(h_safe + b)
    u_snap.append(q / h_safe)
    theta = s / h_safe
    c_snap.append(theta)
    P_snap.append(0.5 * h_safe * theta**2)
    t_list.append(t)
    next_dump += dump_every

    while t < Tmax:
        # Pas de temps CFL avec securite
        h_safe = np.maximum(h, h_min)
        u_loc  = q / h_safe
        theta  = s / h_safe
        celer  = np.sqrt(g * theta * h_safe)
        denom  = np.max(np.abs(u_loc) + celer)
        if (not np.isfinite(denom)) or denom <= 0.0:
            denom = np.sqrt(g * h_min)       # securite si tout est quasi-sec

        dt = cfl * dx_loc / denom
        dt = min(dt, Tmax - t)

        # Avance avec capteur de choc actif
        h, q, s = step_tc_choc(h, q, s, b, dx_loc, dt,
                               q_in=q_in, h_down=h_down,
                               friction=friction)

        t += dt

        # Sauvegarde periodique
        if t >= next_dump - 1e-12 or t >= Tmax - 1e-12:
            h_safe = np.maximum(h, h_min)
            eta_snap.append(h_safe + b)
            u_snap.append(q / h_safe)
            theta = s / h_safe
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)
            next_dump += dump_every

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=name, save=True)
    if return_snapshots:
        return np.array(eta_snap), np.array(u_snap)


# ============================================================================
#  RUNNER 4 : Equilibre Ripa (u=0, h^2*theta = C)
# ============================================================================

def run_test_equilibrium(T=10.0, b=1.0, C=1.0,
                         outname="Equilibrium_eq5", dump_every=2.0):
    """
    Valide la preservation de l'equilibre exact du systeme de Ripa (eq. 5).

    Conditions :
      theta = b = constante  (profil plat)
      h = sqrt(C / theta) = constante
      u = 0

    L'etat doit rester PARFAITEMENT stationnaire (a la precision machine).
    On mesure les erreurs max|h^2*theta - C| et max|q| en fin de simulation.
    """
    # Construction du maillage
    Nx_loc = int(Lx / dx) + 1
    x = np.linspace(0.0, Lx, Nx_loc, endpoint=False)
    btopo = np.zeros_like(x)              # fond plat

    # Condition initiale : equilibre avec a=0 (theta constant)
    h, q, s = ripa_equilibrium(x, a=0.0, b=b, C=C)

    # Listes de snapshots
    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    next_dump = 0.0

    # Boucle en temps
    t = 0.0
    while t < T - 1e-12:
        h_safe = np.maximum(h, h_min)
        theta = s / h_safe
        celer = np.sqrt(g * h_safe * theta)
        dt = CFL * dx / (np.max(celer) + 1e-16)
        dt = min(dt, T - t)

        # Avance avec le solveur simple (Ripa, ordre 1)
        h, q, s = step_wb(h, q, s, btopo, dx, dt)
        t += dt

        # Snapshots periodiques
        if t >= next_dump - 1e-12 or t >= T - 1e-12:
            eta_snap.append(h_safe)           # eta = h (car Z = 0)
            u_snap.append(q / h_safe)
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)
            next_dump += dump_every

    # Erreurs numeriques par rapport a l'equilibre exact
    E_h2t = np.max(np.abs(h**2 * (s / h) - C))   # erreur sur h^2*theta
    E_u = np.max(np.abs(q))                        # erreur sur le debit (doit etre 0)
    print(f"{outname}: max|h^2*theta-C| = {E_h2t:.3e}, max|q| = {E_u:.3e}")

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=outname, save=True)


# ============================================================================
#  RUNNER 5 : Equilibre Ripa perturbe (oscillations)
# ============================================================================

def run_test_equilibrium_perturbation(T=10.0, a=0.2, b=1.0, C=1.0,
                                      eps=1e-3, k=1,
                                      outname="Equilibrium_perturb",
                                      dump_every=2.0, return_snapshots=False):
    """
    Teste la reponse du systeme a une petite perturbation autour de l'equilibre.

    On construit l'equilibre Ripa avec theta(x) = a*x + b (variable !),
    puis on ajoute une perturbation sinusoidale de hauteur eps * sin(2*pi*k*x/L).

    L'erreur finale E = sqrt(mean((h^2*theta - C)^2)) est affichee.
    Si le schema est bien-balance, E reste petit (de l'ordre de eps).
    """
    # Maillage
    Nx = int(Lx / dx) + 1
    x  = np.linspace(0.0, Lx, Nx, endpoint=False)
    btopo = np.zeros_like(x)                  # fond plat

    # CI : equilibre + petite perturbation
    h, q, s = ripa_equilibrium(x, a=a, b=b, C=C)
    h, q, s = perturb_height(h, q, s, eps=eps, k=k)

    # Snapshots
    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    next_dump = 0.0

    # Boucle en temps
    t = 0.0
    while t < T - 1e-12:
        h_safe = np.maximum(h, h_min)
        u_loc  = q / h_safe
        theta  = s / h_safe
        celer  = np.sqrt(g * h_safe * theta)
        dt     = CFL * dx / (np.max(np.abs(u_loc) + celer) + 1e-16)
        dt     = min(dt, T - t)

        # Avance d'un pas (solveur Ripa simple)
        h, q, s = step_wb(h, q, s, btopo, dx, dt)
        t += dt

        # Sauvegarde periodique
        if t >= next_dump - 1e-12 or t >= T - 1e-12:
            h_safe = np.maximum(h, h_min)
            eta_snap.append(h_safe)               # eta = h (Z = 0)
            u_snap.append(q / h_safe)
            theta  = s / h_safe
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)
            next_dump += dump_every

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=outname, save=True)

    # Erreur finale
    E = np.sqrt(np.mean((h**2 * theta - C)**2))
    print(f"{outname}: E_final = {E:.3e} apres {len(t_list)-1} dumps")

    if return_snapshots:
        return np.array(eta_snap), np.array(u_snap)


# ============================================================================
#  RUNNER 6 : Lac au repos perturbe (Touma 2015, section 3.1.2)
# ============================================================================

def run_test_lake_rest(T=1.0, H=6.0, theta0=4.0,
                       Nx_dom=400, outname="Lake_at_rest",
                       dump_every=0.2):
    """
    Verifie la propriete bien-balancee sur un lac au repos
    avec bathymetrie non-plate (deux bosses cosinus).

    L'equilibre exact est :
      eta = h + Z = H = 6.0 m (constant)
      u = 0
      theta = theta0 = 4.0

    On mesure les erreurs max|eta - H| et max|q| : elles doivent
    rester a la precision machine (~1e-15) pour un schema WB exact.

    Utilise le solveur step_wb_bc avec H_target = H et q_in = 0.
    """
    # Domaine [-1, 1]
    x = np.linspace(-1.0, 1.0, Nx_dom, endpoint=False)
    dx_loc = x[1] - x[0]

    # CI : lac au repos exact (Touma 2015, section 3.1.2)
    h, q, s, Z = lake_at_rest_perturbation(x, H=H, theta0=theta0)

    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    next_dump = 0.0
    t = 0.0

    while t < T - 1e-12:
        # Pas de temps CFL
        h_safe = np.maximum(h, h_min)
        theta  = s / h_safe
        celer  = np.sqrt(g * h_safe * theta)
        dt     = CFL * dx_loc / (np.max(celer) + 1e-16)
        dt     = min(dt, T - t)

        # Avance : solveur WB-BC (ghost cells + conditions limites)
        h, q, s = step_wb_bc(
            h, q, s, Z, dx_loc, dt,
            H_target=H,        # impose eta = H sur la ghost cell amont
            q_in=0.0            # debit nul (repos)
        )

        t += dt

        # Snapshots periodiques
        if t >= next_dump - 1e-12 or t >= T - 1e-12:
            h_safe = np.maximum(h, h_min)
            eta_snap.append(h_safe + Z)          # surface libre
            u_snap.append(q / h_safe)
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)
            next_dump += dump_every

    # Erreurs par rapport a l'equilibre exact
    E_eta = np.max(np.abs((h + Z) - H))    # ecart de la surface libre
    E_q   = np.max(np.abs(q))               # debit residuel (doit etre 0)
    print(f"{outname}: max|eta-H| = {E_eta:.3e}, max|q| = {E_q:.3e}")

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=outname, save=True)


# ============================================================================
#  RUNNER 7 : Rupture de barrage sur bosse rectangulaire (Touma 2015, 3.1.3)
# ============================================================================

def run_test_dambreak_bump(T=60.0, Nx_dom=1200,
                           outname="Dam_break_bump",
                           dump_every=5.0,
                           friction=0.0,
                           return_snapshots=False):
    """
    Rupture de barrage au-dessus d'une bosse rectangulaire.

    Ce cas combine une discontinuite de hauteur, de temperature ET
    de bathymetrie. Le domaine est [0, 600] m avec 1200 mailles.

    On utilise le solveur Ripa simple (step_ripa = solver_wb.step)
    avec des conditions Neumann (pas de CL imposees).
    """
    # Domaine et conditions initiales (Touma 2015)
    x       = np.linspace(0.0, 600.0, Nx_dom, endpoint=False)
    dx_loc  = x[1] - x[0]
    h, q, s, Z = dam_break_rect_bump(x)           # CI de l'article

    # Boucle en temps
    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    next_dump = 0.0
    t = 0.0

    while t < T - 1e-12:
        # CFL local
        h_safe = np.maximum(h, h_min)
        u_loc  = np.where(h_safe > 0.0, q / h_safe, 0.0)
        theta  = s / h_safe
        celer  = np.sqrt(g * theta * h_safe)
        dt     = CFL * dx_loc / (np.max(np.abs(u_loc) + celer) + 1e-16)
        dt     = min(dt, T - t)

        # Avance avec le solveur Ripa (WB simple)
        h, q, s = step_ripa(h, q, s, Z, dx_loc, dt, friction=friction)
        t += dt

        # Sauvegarde des champs
        if t >= next_dump - 1e-12 or t >= T - 1e-12:
            h_safe = np.maximum(h, h_min)
            u_loc  = np.where(h_safe > 0.0, q / h_safe, 0.0)
            theta  = s / h_safe

            eta_snap.append(h + Z)                    # surface libre
            u_snap.append(u_loc)                      # vitesse
            c_snap.append(theta)                      # theta
            P_snap.append(0.5 * h_safe * theta**2)   # pression Ripa
            t_list.append(t)
            next_dump += dump_every

    # Figure recapitulative (avec bathymetrie Z en tirets)
    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                Z=Z, title=outname, save=True)

    if return_snapshots:
        return (np.array(eta_snap), np.array(u_snap))


# ============================================================================
#  RUNNER 8 : Tests generiques avec solveur Ripa simple  (solver_wb)
# ============================================================================

def run_test_ripa(name, h, q, s, b, *, friction=0.0,
                  dump_every=2.0, return_snapshots=False):
    """
    Execute un cas-test avec le solveur Ripa simple (solver_wb.step).

    Ce runner est similaire a run_test, mais utilise le solveur WB simple
    (sans conditions limites imposees, Neumann partout).
    Adapte pour les ruptures de barrage sans contraintes aux bords.
    """
    x      = np.linspace(0.0, Lx, Nx)
    dx_loc = x[1] - x[0]

    eta_snap, u_snap, c_snap, P_snap, t_list = [], [], [], [], []
    next_dump = 0.0
    t = 0.0

    while t < Tmax - 1e-12:
        # Pas de temps CFL
        h_safe = np.maximum(h, h_min)
        u_loc  = np.where(h_safe > 0.0, q / h_safe, 0.0)
        theta  = s / h_safe
        celer  = np.sqrt(g * theta * h_safe)
        dt     = CFL * dx_loc / (np.max(np.abs(u_loc) + celer) + 1e-16)
        dt     = min(dt, Tmax - t)

        # Avance avec le solveur Ripa
        h, q, s = step_ripa(h, q, s, b, dx_loc, dt, friction=friction)
        t += dt

        # Snapshots periodiques
        if t >= next_dump - 1e-12 or t >= Tmax - 1e-12:
            theta = np.where(h_safe > 0.0, s / h_safe, 0.0)
            eta_snap.append(h + b)
            u_snap.append(u_loc)
            c_snap.append(theta)
            P_snap.append(0.5 * h_safe * theta**2)
            t_list.append(t)
            next_dump += dump_every

    plot_fields(x, eta_snap, u_snap, c_snap, t_list, P_snap,
                title=name, save=True)

    if return_snapshots:
        return np.array(eta_snap), np.array(u_snap)


# ============================================================================
#  POINT D'ENTREE PRINCIPAL
# ============================================================================

if __name__ == "__main__":

    # Maillage de reference
    x = np.linspace(0.0, Lx, Nx)

    # ---- Test 1 : Rupture de barrage sur bosse rectangulaire ----
    print("==> Dam-break sur bosse rectangulaire")
    run_test_dambreak_bump()

    # ---- Test 2 : Lac au repos perturbe (validation WB + bathymetrie) ----
    print("==> Lac au repos perturbe -- validation WB avec topographie")
    run_test_lake_rest(T=1.0)

    # ---- Test 3 : Rupture de barrage (fond sec) avec solveur Ripa ----
    print("==> Rupture de barrage (fond sec a droite) -- solveur Ripa")
    hdb, qdb, sdb = dambreak_dryright(x)
    bdb = np.zeros_like(x)
    run_test_ripa("Dambreak_dry_friction_Ripa",
              hdb, qdb, sdb, bdb,
              friction=n_manning)

    # ---- Test 4 : Rupture de barrage (fond sec) + Manning ----
    print("==> Rupture de barrage (fond sec a droite) avec frottement Manning")
    hdb, qdb, sdb = dambreak_dryright(x)
    bdb = np.zeros_like(x)
    run_test("Dambreak_dry_friction",
             hdb, qdb, sdb, bdb,
             friction=n_manning, impose_debit=False)

    # ---- Test 5 : Transcritique avec choc ----
    print("==> Ecoulement transcritique avec choc")
    h3, q3, s3, b3 = transcritique_avec_choc(x)
    run_test_tc_choc("Transcritique_avec_choc",
                     h3, q3, s3, b3,
                     q_in=0.18, h_down=0.33,
                     friction=n_manning)

    # ---- Test 6 : Fluvial stationnaire sur bosse ----
    print("==> Fluvial stationnaire sur bosse")
    h, q, s, b = stationary_fluvial_bump(x)
    run_test("Fluvial_bosse_these", h, q, s, b,
             impose_debit=True, friction=0.0)

    # ---- Test 7 : Transcritique sans choc ----
    print("==> Ecoulement transcritique sans choc")
    h0, q0, s0, b0 = transcritique_sans_choc(x)
    q_in   = 1.53          # debit impose en amont
    h_down = 0.66          # hauteur imposee en aval
    h_up   = 1.0           # hauteur imposee en amont
    run_test_tc("Transcritique_sans_choc",
                h0, q0, s0, b0,
                q_in=q_in, h_down=h_down, h_up=h_up,
                friction=0.0)

    # ---- Test 8 : Equilibre Ripa perturbe (sinusoide) ----
    print("==> Equilibre Ripa + sinus")
    run_test_equilibrium_perturbation()

    # ---- Test 9 : Equilibre exact (u=0, h^2*theta = C) ----
    print("==> Equilibre exact (u=0, h^2*theta=C) -- test bien-balance")
    run_test_equilibrium(T=10.0)
