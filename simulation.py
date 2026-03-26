import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


@dataclass
class Params:
    mode: str  # "nve" | "robust"
    scenario: str  # "collision2" | "gas"
    seed: int

    n: int
    steps: int
    dt: float
    box: float

    sigma: float
    epsilon: float
    m: float
    rc: float  # cutoff radius

    vmax: float
    r0: float  # collision2 only (in units of sigma)

    softcore_a: float  # robust only (in units of sigma; can be 0 for none)
    dv_target: float  # robust only
    safety: float  # robust only
    max_substeps: int  # robust only

    steps_per_frame: int
    plot_every: int
    pause_s: float
    no_gui: bool

    out_dir: str
    prefix: str
    save_frames: bool
    gif: bool
    gif_fps: int


def lj_potential_shift(epsilon: float, sigma2: float, rc2: float) -> float:
    inv_rc2 = 1.0 / rc2
    sr2 = sigma2 * inv_rc2
    sr6 = sr2 * sr2 * sr2
    sr12 = sr6 * sr6
    return 4.0 * epsilon * (sr12 - sr6)


def compute_forces(
    positions: np.ndarray,
    box: float,
    epsilon: float,
    sigma2: float,
    rc2: float,
    v_shift: float,
    a2: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    Minimum image + cutoff (potential-shifted) Lennard-Jones.
    Returns (forces, r_min, Epot, Fmax).

    If a2 > 0: uses soft-core distance r_eff^2 = r^2 + a^2.
    """
    n = positions.shape[0]
    forces = np.zeros_like(positions)
    r2_min_eff = np.inf
    Epot = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            rij = positions[j] - positions[i]
            rij -= box * np.rint(rij / box)

            r2 = float(rij[0] * rij[0] + rij[1] * rij[1])
            r2_eff = r2 + a2
            if r2_eff > rc2:
                continue

            if r2_eff < 1e-18:
                r2_eff = 1e-18

            inv_r2 = 1.0 / r2_eff
            sr2 = sigma2 * inv_r2
            sr6 = sr2 * sr2 * sr2
            sr12 = sr6 * sr6

            V = 4.0 * epsilon * (sr12 - sr6) - v_shift
            Epot += V

            coeff = 24.0 * epsilon * (2.0 * sr12 - sr6) * inv_r2
            fij = coeff * rij
            forces[i] += fij
            forces[j] -= fij

            if r2_eff < r2_min_eff:
                r2_min_eff = r2_eff

    r_min = float(np.sqrt(r2_min_eff)) if np.isfinite(r2_min_eff) else np.inf
    Fmax = float(np.sqrt(np.max(np.sum(forces * forces, axis=1))))
    return forces, r_min, Epot, Fmax


def vv_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    dt: float,
    box: float,
    m: float,
    epsilon: float,
    sigma2: float,
    rc2: float,
    v_shift: float,
    a2: float,
) -> tuple[np.ndarray, float, float, float]:
    """
    One velocity-verlet step of size dt.
    Returns (forces_new, r_min, Epot, Fmax).
    """
    velocities += 0.5 * forces / m * dt
    positions[:] = (positions + velocities * dt) % box
    forces_new, r_min, Epot, Fmax = compute_forces(positions, box, epsilon, sigma2, rc2, v_shift, a2)
    velocities += 0.5 * forces_new / m * dt
    return forces_new, r_min, Epot, Fmax


def advance_nve_fixed_dt(
    dt_total: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    box: float,
    m: float,
    epsilon: float,
    sigma2: float,
    rc2: float,
    v_shift: float,
    a2: float,
) -> tuple[np.ndarray, float, float, float]:
    return vv_step(positions, velocities, forces, dt_total, box, m, epsilon, sigma2, rc2, v_shift, a2)


def advance_robust_adaptive(
    dt_total: float,
    positions: np.ndarray,
    velocities: np.ndarray,
    forces: np.ndarray,
    Fmax: float,
    box: float,
    m: float,
    epsilon: float,
    sigma2: float,
    rc2: float,
    v_shift: float,
    a2: float,
    dv_target: float,
    safety: float,
    max_substeps: int,
) -> tuple[np.ndarray, float, float, float, int]:
    """
    Advance exactly dt_total with adaptive dt chosen to limit dv ~ (Fmax/m)*dt.
    Recomputes forces (and thus Fmax) every substep.
    """
    dt_rem = dt_total
    r_min_macro = np.inf
    Epot = 0.0
    substeps = 0

    while dt_rem > 0.0:
        substeps += 1
        if substeps > max_substeps:
            raise RuntimeError(
                "Too many adaptive substeps; increase dv_target, increase soft-core a, "
                "or reduce the macro dt."
            )

        Fmax_safe = max(Fmax, 1e-12)
        dt_cap = safety * dv_target * m / Fmax_safe
        dt = dt_cap if dt_cap < dt_rem else dt_rem

        forces, r_min, Epot, Fmax = vv_step(positions, velocities, forces, dt, box, m, epsilon, sigma2, rc2, v_shift, a2)
        if r_min < r_min_macro:
            r_min_macro = r_min
        dt_rem -= dt

    return forces, r_min_macro, Epot, Fmax, substeps


def init_collision2(box: float, sigma: float, r0_sigma: float, vmax: float) -> tuple[np.ndarray, np.ndarray]:
    r0 = r0_sigma * sigma
    positions = np.zeros((2, 2), dtype=float)
    positions[0] = [box / 2.0 - r0 / 2.0, box / 2.0]
    positions[1] = [box / 2.0 + r0 / 2.0, box / 2.0]
    velocities = np.array([[vmax, 0.0], [-vmax, 0.0]], dtype=float)
    return positions, velocities


def init_gas(n: int, box: float, vmax: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    positions = np.zeros((n, 2), dtype=float)
    n_side = int(np.ceil(np.sqrt(n)))
    spacing = box / n_side

    k = 0
    for i in range(n_side):
        for j in range(n_side):
            if k >= n:
                break
            positions[k, 0] = (i + 0.5) * spacing
            positions[k, 1] = (j + 0.5) * spacing
            k += 1

    velocities = rng.normal(size=(n, 2)) * vmax
    velocities -= velocities.mean(axis=0)
    return positions, velocities


def setup_plot(box: float, positions: np.ndarray):
    fig, ax = plt.subplots(2, 1)
    ax[0].set_xlim(0, box)
    ax[0].set_ylim(0, box)
    ax[0].set_aspect("equal", adjustable="box")
    scat = ax[0].scatter(positions[:, 0], positions[:, 1], s=25)

    ax[1].set_xlabel("t")
    ax[1].set_ylabel("Etot - Etot0")
    line, = ax[1].plot([], [], lw=1)
    return fig, ax, scat, line


def render(fig, ax, scat, line, positions, t_hist, dEtot_hist, pause_s: float):
    if not t_hist:
        return
    scat.set_offsets(positions)
    line.set_data(t_hist, dEtot_hist)

    ax[1].set_xlim(0.0, t_hist[-1])
    ymin = min(dEtot_hist)
    ymax = max(dEtot_hist)
    if ymax == ymin:
        ymax = ymin + 1.0
    pad = 0.05 * (ymax - ymin)
    ax[1].set_ylim(ymin - pad, ymax + pad)

    # Use draw() so this also works with non-interactive backends (e.g., Agg).
    fig.canvas.draw()
    if pause_s > 0.0:
        plt.pause(pause_s)


def save_energy_plot(path_png: str, t_hist: list[float], dEtot_hist: list[float]):
    if not t_hist:
        return
    plt.figure()
    plt.plot(t_hist, dEtot_hist, lw=1)
    plt.xlabel("t")
    plt.ylabel("Etot - Etot0")
    plt.title("Energy Drift (Etot - Etot0)")
    plt.tight_layout()
    plt.savefig(path_png, dpi=160)
    plt.close()


def parse_args() -> Params:
    p = argparse.ArgumentParser(
        description="2D Lennard-Jones Molecular Dynamics with PBC and Velocity-Verlet."
    )
    p.add_argument("--mode", choices=["nve", "robust"], default="robust")
    p.add_argument("--scenario", choices=["collision2", "gas"], default="gas")
    p.add_argument("--seed", type=int, default=1)

    p.add_argument("--n", type=int, default=64)
    p.add_argument("--steps", type=int, default=200000)
    p.add_argument("--dt", type=float, default=5e-5, help="Macro dt (each outer step advances this much time).")
    p.add_argument("--box", type=float, default=8.0)

    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--epsilon", type=float, default=1.0)
    p.add_argument("--m", type=float, default=1.0)
    p.add_argument("--rc", type=float, default=2.5, help="Cutoff radius in units of sigma.")

    p.add_argument("--vmax", type=float, default=0.5)
    p.add_argument("--r0", type=float, default=1.5, help="Initial separation in units of sigma (collision2).")

    p.add_argument("--a", type=float, default=None, help="Soft-core length in units of sigma (robust default 0.6).")
    p.add_argument("--dv-target", type=float, default=0.02)
    p.add_argument("--safety", type=float, default=0.5)
    p.add_argument("--max-substeps", type=int, default=50000)

    p.add_argument("--steps-per-frame", type=int, default=250)
    p.add_argument("--plot-every", type=int, default=10)
    p.add_argument("--pause", type=float, default=0.001)
    p.add_argument("--no-gui", action="store_true", help="Run without interactive plotting (faster and more stable).")

    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--prefix", type=str, default="")
    p.add_argument(
        "--save-frames",
        action="store_true",
        help="Save PNG frames to outputs/<prefix>_frames/.",
    )
    p.add_argument(
        "--gif",
        action="store_true",
        help="If used with --save-frames, try to build outputs/<prefix>.gif (requires pillow).",
    )
    p.add_argument("--gif-fps", type=int, default=30)
    a = p.parse_args()

    softcore_a = 0.0
    if a.a is None:
        softcore_a = 0.0 if a.mode == "nve" else 0.6
    else:
        softcore_a = float(a.a)

    prefix = a.prefix.strip()
    if not prefix:
        prefix = f"{a.scenario}_{a.mode}"

    return Params(
        mode=a.mode,
        scenario=a.scenario,
        seed=int(a.seed),
        n=int(a.n),
        steps=int(a.steps),
        dt=float(a.dt),
        box=float(a.box),
        sigma=float(a.sigma),
        epsilon=float(a.epsilon),
        m=float(a.m),
        rc=float(a.rc),
        vmax=float(a.vmax),
        r0=float(a.r0),
        softcore_a=float(softcore_a),
        dv_target=float(a.dv_target),
        safety=float(a.safety),
        max_substeps=int(a.max_substeps),
        steps_per_frame=int(a.steps_per_frame),
        plot_every=int(a.plot_every),
        pause_s=float(a.pause),
        no_gui=bool(a.no_gui),
        out_dir=a.out_dir,
        prefix=prefix,
        save_frames=bool(a.save_frames),
        gif=bool(a.gif),
        gif_fps=int(a.gif_fps),
    )


def main() -> int:
    params = parse_args()
    os.makedirs(params.out_dir, exist_ok=True)
    rng = np.random.default_rng(params.seed)

    sigma2 = params.sigma * params.sigma
    rc = params.rc * params.sigma
    rc2 = rc * rc
    a = params.softcore_a * params.sigma
    a2 = a * a

    v_shift = lj_potential_shift(params.epsilon, sigma2, rc2)

    if params.scenario == "collision2":
        positions, velocities = init_collision2(params.box, params.sigma, params.r0, params.vmax)
    else:
        positions, velocities = init_gas(params.n, params.box, params.vmax, rng)

    # If requested, switch to a non-interactive backend to avoid Tk issues and speed things up.
    if params.no_gui:
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    fig = ax = scat = line = None
    window_closed = {"closed": False}

    # Create the figure when interactive, or when saving frames (even in headless mode).
    if (not params.no_gui) or params.save_frames:
        fig, ax, scat, line = setup_plot(params.box, positions)

        if not params.no_gui:
            def _on_close(_event):
                window_closed["closed"] = True

            fig.canvas.mpl_connect("close_event", _on_close)

    frames_dir = os.path.join(params.out_dir, f"{params.prefix}_frames")
    frame_idx = 0
    if params.save_frames:
        os.makedirs(frames_dir, exist_ok=True)

    forces, r_min, Epot, Fmax = compute_forces(
        positions, params.box, params.epsilon, sigma2, rc2, v_shift, a2
    )
    Etot0 = 0.5 * params.m * float(np.sum(velocities**2)) + Epot

    t = 0.0
    t_hist: list[float] = []
    dEtot_hist: list[float] = []

    record_every = max(1, params.steps_per_frame * params.plot_every)

    try:
        for step in range(params.steps):
            if window_closed["closed"]:
                break

            if params.mode == "nve":
                forces, r_min, Epot, Fmax = advance_nve_fixed_dt(
                    params.dt,
                    positions,
                    velocities,
                    forces,
                    params.box,
                    params.m,
                    params.epsilon,
                    sigma2,
                    rc2,
                    v_shift,
                    a2,
                )
            else:
                forces, r_min, Epot, Fmax, _substeps = advance_robust_adaptive(
                    params.dt,
                    positions,
                    velocities,
                    forces,
                    Fmax,
                    params.box,
                    params.m,
                    params.epsilon,
                    sigma2,
                    rc2,
                    v_shift,
                    a2,
                    params.dv_target,
                    params.safety,
                    params.max_substeps,
                )
            t += params.dt

            if (step + 1) % record_every == 0:
                Ecin = 0.5 * params.m * float(np.sum(velocities**2))
                Etot = Ecin + Epot
                t_hist.append(t)
                dEtot_hist.append(Etot - Etot0)

                if fig is not None:
                    pause_s = 0.0 if params.no_gui else params.pause_s
                    render(fig, ax, scat, line, positions, t_hist, dEtot_hist, pause_s)

                    if params.save_frames:
                        fig.savefig(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), dpi=140)
                        frame_idx += 1
    except KeyboardInterrupt:
        print("Interrupted by user. Saving energy plot and exiting...")
    finally:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass

    energy_png = os.path.join(params.out_dir, f"{params.prefix}_energy.png")
    save_energy_plot(energy_png, t_hist, dEtot_hist)

    if params.gif and params.save_frames:
        try:
            from PIL import Image
        except Exception:
            print("Could not import pillow. Install 'pillow' or run without --gif.")
            return 0

        frame_paths = [
            os.path.join(frames_dir, name)
            for name in sorted(os.listdir(frames_dir))
            if name.lower().endswith(".png")
        ]
        if not frame_paths:
            return 0

        images = [Image.open(p).convert("P", palette=Image.Palette.ADAPTIVE) for p in frame_paths]
        gif_path = os.path.join(params.out_dir, f"{params.prefix}.gif")
        duration_ms = int(1000 / max(1, params.gif_fps))
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=duration_ms,
            loop=0,
            optimize=False,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
