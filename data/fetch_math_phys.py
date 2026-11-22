# -*- coding: utf-8 -*-
"""Wave 8: deep math + physics.

~75 new math pages, ~80 new physics pages (every major branch), ~12
formula-table pages (wikitables KEPT as pipe-separated text lines — the
default stripper deletes tables, which would gut these), ~40 RU mirrors
for the majors. Idempotent: existing files are skipped.
"""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
MATH = ROOT / "data" / "corpus" / "math_2026"
PHYS = ROOT / "data" / "corpus" / "physics_2026"
for d in (MATH, PHYS):
    d.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}

MATH_EN = [
    # analysis
    "Real_analysis", "Functional_analysis", "Measure_(mathematics)",
    "Lebesgue_integration", "Fourier_series", "Fourier_transform",
    "Harmonic_analysis", "Vector_calculus", "Multivariable_calculus",
    "Integral", "Derivative", "Taylor_series", "Gamma_function",
    "Riemann_zeta_function", "Green's_function",
    # algebra
    "Ring_(mathematics)", "Field_(mathematics)", "Module_(mathematics)",
    "Galois_theory", "Linear_map", "Eigendecomposition_of_a_matrix",
    "Singular_value_decomposition", "Determinant", "Vector_space",
    "Inner_product_space", "Quadratic_form", "Representation_theory",
    "Category_theory", "Abstract_algebra",
    # discrete / applied
    "Graph_theory", "Game_theory", "Information_theory", "Coding_theory",
    "Cryptography", "Control_theory", "Signal_processing",
    "Linear_programming", "Convex_optimization",
    # probability / stats
    "Probability_distribution", "Normal_distribution",
    "Central_limit_theorem", "Law_of_large_numbers", "Bayes'_theorem",
    "Stochastic_process", "Markov_chain", "Poisson_process",
    "Monte_Carlo_method", "Bayesian_inference",
    "Statistical_hypothesis_testing", "Regression_analysis", "Time_series",
    # geometry / topology
    "Differential_geometry", "Manifold", "Riemannian_geometry",
    "Algebraic_geometry", "Projective_geometry", "Non-Euclidean_geometry",
    "Fractal", "Knot_theory", "Metric_space", "Algebraic_topology",
    "Fundamental_group",
    # DE / numerics
    "Partial_differential_equation", "Ordinary_differential_equation",
    "Dynamical_system", "Numerical_linear_algebra", "Finite_element_method",
    "Interpolation", "Numerical_integration",
    # number theory / logic
    "Prime_number", "Modular_arithmetic", "Diophantine_equation",
    "Elliptic_curve", "Analytic_number_theory", "Type_theory",
    "Computability_theory", "Gödel's_incompleteness_theorems",
]

PHYS_EN = [
    # mechanics
    "Lagrangian_mechanics", "Hamiltonian_mechanics", "Rigid_body_dynamics",
    "Angular_momentum", "Moment_of_inertia", "Kepler's_laws_of_planetary_motion",
    "Gravitation", "Continuum_mechanics", "Elasticity_(physics)",
    "Harmonic_oscillator", "Pendulum", "Resonance",
    # electromagnetism
    "Maxwell's_equations", "Electrostatics", "Magnetostatics",
    "Electric_field", "Magnetic_field", "Electromagnetic_radiation",
    "Capacitor", "Inductor", "Alternating_current", "Plasma_(physics)",
    "Magnetohydrodynamics",
    # quantum
    "Schrödinger_equation", "Path_integral_formulation",
    "Quantum_field_theory", "Quantum_electrodynamics",
    "Quantum_chromodynamics", "Standard_Model", "Dirac_equation",
    "Uncertainty_principle", "Spin_(physics)", "Pauli_exclusion_principle",
    "Quantum_entanglement", "Quantum_information", "Qubit",
    "Quantum_computing",
    # thermo / statistical
    "Entropy", "Enthalpy", "Thermodynamic_free_energy", "Heat_transfer",
    "Thermal_conduction", "Phase_transition", "Boltzmann_constant",
    "Partition_function_(statistical_mechanics)", "Brownian_motion",
    "Carnot_cycle", "Heat_engine",
    # condensed matter
    "Electronic_band_structure", "Semiconductor", "Superconductivity",
    "Crystal_structure", "Phonon", "Fermi_energy",
    # nuclear / particle
    "Radioactive_decay", "Nuclear_fission", "Nuclear_fusion",
    "Particle_accelerator", "Particle_detector", "Neutrino", "Higgs_boson",
    "Quark", "Lepton", "Fundamental_interaction",
    # relativity / astro
    "Black_hole", "Gravitational_wave", "Neutron_star", "Supernova",
    "Big_Bang", "Cosmic_microwave_background", "Dark_matter", "Dark_energy",
    "Redshift", "Spacetime", "Lorentz_transformation",
    # optics / waves
    "Laser", "Diffraction", "Interference_(wave_propagation)",
    "Polarization_(waves)", "Refraction", "Acoustics", "Doppler_effect",
    "Standing_wave",
    # fluids
    "Navier–Stokes_equations", "Turbulence", "Reynolds_number",
    "Bernoulli's_principle", "Aerodynamics",
    # general
    "Dimensional_analysis", "Physical_constant", "Planck_constant",
    "Speed_of_light",
]

# formula tables — wikitables KEPT as text
TABLES = [
    "List_of_trigonometric_identities", "List_of_mathematical_series",
    "List_of_limits", "Vector_calculus_identities", "Laplace_transform",
    "List_of_integrals_of_exponential_functions",
    "List_of_integrals_of_trigonometric_functions",
    "List_of_quantum-mechanical_systems_with_analytical_solutions",
    "Table_of_thermodynamic_equations", "List_of_electromagnetism_equations",
    "List_of_equations_in_classical_mechanics",
]

RU_MATH = [
    "Функциональный_анализ", "Ряд_Фурье", "Преобразование_Фурье",
    "Преобразование_Лапласа", "Нормальное_распределение",
    "Центральная_предельная_теорема", "Теорема_Байеса", "Цепь_Маркова",
    "Дифференциальная_геометрия", "Многообразие", "Теория_категорий",
    "Теория_графов", "Теория_информации", "Теория_игр", "Простое_число",
    "Модульная_арифметика", "Эллиптическая_кривая",
]

RU_PHYS = [
    "Лагранжева_механика", "Гамильтонова_механика", "Уравнения_Максвелла",
    "Электростатика", "Уравнение_Шрёдингера", "Квантовая_теория_поля",
    "Стандартная_модель", "Энтропия", "Теплопередача", "Фазовый_переход",
    "Полупроводник", "Сверхпроводимость", "Радиоактивный_распад",
    "Ядерный_реактор", "Нейтрино", "Чёрная_дыра", "Гравитационная_волна",
    "Большой_взрыв", "Тёмная_материя", "Лазер", "Дифракция",
    "Турбулентность", "Число_Рейнольдса", "Эффект_Доплера",
]


def fetch(url: str, timeout: int = 90) -> bytes | None:
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                return None
            wait = 30 if code == 429 else 2 * attempt
            print(f"  fail ({attempt}/5) {url.rsplit('=', 1)[-1][:44]}: "
                  f"{exc} -> wait {wait}s", flush=True)
            time.sleep(wait)
    return None


def strip_wikitext(wt: str, keep_tables: bool = False) -> str:
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    for _ in range(6):
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    if keep_tables:
        # keep table content as text lines: rows/cells -> pipe text
        wt = re.sub(r"\{\|", "\n", wt)
        wt = re.sub(r"\|\}", "\n", wt)
        wt = re.sub(r"^\|-.*$", "", wt, flags=re.M)
        wt = re.sub(r"\|\|", " | ", wt)
        wt = re.sub(r"^\|", "  ", wt, flags=re.M)
        wt = re.sub(r"^!", "  ", wt, flags=re.M)
    else:
        wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)
    wt = re.sub(r"<![^>]*-->", "", wt)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"'''?", "", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ]+", "_", s).strip("_").lower()[:80]


def pull(lang: str, page: str, out_dir: Path, tag: str, keep: bool) -> bool:
    out = out_dir / f"{tag}.md"
    if out.exists():
        return True
    url = (f"https://{lang}.wikipedia.org/w/index.php?title="
           + urllib.parse.quote(page.replace(" ", "_")) + "&action=raw")
    data = fetch(url)
    time.sleep(1.2)
    if not data:
        return False
    text = strip_wikitext(data.decode("utf-8", errors="ignore"),
                          keep_tables=keep)
    if len(text) < 400:
        return False
    out.write_text(text, encoding="utf-8")
    print(f"[{lang}] {page}: {len(text)//1024} KB", flush=True)
    return True


def main() -> int:
    t0 = time.time()
    ok = miss = 0
    jobs = (
        [("en", p, MATH, f"math2_{slug(p)}", False) for p in MATH_EN]
        + [("en", p, PHYS, f"phys2_{slug(p)}", False) for p in PHYS_EN]
        + [("en", p, PHYS, f"tbl_{slug(p)}", True) for p in TABLES]
        + [("ru", p, MATH, f"rumath_{slug(p)}", False) for p in RU_MATH]
        + [("ru", p, PHYS, f"ruphys_{slug(p)}", False) for p in RU_PHYS]
    )
    for lang, page, out_dir, tag, keep in jobs:
        if pull(lang, page, out_dir, tag, keep):
            ok += 1
        else:
            miss += 1
    print(f"DONE ok={ok} miss={miss} elapsed={int(time.time()-t0)}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
