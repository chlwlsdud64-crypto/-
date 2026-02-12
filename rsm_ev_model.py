import argparse
import csv
import random
from pathlib import Path

REQUIRED_COLUMNS = ["temp_c", "time_d", "particle_n", "pdi", "zeta_mV", "rna_ng_uL"]


def generate_demo_data(n=40, seed=42):
    """Synthetic data for demo only. Use --input for real RSM."""
    random.seed(seed)
    rows = []
    for _ in range(n):
        temp_c = random.uniform(-80, 25)
        time_d = random.uniform(1, 90)

        particle_n = (
            2.4e11
            - 2.3e8 * (temp_c + 20)
            - 6.0e8 * time_d
            - 1.2e6 * (temp_c + 20) ** 2
            - 1.9e6 * (time_d ** 2)
            + 5.0e6 * (temp_c + 20) * time_d
            + random.gauss(0, 8e9)
        )
        pdi = (
            0.22
            + 0.0009 * (temp_c + 20)
            + 0.0015 * time_d
            + 0.00002 * (temp_c + 20) ** 2
            + 0.000015 * (time_d ** 2)
            + 0.00001 * (temp_c + 20) * time_d
            + random.gauss(0, 0.015)
        )
        zeta_mV = (
            -21
            + 0.07 * (temp_c + 20)
            + 0.08 * time_d
            + 0.0015 * (temp_c + 20) ** 2
            + 0.0008 * (time_d ** 2)
            - 0.001 * (temp_c + 20) * time_d
            + random.gauss(0, 1.2)
        )
        rna_ng_uL = (
            44
            - 0.09 * (temp_c + 20)
            - 0.16 * time_d
            - 0.0012 * (temp_c + 20) ** 2
            - 0.0010 * (time_d ** 2)
            + 0.0008 * (temp_c + 20) * time_d
            + random.gauss(0, 1.5)
        )
        rows.append(
            {
                "temp_c": temp_c,
                "time_d": time_d,
                "particle_n": particle_n,
                "pdi": pdi,
                "zeta_mV": zeta_mV,
                "rna_ng_uL": rna_ng_uL,
            }
        )
    return rows


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in cols]
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(missing)}")
        rows = []
        for row in rdr:
            rows.append({k: float(v) for k, v in row.items() if k in REQUIRED_COLUMNS})
    if len(rows) < 12:
        raise ValueError("Need at least 12 rows for stable 2-factor quadratic RSM fitting")
    return rows


def minmax(rows, key):
    vals = [r[key] for r in rows]
    return min(vals), max(vals)


def build_coded_rows(rows, tmin=None, tmax=None, dmin=None, dmax=None):
    if tmin is None or tmax is None:
        tmin, tmax = minmax(rows, "temp_c")
    if dmin is None or dmax is None:
        dmin, dmax = minmax(rows, "time_d")

    t_center = (tmin + tmax) / 2.0
    d_center = (dmin + dmax) / 2.0
    t_step = (tmax - tmin) / 2.0
    d_step = (dmax - dmin) / 2.0

    if t_step == 0 or d_step == 0:
        raise ValueError("Temperature/time range cannot be zero")

    coded = []
    for r in rows:
        x1 = (r["temp_c"] - t_center) / t_step
        x2 = (r["time_d"] - d_center) / d_step
        new_r = dict(r)
        new_r["x1"] = x1
        new_r["x2"] = x2
        coded.append(new_r)

    coding = {
        "tmin": tmin,
        "tmax": tmax,
        "dmin": dmin,
        "dmax": dmax,
        "t_center": t_center,
        "d_center": d_center,
        "t_step": t_step,
        "d_step": d_step,
    }
    return coded, coding


def design_vector(x1, x2):
    return [1.0, x1, x2, x1 * x1, x2 * x2, x1 * x2]


def transpose(A):
    return [list(x) for x in zip(*A)]


def matmul(A, B):
    r, c, n = len(A), len(B[0]), len(B)
    out = [[0.0 for _ in range(c)] for _ in range(r)]
    for i in range(r):
        for k in range(n):
            aik = A[i][k]
            for j in range(c):
                out[i][j] += aik * B[k][j]
    return out


def solve_linear_system(A, b):
    n = len(A)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]

    for i in range(n):
        pivot = max(range(i, n), key=lambda r: abs(M[r][i]))
        if abs(M[pivot][i]) < 1e-12:
            raise ValueError("Singular matrix")
        M[i], M[pivot] = M[pivot], M[i]

        div = M[i][i]
        for j in range(i, n + 1):
            M[i][j] /= div

        for r in range(n):
            if r != i:
                factor = M[r][i]
                for j in range(i, n + 1):
                    M[r][j] -= factor * M[i][j]

    return [M[i][n] for i in range(n)]


def predict_row(beta, x1, x2):
    x = design_vector(x1, x2)
    return sum(beta[i] * x[i] for i in range(len(beta)))


def fit_quadratic(rows, y_col):
    X = [design_vector(r["x1"], r["x2"]) for r in rows]
    y = [r[y_col] for r in rows]

    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = [sum(Xt[i][k] * y[k] for k in range(len(y))) for i in range(len(Xt))]

    beta = solve_linear_system(XtX, Xty)

    y_mean = sum(y) / len(y)
    ss_tot = sum((v - y_mean) ** 2 for v in y)
    ss_res = 0.0
    for i, row in enumerate(rows):
        pred = predict_row(beta, row["x1"], row["x2"])
        ss_res += (y[i] - pred) ** 2
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)
    return beta, r2


def to_natural_units(x1, x2, coding):
    temp_c = coding["t_center"] + x1 * coding["t_step"]
    time_d = coding["d_center"] + x2 * coding["d_step"]
    return temp_c, time_d


def desirability_max(y, low, high):
    if y <= low:
        return 0.0
    if y >= high:
        return 1.0
    return (y - low) / (high - low)


def desirability_min(y, low, high):
    if y <= low:
        return 1.0
    if y >= high:
        return 0.0
    return (high - y) / (high - low)


def grid_search_optimum(models, response_ranges, n=41):
    best = None
    for i in range(n):
        x1 = -1.0 + (2.0 * i / (n - 1))
        for j in range(n):
            x2 = -1.0 + (2.0 * j / (n - 1))
            p = predict_row(models["particle_n"][0], x1, x2)
            pdi = predict_row(models["pdi"][0], x1, x2)
            z = predict_row(models["zeta_mV"][0], x1, x2)
            rna = predict_row(models["rna_ng_uL"][0], x1, x2)

            dp = desirability_max(p, *response_ranges["particle_n"])
            dr = desirability_max(rna, *response_ranges["rna_ng_uL"])
            dd = desirability_min(pdi, *response_ranges["pdi"])
            # target zeta around -20 (stable region), tolerance window from data range
            dz = 1.0 - min(abs(z + 20.0) / max(1.0, (response_ranges["zeta_mV"][1] - response_ranges["zeta_mV"][0]) / 2.0), 1.0)

            D = (max(dp, 1e-12) * max(dr, 1e-12) * max(dd, 1e-12) * max(dz, 1e-12)) ** 0.25
            candidate = {"x1": x1, "x2": x2, "D": D, "pred_particle_n": p, "pred_pdi": pdi, "pred_zeta_mV": z, "pred_rna_ng_uL": rna}
            if best is None or candidate["D"] > best["D"]:
                best = candidate
    return best


def write_predictions(pred_rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(pred_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pred_rows)


def main():
    parser = argparse.ArgumentParser(description="Two-factor quadratic RSM for EV storage")
    parser.add_argument("--input", type=str, default="", help="CSV input with real measurements")
    parser.add_argument("--output", type=str, default="results/rsm_predictions.csv")
    parser.add_argument("--report", type=str, default="results/rsm_report.txt")
    parser.add_argument("--temp-min", type=float, default=None, help="coding range min temp")
    parser.add_argument("--temp-max", type=float, default=None, help="coding range max temp")
    parser.add_argument("--time-min", type=float, default=None, help="coding range min time")
    parser.add_argument("--time-max", type=float, default=None, help="coding range max time")
    args = parser.parse_args()

    if args.input:
        rows = read_csv_rows(args.input)
        source = f"input file: {args.input}"
    else:
        rows = generate_demo_data()
        source = "synthetic demo data (for pipeline check only)"

    coded_rows, coding = build_coded_rows(rows, args.temp_min, args.temp_max, args.time_min, args.time_max)

    targets = ["particle_n", "pdi", "zeta_mV", "rna_ng_uL"]
    models = {t: fit_quadratic(coded_rows, t) for t in targets}
    response_ranges = {t: minmax(coded_rows, t) for t in targets}

    grid_steps = [-1.0, -0.5, 0.0, 0.5, 1.0]
    pred_rows = []
    for x1 in grid_steps:
        for x2 in grid_steps:
            temp_c, time_d = to_natural_units(x1, x2, coding)
            row = {"x1": x1, "x2": x2, "temp_c": temp_c, "time_d": time_d}
            for t in targets:
                beta, _ = models[t]
                row[f"pred_{t}"] = predict_row(beta, x1, x2)
            pred_rows.append(row)

    optimum = grid_search_optimum(models, response_ranges, n=61)
    opt_temp, opt_time = to_natural_units(optimum["x1"], optimum["x2"], coding)

    outp = Path(args.output)
    rept = Path(args.report)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rept.parent.mkdir(parents=True, exist_ok=True)
    write_predictions(pred_rows, outp)

    names = ["Intercept", "x1", "x2", "x1^2", "x2^2", "x1:x2"]
    with open(rept, "w", encoding="utf-8") as f:
        f.write("EV RSM Model Report\n")
        f.write("===================\n")
        f.write(f"Data source: {source}\n")
        f.write(f"Rows: {len(coded_rows)}\n\n")
        f.write("Coding (natural -> coded):\n")
        f.write(f"x1 = (temp_c - {coding['t_center']:.6g}) / {coding['t_step']:.6g}\n")
        f.write(f"x2 = (time_d - {coding['d_center']:.6g}) / {coding['d_step']:.6g}\n\n")

        for t in targets:
            beta, r2 = models[t]
            f.write(f"[{t}]\n")
            f.write(f"R-squared: {r2:.4f}\n")
            for n, b in zip(names, beta):
                f.write(f"  {n:10s}: {b:.8g}\n")
            f.write("\n")

        f.write("Recommended storage condition by multi-response desirability\n")
        f.write(f"x1={optimum['x1']:.3f}, x2={optimum['x2']:.3f}, D={optimum['D']:.4f}\n")
        f.write(f"temp_c={opt_temp:.3f}, time_d={opt_time:.3f}\n")
        f.write(f"pred_particle_n={optimum['pred_particle_n']:.6g}\n")
        f.write(f"pred_pdi={optimum['pred_pdi']:.6g}\n")
        f.write(f"pred_zeta_mV={optimum['pred_zeta_mV']:.6g}\n")
        f.write(f"pred_rna_ng_uL={optimum['pred_rna_ng_uL']:.6g}\n")

    print(f"Saved predictions: {outp}")
    print(f"Saved report: {rept}")
    print("\nRecommended condition:")
    print(f"  x1={optimum['x1']:.3f}, x2={optimum['x2']:.3f}, D={optimum['D']:.4f}")
    print(f"  temp_c={opt_temp:.3f}, time_d={opt_time:.3f}")


if __name__ == "__main__":
    main()
