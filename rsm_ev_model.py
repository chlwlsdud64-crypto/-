import argparse
import csv
import math
import random
from pathlib import Path


def generate_demo_data(n=40, seed=42):
    random.seed(seed)
    rows = []
    for _ in range(n):
        temp_c = random.uniform(-80, 25)
        time_d = random.uniform(1, 90)

        # synthetic EV behavior for demonstration
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
        rows.append({
            "temp_c": temp_c,
            "time_d": time_d,
            "particle_n": particle_n,
            "pdi": pdi,
            "zeta_mV": zeta_mV,
            "rna_ng_uL": rna_ng_uL,
        })
    return rows


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        return [{k: float(v) for k, v in row.items()} for row in rdr]


def design_vector(temp_c, time_d):
    return [1.0, temp_c, time_d, temp_c * temp_c, time_d * time_d, temp_c * time_d]


def matmul(A, B):
    r, c, n = len(A), len(B[0]), len(B)
    out = [[0.0 for _ in range(c)] for _ in range(r)]
    for i in range(r):
        for k in range(n):
            aik = A[i][k]
            for j in range(c):
                out[i][j] += aik * B[k][j]
    return out


def transpose(A):
    return [list(x) for x in zip(*A)]


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


def fit_quadratic(rows, y_col):
    X = [design_vector(r["temp_c"], r["time_d"]) for r in rows]
    y = [r[y_col] for r in rows]

    Xt = transpose(X)
    XtX = matmul(Xt, X)
    Xty = [sum(Xt[i][k] * y[k] for k in range(len(y))) for i in range(len(Xt))]

    beta = solve_linear_system(XtX, Xty)

    y_mean = sum(y) / len(y)
    ss_tot = sum((v - y_mean) ** 2 for v in y)
    ss_res = 0.0
    for i, row in enumerate(rows):
        pred = predict_row(beta, row["temp_c"], row["time_d"])
        ss_res += (y[i] - pred) ** 2
    r2 = 1.0 - (ss_res / ss_tot if ss_tot else 0.0)

    return beta, r2


def predict_row(beta, temp_c, time_d):
    x = design_vector(temp_c, time_d)
    return sum(beta[i] * x[i] for i in range(len(beta)))


def write_predictions(pred_rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        fields = list(pred_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(pred_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="", help="Optional CSV input")
    parser.add_argument("--output", type=str, default="results/rsm_predictions.csv")
    parser.add_argument("--report", type=str, default="results/rsm_report.txt")
    args = parser.parse_args()

    if args.input:
        rows = read_csv_rows(args.input)
        source = f"input file: {args.input}"
    else:
        rows = generate_demo_data()
        source = "synthetic demo data"

    targets = ["particle_n", "pdi", "zeta_mV", "rna_ng_uL"]
    models = {}
    for t in targets:
        models[t] = fit_quadratic(rows, t)

    grid_temps = [-80, -20, 4, 25]
    grid_times = [1, 7, 30, 90]
    pred_rows = []
    for t in grid_temps:
        for d in grid_times:
            row = {"temp_c": t, "time_d": d}
            for target in targets:
                beta, _ = models[target]
                row[f"pred_{target}"] = predict_row(beta, t, d)
            pred_rows.append(row)

    outp = Path(args.output)
    rept = Path(args.report)
    outp.parent.mkdir(parents=True, exist_ok=True)
    rept.parent.mkdir(parents=True, exist_ok=True)

    write_predictions(pred_rows, outp)

    names = ["Intercept", "temp_c", "time_d", "temp_c^2", "time_d^2", "temp_c:time_d"]
    with open(rept, "w", encoding="utf-8") as f:
        f.write("EV RSM Model Report\n")
        f.write("===================\n")
        f.write(f"Data source: {source}\n")
        f.write(f"Rows: {len(rows)}\n\n")
        for target in targets:
            beta, r2 = models[target]
            f.write(f"[{target}]\n")
            f.write(f"R-squared: {r2:.4f}\n")
            for n, b in zip(names, beta):
                f.write(f"  {n:12s}: {b:.8g}\n")
            f.write("\n")
        f.write("Prediction grid (first 8 rows)\n")
        f.write("temp,time,particle,pdi,zeta,rna\n")
        for r in pred_rows[:8]:
            f.write(
                f"{r['temp_c']},{r['time_d']},{r['pred_particle_n']:.6g},{r['pred_pdi']:.6g},{r['pred_zeta_mV']:.6g},{r['pred_rna_ng_uL']:.6g}\n"
            )

    print(f"Saved predictions: {outp}")
    print(f"Saved report: {rept}")
    print("\nSample predictions:")
    for r in pred_rows[:8]:
        print(
            f"temp={r['temp_c']:>5}, time={r['time_d']:>3} -> "
            f"particle={r['pred_particle_n']:.4e}, pdi={r['pred_pdi']:.4f}, "
            f"zeta={r['pred_zeta_mV']:.3f}, rna={r['pred_rna_ng_uL']:.3f}"
        )


if __name__ == "__main__":
    main()
