"""Hàm EDA dùng chung cho notebooks/01_eda_base.ipynb và notebooks/02_eda_processed.ipynb.

Hai notebook gọi đúng các hàm này, chỉ khác dữ liệu đầu vào:
    load("base")    dữ liệu nền   data/interim/base/
    load("clean")   dữ liệu sạch  data/processed/
Sửa một hàm ở đây thì cả hai notebook cùng thay đổi.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "interim" / "base"
PROCESSED = ROOT / "data" / "processed"

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})
pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 200)

ID_COLS = ["province_id", "province_name", "region", "year"]
FLAG_COLS = ["post2020", "covid", "has_preliminary", "has_break", "break_vars", "corrected_vars"]

# Biến chính cho thống kê mô tả: tên cột → (nhóm, tên hiển thị, đơn vị)
KEY_VARS = {
    "population":                    ("Dân số, lao động", "Dân số", "nghìn người"),
    "population_density":            ("Dân số, lao động", "Mật độ dân số", "người/km²"),
    "urbanization_rate":             ("Dân số, lao động", "Tỷ lệ đô thị hóa", "%"),
    "trained_labor_rate":            ("Dân số, lao động", "Lao động đã qua đào tạo", "%"),
    "unemployment_rate":             ("Dân số, lao động", "Tỷ lệ thất nghiệp", "%"),
    "grdp_pc":                       ("Kinh tế", "GRDP bình quân đầu người", "triệu đồng"),
    "grdp_growth_index":             ("Kinh tế", "Chỉ số phát triển GRDP", "năm trước = 100"),
    "labor_productivity":            ("Kinh tế", "Năng suất lao động (giá 2010)", "triệu đồng/người"),
    "invest_to_grdp":                ("Đầu tư", "Vốn đầu tư / GRDP", "%"),
    "invest_fdi_to_grdp":            ("Đầu tư", "Vốn FDI thực hiện / GRDP", "%"),
    "budget_rev_domestic_to_grdp":   ("Ngân sách", "Thu nội địa / GRDP", "%"),
    "budget_exp_dev_invest_to_grdp": ("Ngân sách", "Chi đầu tư phát triển / GRDP", "%"),
    "doctors_per_10k":               ("Xã hội", "Bác sĩ trên 1 vạn dân", "người"),
    "students_per_teacher":          ("Xã hội", "Học sinh trên một giáo viên", "học sinh"),
    "income_per_capita_month":       ("Xã hội", "Thu nhập bình quân đầu người/tháng", "nghìn đồng"),
    "poverty_rate":                  ("Xã hội", "Tỷ lệ hộ nghèo", "%"),
}

# Ràng buộc thành phần – tổng: (tên, các biến thành phần, biến tổng)
GRDP_PARTS = ["grdp_current_agri", "grdp_current_ind", "grdp_current_serv", "grdp_current_tax"]
EMPLOY_PARTS = ["employed_agri", "employed_ind", "employed_serv"]
INVEST_PARTS = ["invest_state", "invest_nonstate", "invest_fdi"]
PCT_VARS = ["literacy_rate", "trained_labor_rate", "unemployment_rate", "underemployment_rate",
            "vaccination_rate", "poverty_rate", "urbanization_rate"]


# ----------------------------------------------------------------------------- đọc dữ liệu
def load(kind):
    """Đọc dữ liệu. kind = "base" (dữ liệu nền) hoặc "clean" (dữ liệu sạch).

    Trả về dict gồm provinces, indicators, obs, panel và danh sách biến:
    direct (lấy trực tiếp từ ấn phẩm), derived (biến phái sinh), vars (cả hai).
    """
    if kind == "base":
        obs, panel = pd.read_csv(BASE / "observations.csv"), pd.read_csv(BASE / "panel_wide.csv")
    elif kind == "clean":
        obs, panel = (pd.read_csv(PROCESSED / "observations_clean.csv"),
                      pd.read_csv(PROCESSED / "panel_clean.csv"))
    else:
        raise ValueError('kind phải là "base" hoặc "clean"')
    indicators = pd.read_csv(BASE / "indicators.csv")      # danh mục không đổi qua làm sạch
    provinces = pd.read_csv(BASE / "provinces.csv")

    direct = [c for c in panel.columns if c in set(indicators.panel_var.dropna())]
    derived = [c for c in panel.columns if c not in ID_COLS + FLAG_COLS + direct]
    for name, df in [("provinces", provinces), ("indicators", indicators),
                     ("observations", obs), ("panel", panel)]:
        print(f"{name:13s} {df.shape[0]:>7,} dòng × {df.shape[1]:>3} cột")
    print(f"{len(direct)} biến lấy trực tiếp, {len(derived)} biến phái sinh")
    return {"provinces": provinces, "indicators": indicators, "obs": obs, "panel": panel,
            "direct": direct, "derived": derived, "vars": direct + derived}


# ----------------------------------------------------------------------------- cấu trúc
def structure(d):
    """Kiểm tra panel cân bằng; trả về số tỉnh theo vùng."""
    p = d["panel"]
    print("Số tỉnh:", p.province_id.nunique())
    print("Số năm :", p.year.nunique(), f"({p.year.min()}–{p.year.max()})")
    print("Cặp (tỉnh, năm) bị trùng:", p.duplicated(["province_id", "year"]).sum())
    print("Panel cân bằng:", p.groupby("province_id").size().eq(10).all())
    return d["provinces"].groupby(["region", "region_vi"]).size().rename("Số tỉnh").reset_index()


def indicator_coverage(d):
    """Số chỉ tiêu theo mức độ phủ (tỷ lệ ô tỉnh × năm có số liệu)."""
    bins = pd.cut(d["indicators"].coverage_pct, bins=[0, 25, 50, 75, 99.9, 100],
                  labels=["< 25%", "25–50%", "50–75%", "75–99,9%", "đủ 100%"], include_lowest=True)
    return bins.value_counts().sort_index().rename("Số chỉ tiêu")


# ----------------------------------------------------------------------------- thống kê mô tả
def describe(d, variables=KEY_VARS):
    """Bảng thống kê mô tả các biến chính trên toàn bộ panel (tỉnh × năm)."""
    p = d["panel"]
    rows = []
    for var, (group, label, unit) in variables.items():
        x = p[var].dropna()
        rows.append({"Nhóm": group, "Biến": label, "Đơn vị": unit, "N": len(x),
                     "Trung bình": x.mean(), "Độ lệch chuẩn": x.std(), "Nhỏ nhất": x.min(),
                     "Trung vị": x.median(), "Lớn nhất": x.max(), "Độ lệch (skew)": x.skew()})
    return pd.DataFrame(rows).set_index(["Nhóm", "Biến"]).round(2)


def describe_by_period(d, variables=KEY_VARS):
    """Trung bình các biến chính theo hai giai đoạn ứng với hai ấn phẩm nguồn."""
    p = d["panel"]
    period = np.where(p.year <= 2019, "2015–2019", "2020–2024")
    out = p[list(variables)].groupby(period).mean().T
    out.index = [variables[v][1] for v in out.index]
    out["Chênh lệch (%)"] = (out["2020–2024"] / out["2015–2019"] - 1) * 100
    return out.round(2)


def describe_by_region(d, variables=KEY_VARS):
    """Trung bình các biến chính theo vùng (toàn giai đoạn)."""
    p = d["panel"]
    out = p.groupby("region")[list(variables)].mean().T
    out.index = [variables[v][1] for v in out.index]
    return out.round(2)


def _corr_heatmap(corr, labels, method):
    """Vẽ bản đồ nhiệt của ma trận tương quan, ghi giá trị r trong từng ô."""
    fig, ax = plt.subplots(figsize=(10, 8.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.grid(False)
    for i in range(len(labels)):
        for j in range(len(labels)):
            r = corr.iat[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if abs(r) > 0.6 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Hệ số tương quan r")
    ax.set_title(f"Ma trận tương quan giữa các biến chính ({method.capitalize()})")
    plt.tight_layout()
    plt.show()


def correlation(d, variables=KEY_VARS, threshold=0.7, method="pearson", plot=True):
    """Ma trận tương quan giữa các biến chính, vẽ thành bản đồ nhiệt.

    method = "pearson" (tương quan tuyến tính) hoặc "spearman" (tương quan thứ hạng, ít bị ngoại lai chi phối).

    Tính trên mọi quan sát tỉnh × năm, theo từng cặp biến có số liệu (thu nhập, hộ nghèo chỉ có 2020–2024).
    Trả về các cặp biến có |r| ≥ threshold.
    """
    corr = d["panel"][list(variables)].corr(method=method)
    labels = [variables[v][1] for v in corr.columns]

    if plot:
        _corr_heatmap(corr, labels, method)

    # Các cặp tương quan mạnh (chỉ lấy nửa trên của ma trận để không lặp cặp)
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1)).stack()
    strong = upper[upper.abs() >= threshold].sort_values(key=abs, ascending=False)
    return pd.DataFrame({"Biến 1": [variables[a][1] for a, _ in strong.index],
                         "Biến 2": [variables[b][1] for _, b in strong.index],
                         "r": strong.values.round(2)})


def variation(d, variables=KEY_VARS):
    """Phân rã biến thiên của mỗi biến thành phần giữa các tỉnh và phần trong tỉnh (giống lệnh xtsum của Stata).

    - Giữa các tỉnh: độ lệch chuẩn của trung bình 10 năm của từng tỉnh.
    - Trong tỉnh: độ lệch chuẩn của (giá trị năm t − trung bình của chính tỉnh đó).
    - % còn lại sau khi bỏ hiệu ứng cố định tỉnh và năm: phần phương sai còn lại sau khi trừ trung bình tỉnh
      và trung bình năm. Đây là phần biến thiên mà mô hình có hiệu ứng cố định tỉnh + năm dùng để ước lượng.
    """
    p = d["panel"]
    rows = []
    for var, (_, label, _) in variables.items():
        x = p[["province_id", "year", var]].dropna()
        v = x[var]
        within = v - x.groupby("province_id")[var].transform("mean")
        twoway = within - within.groupby(x.year).transform("mean")
        rows.append({"Biến": label, "N": len(v),
                     "Độ lệch chuẩn chung": v.std(),
                     "Giữa các tỉnh": x.groupby("province_id")[var].mean().std(),
                     "Trong tỉnh": within.std(),
                     "% phương sai trong tỉnh": within.var() / v.var() * 100,
                     "% còn lại sau bỏ FE tỉnh và năm": twoway.var() / v.var() * 100})
    return pd.DataFrame(rows).set_index("Biến").round(2)


# ----------------------------------------------------------------------------- thiếu dữ liệu
def missing_by_var(d):
    """Tỷ lệ ô trống theo biến (chỉ biến có ô trống), kèm biểu đồ."""
    miss = d["panel"][d["vars"]].isna().mean().mul(100).sort_values()
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(8, 9))
    miss.plot.barh(ax=ax, color="tab:red")
    ax.set_xlabel("% ô trống (trên 630 dòng)")
    ax.set_title("Tỷ lệ thiếu số liệu theo biến")
    plt.tight_layout()
    plt.show()
    print(f"{len(miss)}/{len(d['vars'])} biến có ô trống")
    return miss


def missing_heatmap(d, variables):
    """Bản đồ nhiệt: số tỉnh thiếu số liệu theo biến × năm."""
    table = d["panel"].groupby("year")[list(variables)].agg(lambda s: s.isna().sum()).T
    fig, ax = plt.subplots(figsize=(9, 9))
    im = ax.imshow(table, aspect="auto", cmap="Reds", vmin=0, vmax=63)
    ax.set_xticks(range(len(table.columns)), table.columns)
    ax.set_yticks(range(len(table.index)), table.index)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Số tỉnh thiếu (tối đa 63)")
    ax.set_title("Số tỉnh thiếu số liệu theo biến và năm")
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------------- cờ chất lượng
def preliminary_by_year(d):
    """Số quan sát là số sơ bộ theo năm (toàn bộ observations)."""
    s = d["obs"].groupby("year").is_preliminary.sum()
    fig, ax = plt.subplots(figsize=(8, 3))
    s.plot.bar(ax=ax, color="tab:orange")
    ax.set_ylabel("Số quan sát")
    ax.set_title("Số quan sát là số sơ bộ theo năm")
    plt.tight_layout()
    plt.show()
    return s.rename("Số quan sát sơ bộ").to_frame().T


def breaks(d, top=20):
    """Cờ series_break: số tỉnh có cờ theo năm và các biến bị gắn cờ nhiều nhất."""
    p = d["panel"]
    print("Số tỉnh có ít nhất một biến bị gắn cờ, theo năm:")
    print(p.groupby("year").has_break.sum().to_dict())
    count = p.break_vars.dropna().str.split(";").explode().value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    count.head(top).sort_values().plot.barh(ax=ax, color="tab:purple")
    ax.set_xlabel("Số cặp (tỉnh, năm) bị gắn cờ")
    ax.set_title(f"{top} biến bị gắn cờ series_break nhiều nhất")
    plt.tight_layout()
    plt.show()
    return count


# ----------------------------------------------------------------------------- tính hợp lý
def negative_values(d):
    """Số ô âm theo biến."""
    neg = (d["panel"][d["vars"]] < 0).sum()
    return neg[neg > 0].rename("Số ô âm").to_frame()


def pct_out_of_range(d):
    """Các dòng có biến tỷ lệ % nằm ngoài khoảng 0–100."""
    p = d["panel"]
    bad = (p[PCT_VARS] < 0).any(axis=1) | (p[PCT_VARS] > 100).any(axis=1)
    return p.loc[bad, ["province_name", "year"] + PCT_VARS]


def _gap(parts, total):
    """Độ lệch tương đối giữa tổng các thành phần và tổng chung (cần đủ mọi thành phần)."""
    return (parts.sum(axis=1, min_count=parts.shape[1]) / total - 1).abs()


def logic_rules(d):
    """Các ràng buộc logic. Trả về dict: tên ràng buộc → mặt nạ True ở dòng VI PHẠM.

    Tổng các thành phần được phép lệch tổng chung tối đa 1% (do làm tròn).
    """
    p = d["panel"]
    return {
        "Dân số thành thị > tổng dân số": p.population_urban > p.population,
        "Lao động có việc làm > lực lượng lao động": p.employed > p.labor_force,
        "Lao động 3 khu vực ≠ tổng lao động (lệch > 1%)": _gap(p[EMPLOY_PARTS], p.employed) > 0.01,
        "4 thành phần GRDP ≠ tổng GRDP (lệch > 1%)": _gap(p[GRDP_PARTS], p.grdp_current) > 0.01,
        "Vốn đầu tư 3 khu vực ≠ tổng vốn đầu tư (lệch > 1%)": _gap(p[INVEST_PARTS], p.invest_total) > 0.01,
        "Tăng trưởng GRDP tự tính lệch chỉ số GSO > 5 điểm %":
            (p.grdp_real_growth - (p.grdp_growth_index - 100)).abs() > 5,
    }


def logic_summary(d):
    """Bảng số dòng vi phạm từng ràng buộc, và in danh sách dòng vi phạm."""
    p, rules = d["panel"], logic_rules(d)
    for name, mask in rules.items():
        if mask.sum():
            rows = ", ".join(f"{a} {b}" for a, b in zip(p.province_name[mask], p.year[mask]))
            print(f"▶ {name}: {rows}")
    return pd.Series({k: int(v.sum()) for k, v in rules.items()}, name="Số dòng vi phạm").to_frame()


# ----------------------------------------------------------------------------- bước nhảy
def jumps(d, threshold=5):
    """Số lần một biến của một tỉnh tăng ≥ threshold lần hoặc giảm còn ≤ 1/threshold so với năm trước.

    Chỉ xét cặp năm mà cả hai giá trị đều dương; bỏ grdp_real_growth vì đã là tốc độ thay đổi.
    """
    variables = [v for v in d["vars"] if v != "grdp_real_growth"]
    p = d["panel"].sort_values(["province_id", "year"])
    prev = p.groupby("province_id")[variables].shift(1)
    ratio = (p[variables] / prev).where((p[variables] > 0) & (prev > 0))
    count = ((ratio >= threshold) | (ratio <= 1 / threshold)).sum()
    return count[count > 0].sort_values(ascending=False).rename("Số lần nhảy").to_frame()


# Biến quy mô (số người, số cơ sở, số thuê bao…): bình thường thay đổi chậm qua các năm,
# nên một năm lệch hẳn so với hai năm lân cận thường là lỗi số liệu.
STOCK_VARS = ["population", "population_urban", "labor_force", "employed", "employed_agri", "employed_ind",
              "employed_serv", "social_ins_members", "health_ins_members", "unemp_ins_members",
              "firms_active", "firm_workers", "coops", "household_biz", "household_biz_workers",
              "fixed_phone_subs", "mobile_subs", "preschool_students", "schools", "classes", "teachers",
              "students", "hospitals", "hospital_beds", "doctors"]


def spikes(d, variables=STOCK_VARS, threshold=0.25, calm=0.15):
    """Đỉnh một năm: giá trị năm t lệch hơn threshold so với trung bình năm t−1 và t+1,
    trong khi năm t−1 và t+1 gần nhau (lệch nhau dưới calm). Chỉ xét biến quy mô."""
    p = d["panel"].sort_values(["province_id", "year"])
    g = p.groupby("province_id")
    rows = []
    for var in variables:
        prev, nxt, x = g[var].shift(1), g[var].shift(-1), p[var]
        neighbors = (prev + nxt) / 2
        hit = (prev > 0) & (nxt > 0) & ((nxt / prev - 1).abs() < calm) & ((x / neighbors - 1).abs() > threshold)
        for i in p.index[hit]:
            rows.append({"Biến": var, "Tỉnh": p.at[i, "province_name"], "Năm": p.at[i, "year"],
                         "Giá trị": x[i], "TB 2 năm lân cận": round(neighbors[i], 1),
                         "Lệch (%)": round((x[i] / neighbors[i] - 1) * 100)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- diễn biến, ngoại lai
def lines(d, var, title, highlight=None, ax=None):
    """Mỗi đường mảnh là một tỉnh, đường đen là trung vị 63 tỉnh; highlight: tên tỉnh (không dấu)."""
    wide = d["panel"].pivot(index="year", columns="province_name", values=var)
    show = ax is None
    if show:
        fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(wide.index, wide.values, color="grey", alpha=0.25, lw=0.8)
    ax.plot(wide.index, wide.median(axis=1), color="black", lw=2.5, label="Trung vị 63 tỉnh")
    if highlight:
        ax.plot(wide.index, wide[highlight], color="tab:red", lw=2, label=highlight)
        ax.legend()
    ax.set_title(title, fontsize=10)
    if show:
        plt.tight_layout()
        plt.show()


def trends(d, variables):
    """Lưới biểu đồ diễn biến 2015–2024; đường đỏ đứt: năm chuyển ấn phẩm nguồn (2020)."""
    n = len(variables)
    fig, axes = plt.subplots((n + 2) // 3, 3, figsize=(13, 3.3 * ((n + 2) // 3)))
    for ax, var in zip(axes.flat, variables):
        lines(d, var, KEY_VARS.get(var, (None, var))[1], ax=ax)
        ax.axvline(2019.5, color="tab:red", ls="--", lw=1)
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    fig.suptitle("Diễn biến 2015–2024 (đường đỏ: chuyển ấn phẩm nguồn năm 2020)")
    plt.tight_layout()
    plt.show()


def top_values(d, variables, n=5):
    """In n giá trị lớn nhất của mỗi biến để xem là lỗi hay thực tế."""
    for var in variables:
        print(f"\n▶ {var}")
        print(d["panel"].nlargest(n, var)[["province_name", "year", var]].to_string(index=False))


# ----------------------------------------------------------------------------- so sánh trước/sau
def compare_describe(before, after, variables):
    """Thống kê mô tả của cùng một biến trước và sau làm sạch."""
    rows = []
    for var in variables:
        for label, d in [("Trước", before), ("Sau", after)]:
            x = d["panel"][var].dropna()
            rows.append({"Biến": var, "Dữ liệu": label, "N": len(x), "Trung bình": x.mean(),
                         "Độ lệch chuẩn": x.std(), "Nhỏ nhất": x.min(), "Lớn nhất": x.max(),
                         "Độ lệch (skew)": x.skew()})
    return pd.DataFrame(rows).set_index(["Biến", "Dữ liệu"]).round(2)


def changed_vars(before, after):
    """Các biến có ít nhất một ô khác nhau giữa hai bộ dữ liệu, kèm số ô khác."""
    a, b = before["panel"][before["vars"]], after["panel"][before["vars"]]
    diff = (a != b) & ~(a.isna() & b.isna())
    count = diff.sum()
    return count[count > 0].rename("Số ô thay đổi").to_frame()


def compare_logic(before, after):
    """Số dòng vi phạm từng ràng buộc logic, trước và sau làm sạch."""
    rb, ra = logic_rules(before), logic_rules(after)
    return pd.DataFrame({"Trước": {k: int(v.sum()) for k, v in rb.items()},
                         "Sau": {k: int(v.sum()) for k, v in ra.items()}})
