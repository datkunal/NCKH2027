"""Tạo bộ dữ liệu nền (base) từ data/interim/extracted/gso_extracted.xlsx.

Đầu ra (thư mục data/interim/base/):
    provinces.csv     danh mục 63 tỉnh, thành phố
    indicators.csv    danh mục chỉ tiêu
    observations.csv  số liệu dạng dài: chỉ tiêu x tỉnh x năm
    panel_wide.csv    bảng panel tỉnh x năm với các biến đã chọn

Chạy:  python scripts/01_build_base.py
"""
from pathlib import Path
import unicodedata

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXTRACTED = ROOT / "data" / "interim" / "extracted" / "gso_extracted.xlsx"
OUT = ROOT / "data" / "interim" / "base"
YEARS = list(range(2015, 2025))

SOURCE_LABEL = {
    "Sách 2015-2018": "GSO 2015-2018",
    "Sách 2016-2021": "GSO 2015-2021",
    "Sách 2020-2024": "GSO 2020-2024",
}
SOURCE_RANK = {"GSO 2015-2018": 1, "GSO 2015-2021": 2, "GSO 2020-2024": 3}

VALUE_NOTE = {
    'Dấu "-" (không phát sinh / không có số liệu)': "-",
    'Dấu "…" (chưa có số liệu)': "…",
    'Giá trị dạng khoảng ("<10")': "<10",
    'Giá trị dạng khoảng ("<10,0")': "<10",
}

# --------------------------------------------------------------------------
# Vùng kinh tế - xã hội (6 vùng, theo phân vùng của Tổng cục Thống kê)
# --------------------------------------------------------------------------
REGIONS = {
    "RRD": ("Đồng bằng sông Hồng", "Red River Delta",
            ["Hà Nội", "Vĩnh Phúc", "Bắc Ninh", "Quảng Ninh", "Hải Dương", "Hải Phòng",
             "Hưng Yên", "Thái Bình", "Hà Nam", "Nam Định", "Ninh Bình"]),
    "NMM": ("Trung du và miền núi phía Bắc", "Northern Midlands and Mountains",
            ["Hà Giang", "Cao Bằng", "Bắc Kạn", "Tuyên Quang", "Lào Cai", "Yên Bái",
             "Thái Nguyên", "Lạng Sơn", "Bắc Giang", "Phú Thọ", "Điện Biên", "Lai Châu",
             "Sơn La", "Hòa Bình"]),
    "NCC": ("Bắc Trung Bộ và duyên hải miền Trung", "North Central and Central Coast",
            ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Bình", "Quảng Trị", "Thừa Thiên - Huế",
             "Đà Nẵng", "Quảng Nam", "Quảng Ngãi", "Bình Định", "Phú Yên", "Khánh Hòa",
             "Ninh Thuận", "Bình Thuận"]),
    "CH": ("Tây Nguyên", "Central Highlands",
           ["Kon Tum", "Gia Lai", "Đắk Lắk", "Đắk Nông", "Lâm Đồng"]),
    "SE": ("Đông Nam Bộ", "Southeast",
           ["Bình Phước", "Tây Ninh", "Bình Dương", "Đồng Nai", "Bà Rịa - Vũng Tàu",
            "TP. Hồ Chí Minh"]),
    "MRD": ("Đồng bằng sông Cửu Long", "Mekong River Delta",
            ["Long An", "Tiền Giang", "Bến Tre", "Trà Vinh", "Vĩnh Long", "Đồng Tháp",
             "An Giang", "Kiên Giang", "Cần Thơ", "Hậu Giang", "Sóc Trăng", "Bạc Liêu",
             "Cà Mau"]),
}
CENTRAL_CITIES = {"Hà Nội", "Hải Phòng", "Đà Nẵng", "TP. Hồ Chí Minh", "Cần Thơ"}

# --------------------------------------------------------------------------
# Biến đưa vào panel_wide: (tên cột, mã chỉ tiêu)
# --------------------------------------------------------------------------
PANEL_VARS = [
    ("area_total", "CT0008"),
    ("population", "CT0017"), ("population_urban", "CT0020"), ("population_density", "CT0033"),
    ("sex_ratio", "CT0022"), ("birth_rate", "CT0023"), ("death_rate", "CT0024"),
    ("tfr", "CT0026"), ("infant_mortality", "CT0027"),
    ("in_migration_rate", "CT0029"), ("out_migration_rate", "CT0030"),
    ("life_expectancy", "CT0031"), ("literacy_rate", "CT0032"),
    ("labor_force", "CT0034"), ("employed", "CT0035"),
    ("employed_agri", "CT0036"), ("employed_ind", "CT0037"), ("employed_serv", "CT0038"),
    ("trained_labor_rate", "CT0039"), ("unemployment_rate", "CT0041"),
    ("underemployment_rate", "CT0043"),
    ("grdp_current", "CT0049"), ("grdp_current_agri", "CT0050"),
    ("grdp_current_ind", "CT0051"), ("grdp_current_serv", "CT0052"),
    ("grdp_current_tax", "CT0053"),
    ("grdp_real", "CT0054"), ("grdp_real_agri", "CT0055"),
    ("grdp_real_ind", "CT0056"), ("grdp_real_serv", "CT0057"),
    ("grdp_growth_index", "CT0064"),
    ("budget_rev_domestic", "CT0076"), ("budget_exp_dev_invest", "CT0209"),
    ("social_ins_members", "CT0347"), ("health_ins_members", "CT0348"),
    ("unemp_ins_members", "CT0349"),
    ("invest_total", "CT0375"), ("invest_state", "CT0376"),
    ("invest_nonstate", "CT0377"), ("invest_fdi", "CT0378"),
    ("fdi_projects", "CT0383"), ("fdi_registered_usd", "CT0384"),
    ("firms_new", "CT0425"), ("firms_active", "CT0426"), ("firm_workers", "CT0444"),
    ("firm_capital", "CT0447"), ("firm_avg_income", "CT0452"), ("firm_profit", "CT0453"),
    ("coops", "CT0455"), ("household_biz", "CT0457"), ("household_biz_workers", "CT0458"),
    ("industrial_production_index", "CT0844"),
    ("retail_goods", "CT1153"), ("retail_accom_food", "CT1154"),
    ("passengers", "CT1323"), ("freight", "CT1327"),
    ("housing_floor_area", "CT0386"),
    ("fishery_output", "CT0818"), ("timber_output", "CT0816"),
    ("fixed_phone_subs", "CT1343"), ("mobile_subs", "CT1344"), ("internet_subs", "CT1345"),
    ("preschool_students", "CT1352"), ("schools", "CT1353"), ("classes", "CT1361"),
    ("teachers", "CT1365"), ("students", "CT1371"),
    ("hospitals", "CT1400"), ("hospital_beds", "CT1407"), ("doctors", "CT1412"),
    ("doctors_per_10k", "CT1433"), ("beds_per_10k", "CT1434"),
    ("vaccination_rate", "CT1435"),
    ("income_per_capita_month", "CT1457"), ("poverty_rate", "CT1458"),
]

# Biến phái sinh: (tên, công thức, đơn vị, mô tả)
DERIVED = [
    ("grdp_pc", lambda d: d.grdp_current / d["population"], "Triệu đồng/người",
     "GRDP bình quân đầu người, giá hiện hành = grdp_current / population"),
    ("grdp_real_pc", lambda d: d.grdp_real / d["population"], "Triệu đồng/người",
     "GRDP bình quân đầu người, giá so sánh 2010 = grdp_real / population"),
    ("labor_productivity", lambda d: d.grdp_real / d.employed, "Triệu đồng/người",
     "Năng suất lao động, giá so sánh 2010 = grdp_real / employed"),
    ("grdp_deflator", lambda d: d.grdp_current / d.grdp_real * 100, "2010 = 100",
     "Chỉ số giảm phát GRDP = grdp_current / grdp_real x 100"),
    ("urbanization_rate", lambda d: d.population_urban / d["population"] * 100, "%",
     "Tỷ lệ dân số thành thị = population_urban / population x 100"),
    ("invest_to_grdp", lambda d: d.invest_total / d.grdp_current * 100, "%",
     "Vốn đầu tư thực hiện / GRDP giá hiện hành x 100"),
    ("invest_fdi_to_grdp", lambda d: d.invest_fdi / d.grdp_current * 100, "%",
     "Vốn đầu tư thực hiện khu vực FDI / GRDP giá hiện hành x 100"),
    ("budget_rev_domestic_to_grdp", lambda d: d.budget_rev_domestic / d.grdp_current * 100, "%",
     "Thu nội địa / GRDP giá hiện hành x 100"),
    ("budget_exp_dev_invest_to_grdp", lambda d: d.budget_exp_dev_invest / d.grdp_current * 100, "%",
     "Chi đầu tư phát triển / GRDP giá hiện hành x 100"),
    ("internet_per_100", lambda d: d.internet_subs / d["population"] * 100, "Thuê bao/100 dân",
     "Thuê bao internet trên 100 dân = internet_subs / population x 100"),
    ("students_per_teacher", lambda d: d.students * 1000 / d.teachers, "Học sinh/giáo viên",
     "Học sinh phổ thông trên một giáo viên = students x 1000 / teachers"),
]

def ascii_name(s: str) -> str:
    s = s.replace("Đ", "D").replace("đ", "d")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c))


def tidy_unit(u):
    if pd.isna(u) or str(u).strip() in ("", "-"):
        return np.nan
    u = str(u).strip()
    return u[0].upper() + u[1:]


# --------------------------------------------------------------------------
def load_raw():
    print("Đọc gso_extracted.xlsx ...")
    prov = pd.read_excel(EXTRACTED, sheet_name="Danh mục tỉnh", dtype=str)
    cat = pd.read_excel(EXTRACTED, sheet_name="Danh mục chỉ tiêu", dtype=str)
    long = pd.read_excel(EXTRACTED, sheet_name="Dữ liệu gộp (dạng dài)", dtype=str)
    return prov, cat, long


def build_provinces(prov):
    p = pd.DataFrame({
        "province_id": prov["Ma_tinh_chinh_thuc"].astype(int),
        "table_no": prov["Ma_bang_tinh"].astype(int),
        "name_vi": prov["Tỉnh/Thành phố"],
    })
    p["name_ascii"] = p.name_vi.map(ascii_name).replace({"TP. Ho Chi Minh": "Ho Chi Minh City"})
    lookup = {n: code for code, (_, _, names) in REGIONS.items() for n in names}
    p["region"] = p.name_vi.map(lookup)
    p["region_vi"] = p.region.map({k: v[0] for k, v in REGIONS.items()})
    p["region_en"] = p.region.map({k: v[1] for k, v in REGIONS.items()})
    p["is_central_city"] = p.name_vi.isin(CENTRAL_CITIES).astype(int)
    assert p.region.notna().all(), p[p.region.isna()]
    return p.sort_values("table_no").reset_index(drop=True)


BOOK_CAT_COLS = {
    "GSO 2015-2018": "Sách 2015-2018: đường dẫn gốc | đơn vị | hệ số | số tỉnh",
    "GSO 2015-2021": "Sách 2016-2021: đường dẫn gốc | đơn vị | hệ số | số tỉnh",
    "GSO 2020-2024": "Sách 2020-2024: đường dẫn gốc | đơn vị | hệ số | số tỉnh",
}


def book_factors(cat):
    """Hệ số quy đổi đơn vị của từng ấn phẩm, theo từng chỉ tiêu."""
    out = {}
    for b, col in BOOK_CAT_COLS.items():
        f = pd.to_numeric(cat[col].str.extract(r"×\s*([0-9.eE-]+)")[0], errors="coerce")
        out[b] = pd.Series(f.values, index=cat["Indicator_ID"]).fillna(1.0)
    return out


def build_observations(long, provinces, factors):
    d = pd.DataFrame({
        "indicator_id": long["Indicator_ID"],
        "province_id": long["Ma_tinh_chinh_thuc"].astype(int),
        "year": long["Năm"].astype(int),
        "value": pd.to_numeric(long["Giá trị (chuẩn hóa)"], errors="coerce"),
        "is_preliminary": long["Sơ bộ"].notna().astype(int),
        "source": long["Nguồn được chọn"].map(SOURCE_LABEL),
        "value_note": long["Loai_thieu_du_lieu"].map(VALUE_NOTE),
    })
    assert d.source.notna().all()
    # Giá trị của từng ấn phẩm ở cùng ô (chỉ dùng nội bộ để xác định cờ)
    books = {
        "GSO 2015-2018": "GT gốc sách 2015-2018",
        "GSO 2015-2021": "GT gốc sách 2016-2021",
        "GSO 2020-2024": "GT gốc sách 2020-2024",
    }
    for b, col in books.items():
        raw = pd.to_numeric(long[col].str.replace(",", ".", regex=False), errors="coerce")
        d["_" + b] = raw * d.indicator_id.map(factors[b]).fillna(1.0).values
    d = d.sort_values(["indicator_id", "province_id", "year"]).reset_index(drop=True)
    d["series_break"] = compute_series_break(d)
    d = d.drop(columns=[c for c in d.columns if c.startswith("_")])
    return d[["indicator_id", "province_id", "year", "value", "is_preliminary",
              "series_break", "source", "value_note"]]


def compute_series_break(d):
    """Cờ series_break cho quan sát năm t khi chuỗi gián đoạn từ t-1 sang t.

    (A) Đổi ấn phẩm nguồn giữa t-1 và t, và hai ấn phẩm cho số khác nhau > 5%
        ở cùng năm t (cùng tỉnh, cùng chỉ tiêu) -> chuỗi ghép hai mức khác nhau.
    (B) Đổi ấn phẩm nguồn giữa t-1 và t, và giá trị nhảy >= 5 lần (lệch bậc đơn vị).
    (C) Tốc độ thay đổi từ t-1 sang t bất thường so với các tỉnh khác cùng năm:
        |z| > 5 theo trung vị/MAD của log-tăng trưởng (ít nhất 20 tỉnh có số liệu)
        và thay đổi lớn hơn 25%.
    (D) Riêng GRDP giá so sánh 2010: tăng trưởng tính từ giá trị tuyệt đối lệch
        hơn 5 điểm phần trăm so với chỉ số phát triển GRDP được công bố.
    """
    g = d.groupby(["indicator_id", "province_id"], sort=False)
    prev_year = g.year.shift()
    prev_val = g.value.shift()
    prev_src = g.source.shift()
    consecutive = (d.year - prev_year) == 1
    switch = consecutive & (d.source != prev_src) & prev_src.notna()

    # (A) so sánh giá trị năm t ở ấn phẩm cũ (của năm t-1) với giá trị được chọn
    old_val_t = pd.Series(np.nan, index=d.index)
    for b in SOURCE_RANK:
        m = prev_src == b
        old_val_t[m] = d.loc[m, "_" + b]
    rel = (d.value / old_val_t - 1).abs()
    flag_a = switch & (rel > 0.05)

    # (B) nhảy bậc đơn vị tại điểm đổi nguồn
    ratio = d.value / prev_val
    pos = (d.value > 0) & (prev_val > 0)
    flag_b = switch & pos & ((ratio >= 5) | (ratio <= 0.2))

    # (C) tăng trưởng bất thường so với các tỉnh khác
    lg = pd.Series(np.nan, index=d.index)
    ok = consecutive & pos
    lg[ok] = np.log(ratio[ok])
    tmp = pd.DataFrame({"iid": d.indicator_id, "year": d.year, "lg": lg})
    grp = tmp.groupby(["iid", "year"]).lg
    med = grp.transform("median")
    mad = grp.transform(lambda s: (s - s.median()).abs().median())
    cnt = grp.transform("count")
    z = (lg - med) / (1.4826 * mad)
    flag_c = (cnt >= 20) & (mad > 0) & (z.abs() > 5) & (lg.abs() > np.log(1.25))

    # (D) GRDP giá so sánh 2010: tăng trưởng tính từ giá trị tuyệt đối lệch hơn
    #     5 điểm phần trăm so với chỉ số phát triển GRDP do ấn phẩm công bố
    idx = (d[d.indicator_id == "CT0064"].set_index(["province_id", "year"]).value - 100)
    is_lvl = d.indicator_id == "CT0054"
    growth = (ratio - 1) * 100
    rep = pd.Series([idx.get(k, np.nan) for k in zip(d.province_id, d.year)], index=d.index)
    flag_d = is_lvl & consecutive & ((growth - rep).abs() > 5)

    return (flag_a | flag_b | flag_c | flag_d).astype(int)


def build_indicators(cat, obs, panel_map):
    c = pd.DataFrame({
        "indicator_id": cat["Indicator_ID"],
        "section": cat["Mục"],
        "name_vi": cat["Chỉ tiêu (Tiếng Việt)"],
        "name_en": cat["Indicator (English)"],
        "unit": cat["Đơn vị chuẩn"].map(tidy_unit),
        "path_vi": cat["Đường dẫn chỉ tiêu (theo sách mới nhất có chỉ tiêu)"],
    })
    c["level"] = c.path_vi.str.count("→") + 1
    stats = obs[obs.value.notna()].groupby("indicator_id").agg(
        first_year=("year", "min"), last_year=("year", "max"),
        n_provinces=("province_id", "nunique"), n_obs=("value", "size"),
        n_preliminary=("is_preliminary", "sum"), n_series_break=("series_break", "sum"),
    )
    srcs = obs.groupby("indicator_id").source.nunique().rename("n_sources")
    c = c.join(stats, on="indicator_id").join(srcs, on="indicator_id")
    c["coverage_pct"] = (c.n_obs / (63 * len(YEARS)) * 100).round(1)
    c["panel_var"] = c.indicator_id.map({v: k for k, v in panel_map.items()})
    c["unit_note"] = cat["Ghi chú đơn vị / cảnh báo"]
    for col in ["first_year", "last_year", "n_provinces", "n_obs", "n_preliminary",
                "n_series_break", "n_sources"]:
        c[col] = c[col].astype("Int64")
    return c


def build_panel(obs, provinces):
    ids = dict(PANEL_VARS)
    rev = {v: k for k, v in ids.items()}
    sub = obs[obs.indicator_id.isin(rev)].copy()
    sub["pvar"] = sub.indicator_id.map(rev)
    grid = pd.MultiIndex.from_product([provinces.province_id, YEARS],
                                      names=["province_id", "year"]).to_frame(index=False)
    wide = sub.pivot(index=["province_id", "year"], columns="pvar", values="value")
    wide = grid.join(wide, on=["province_id", "year"])
    wide = wide[["province_id", "year"] + [k for k, _ in PANEL_VARS]]
    for name, f, _, _ in DERIVED:
        wide[name] = f(wide).round(4)
    wide = wide.sort_values(["province_id", "year"])
    prev = wide.groupby("province_id").grdp_real.shift()
    wide["grdp_real_growth"] = ((wide.grdp_real / prev - 1) * 100).where(
        wide.year - wide.groupby("province_id").year.shift() == 1).round(4)
    # cờ theo dòng
    wide["post2020"] = (wide.year >= 2020).astype(int)
    wide["covid"] = wide.year.isin([2020, 2021]).astype(int)
    pre = sub[sub.is_preliminary == 1].groupby(["province_id", "year"]).size()
    brk = (sub[sub.series_break == 1].groupby(["province_id", "year"])["pvar"]
           .apply(lambda s: ";".join(sorted(s))))
    key = list(zip(wide.province_id, wide.year))
    wide["has_preliminary"] = [int(k in pre.index) for k in key]
    wide["break_vars"] = [brk.get(k, "") for k in key]
    wide["has_break"] = (wide.break_vars != "").astype(int)
    meta = provinces.set_index("province_id")[["name_ascii", "region"]]
    wide = wide.join(meta, on="province_id").rename(columns={"name_ascii": "province_name"})
    front = ["province_id", "province_name", "region", "year"]
    flags = ["post2020", "covid", "has_preliminary", "has_break", "break_vars"]
    rest = [c for c in wide.columns if c not in front + flags]
    return wide[front + rest + flags].sort_values(["province_id", "year"]).reset_index(drop=True)


# --------------------------------------------------------------------------
def main():
    OUT.mkdir(exist_ok=True)
    prov, cat, long = load_raw()
    provinces = build_provinces(prov)
    obs = build_observations(long, provinces, book_factors(cat))
    panel_map = dict(PANEL_VARS)
    indicators = build_indicators(cat, obs, panel_map)
    panel = build_panel(obs, provinces)

    provinces.to_csv(OUT / "provinces.csv", index=False, encoding="utf-8-sig")
    indicators.to_csv(OUT / "indicators.csv", index=False, encoding="utf-8-sig")
    obs.to_csv(OUT / "observations.csv", index=False, encoding="utf-8-sig")
    panel.to_csv(OUT / "panel_wide.csv", index=False, encoding="utf-8-sig")
    print("provinces:", provinces.shape, "indicators:", indicators.shape,
          "observations:", obs.shape, "panel_wide:", panel.shape)
    print("series_break:", int(obs.series_break.sum()), "is_preliminary:", int(obs.is_preliminary.sum()))


if __name__ == "__main__":
    main()
