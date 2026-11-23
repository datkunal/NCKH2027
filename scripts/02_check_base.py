"""Kiểm tra bộ dữ liệu nền trong data/interim/base/.

    python scripts/02_check_base.py          kiểm tra cấu trúc và đối chiếu với gso_extracted.xlsx
    python scripts/02_check_base.py --pdf    thêm đối chiếu ngẫu nhiên 240 ô với 3 bản PDF gốc
"""
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "interim" / "base"
EXTRACTED = ROOT / "data" / "interim" / "extracted" / "gso_extracted.xlsx"
PDF = {
    "GSO 2015-2018": (ROOT / "data" / "raw" / "GSO_Sach_63_tinh_2015_18.pdf", range(2015, 2019)),
    "GSO 2015-2021": (ROOT / "data" / "raw" / "GSO_Sach_63_tinh_2016_21.pdf", range(2015, 2022)),
    "GSO 2020-2024": (ROOT / "data" / "raw" / "GSO_Sach_63_tinh_2020_24.pdf", range(2020, 2025)),
}
SOURCE_LABEL = {"Sách 2015-2018": "GSO 2015-2018", "Sách 2016-2021": "GSO 2015-2021",
                "Sách 2020-2024": "GSO 2020-2024"}

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(("ĐẠT  " if ok else "LỖI  ") + name + (f"  ({detail})" if detail else ""))


def main(do_pdf):
    prov = pd.read_csv(DATA / "provinces.csv")
    ind = pd.read_csv(DATA / "indicators.csv")
    obs = pd.read_csv(DATA / "observations.csv")
    pan = pd.read_csv(DATA / "panel_wide.csv")

    # 1. Cấu trúc và khóa ---------------------------------------------------
    check("provinces: 63 tỉnh, mã không trùng", len(prov) == 63 and prov.province_id.is_unique)
    check("provinces: số tỉnh theo vùng = 11/14/14/5/6/13",
          prov.region.value_counts().to_dict() == {"NMM": 14, "NCC": 14, "MRD": 13, "RRD": 11, "SE": 6, "CH": 5},
          str(prov.region.value_counts().to_dict()))
    check("provinces: 5 thành phố trực thuộc trung ương", prov.is_central_city.sum() == 5)
    check("indicators: mã chỉ tiêu không trùng", ind.indicator_id.is_unique, f"{len(ind)} chỉ tiêu")
    key = ["indicator_id", "province_id", "year"]
    check("observations: khóa (chỉ tiêu, tỉnh, năm) không trùng", not obs.duplicated(key).any())
    check("observations: mọi mã chỉ tiêu có trong indicators", obs.indicator_id.isin(ind.indicator_id).all())
    check("observations: mọi mã tỉnh có trong provinces", obs.province_id.isin(prov.province_id).all())
    check("observations: năm trong 2015–2024", obs.year.between(2015, 2024).all())
    check("observations: cờ chỉ nhận 0/1",
          obs.is_preliminary.isin([0, 1]).all() and obs.series_break.isin([0, 1]).all())
    check("observations: ô không có số đều có ký hiệu ghi chú",
          obs[obs.value.isna()].value_note.notna().all(), f"{obs.value.isna().sum()} ô")
    check("panel_wide: 630 dòng = 63 tỉnh × 10 năm, không trùng",
          len(pan) == 630 and not pan.duplicated(["province_id", "year"]).any())

    # 2. Đối chiếu với gso_extracted.xlsx ---------------------------------------
    print("Đọc gso_extracted.xlsx ...")
    raw = pd.read_excel(EXTRACTED, sheet_name="Dữ liệu gộp (dạng dài)", dtype=str)
    r = pd.DataFrame({
        "indicator_id": raw["Indicator_ID"], "province_id": raw["Ma_tinh_chinh_thuc"].astype(int),
        "year": raw["Năm"].astype(int),
        "raw_value": pd.to_numeric(raw["Giá trị (chuẩn hóa)"], errors="coerce"),
        "raw_prelim": raw["Sơ bộ"].notna().astype(int),
        "raw_source": raw["Nguồn được chọn"].map(SOURCE_LABEL),
        "raw_text": raw["Giá trị gốc (nguồn chọn)"], "page": raw["Trang PDF (nguồn chọn)"],
    })
    check("observations: số dòng bằng file trích xuất", len(obs) == len(r), f"{len(obs):,} / {len(r):,}")
    m = obs.merge(r, on=key, how="outer", indicator=True)
    check("observations: mọi khóa khớp với file trích xuất", (m._merge == "both").all())
    same = np.isclose(m.value, m.raw_value, rtol=1e-12, atol=0) | (m.value.isna() & m.raw_value.isna())
    check("observations: mọi giá trị khớp với file trích xuất", same.all(), f"{(~same).sum()} ô lệch")
    check("observations: cờ sơ bộ và nguồn khớp với file trích xuất",
          (m.is_preliminary == m.raw_prelim).all() and (m.source == m.raw_source).all())

    panel_raw = pd.read_excel(EXTRACTED, sheet_name="Panel chọn (dạng rộng)", dtype={"Ma_bang_tinh": str})
    panel_raw.columns = [str(c) for c in panel_raw.columns]
    tbl2id = dict(zip(prov.table_no.astype(str).str.zfill(2), prov.province_id))
    pl = panel_raw.melt(id_vars=["Indicator_ID", "Ma_bang_tinh"], value_vars=[str(y) for y in range(2015, 2025)],
                        var_name="year", value_name="v")
    pl["province_id"] = pl.Ma_bang_tinh.map(tbl2id)
    pl["year"] = pl.year.astype(int)
    pl["v"] = pd.to_numeric(pl.v, errors="coerce")
    pmap = ind.dropna(subset=["panel_var"]).set_index("panel_var").indicator_id.to_dict()
    bad = 0
    total = 0
    for var, iid in pmap.items():
        a = pan.set_index(["province_id", "year"])[var]
        b = pl[pl.Indicator_ID == iid].set_index(["province_id", "year"]).v.reindex(a.index)
        c = obs[obs.indicator_id == iid].set_index(["province_id", "year"]).value.reindex(a.index)
        ok = (np.isclose(a, b) | (a.isna() & b.isna())) & (np.isclose(a, c) | (a.isna() & c.isna()))
        bad += int((~ok).sum())
        total += len(a)
    check(f"panel_wide: {len(pmap)} biến khớp với observations và với sheet panel của file trích xuất",
          bad == 0, f"{total:,} ô, {bad} ô lệch")

    # 3. Biến phái sinh --------------------------------------------------------
    d = pan
    recompute = {
        "grdp_pc": d.grdp_current / d.population,
        "labor_productivity": d.grdp_real / d.employed,
        "grdp_deflator": d.grdp_current / d.grdp_real * 100,
        "urbanization_rate": d.population_urban / d.population * 100,
        "invest_fdi_to_grdp": d.invest_fdi / d.grdp_current * 100,
        "internet_per_100": d.internet_subs / d.population * 100,
        "students_per_teacher": d.students * 1000 / d.teachers,
    }
    ok = all(np.allclose(d[k], v, atol=1e-3, equal_nan=True) for k, v in recompute.items())
    check("panel_wide: biến phái sinh tính đúng công thức", ok)
    check("panel_wide: tỷ lệ đô thị hóa trong khoảng 0–100",
          d.urbanization_rate.dropna().between(0, 100).all())
    check("panel_wide: cờ post2020, covid đúng theo năm",
          (d.post2020 == (d.year >= 2020)).all() and (d.covid == d.year.isin([2020, 2021])).all())

    # 4. Giá trị đối chiếu tay với PDF (Hà Nội, province_id = 1) ----------------
    ref = {  # chỉ tiêu: {năm: giá trị in trong PDF}
        "CT0049": {2015: 672949, 2016: 599178, 2017: 843508, 2018: 883102, 2019: 973363,
                   2020: 1017596, 2021: 1069078, 2022: 1190460, 2023: 1290576, 2024: 1425521},
        "CT0054": {2020: 689258, 2021: 709493, 2022: 773057, 2023: 823436, 2024: 877138},
        "CT0070": {2015: 225528, 2016: 179054, 2017: 212547, 2018: 246539, 2019: 270019,
                   2020: 389454, 2021: 436089, 2022: 511133, 2023: 549294, 2024: 709426},
        "CT0071": {2020: 287572, 2021: 324133, 2022: 333756, 2023: 411498, 2024: 513799},
        "CT0076": {2015: 145129, 2016: 160171, 2017: 192008, 2018: 226050, 2019: 247765,
                   2020: 266749, 2021: 299550, 2022: 303912, 2023: 381258, 2024: 477178},
    }
    hn = obs[obs.province_id == 1].set_index(["indicator_id", "year"]).value
    miss = [(i, y) for i, ys in ref.items() for y, v in ys.items() if hn.get((i, y)) != v]
    check("Hà Nội: GRDP, thu ngân sách, thu nội địa khớp số in trong PDF", not miss, str(miss) if miss else "")

    # 5. Đối chiếu ngẫu nhiên với PDF -------------------------------------------
    if do_pdf:
        pdf_spotcheck(obs, r, ind)

    n_fail = sum(not ok for _, ok, _ in results)
    print(f"\nTổng: {len(results)} kiểm tra, {n_fail} lỗi")
    return n_fail


def pdf_spotcheck(obs, r, ind, per_source=80, seed=2027):
    import pdfplumber
    m = obs.merge(r[["indicator_id", "province_id", "year", "raw_text", "page"]],
                  on=["indicator_id", "province_id", "year"])
    m = m[m.value.notna() & m.page.notna() & m.raw_text.str.match(r"^-?[\d.,]+$", na=False)]
    core = set(ind.dropna(subset=["panel_var"]).indicator_id)
    parts = []
    for src in PDF:
        s = m[m.source == src]
        parts.append(s[s.indicator_id.isin(core)].sample(per_source // 2, random_state=seed))
        parts.append(s[~s.indicator_id.isin(core)].sample(per_source // 2, random_state=seed))
    sample = pd.concat(parts)

    def norm(x):
        return str(x).replace(".", "").replace(",", "")

    out = []
    for src, (path, years) in PDF.items():
        with pdfplumber.open(path) as pdf:
            for _, row in sample[sample.source == src].iterrows():
                page = pdf.pages[int(float(row.page)) - 1]
                words = page.extract_words()
                top = page.height * 0.2
                hdr = {int(w["text"]): w["x1"] for w in words
                       if w["top"] < top and re.fullmatch(r"20\d\d", w["text"]) and int(w["text"]) in years}
                cands = [w for w in words if w["top"] >= page.height * 0.12 and norm(w["text"]) == norm(row.raw_text)]
                found_years = sorted({min(hdr, key=lambda y: abs(hdr[y] - w["x1"])) for w in cands}) if hdr else []
                out.append({
                    "indicator_id": row.indicator_id, "province_id": row.province_id, "year": row.year,
                    "value": row.value, "value_in_pdf": row.raw_text, "source": src,
                    "pdf_file": path.name, "pdf_page": int(float(row.page)),
                    "found_on_page": int(bool(cands)), "in_year_column": int(row.year in found_years),
                })
    res = pd.DataFrame(out)
    ok = res.in_year_column.sum()
    bad = res[res.in_year_column == 0]
    if len(bad):
        print(bad.to_string(index=False))
    check("Đối chiếu ngẫu nhiên với PDF: giá trị có trên trang và nằm đúng cột năm",
          ok == len(res), f"{ok}/{len(res)} ô")


if __name__ == "__main__":
    sys.exit(1 if main("--pdf" in sys.argv) else 0)
