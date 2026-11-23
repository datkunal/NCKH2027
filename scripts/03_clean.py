"""Làm sạch bộ dữ liệu nền: data/interim/base/ (panel_wide.csv, observations.csv) → data/processed/.

Nguyên tắc
    - Chỉ sửa khi có bằng chứng: đối chiếu bản PDF và ràng buộc kế toán của chính ấn phẩm.
    - Không sửa dữ liệu nền; kết quả ghi ra file mới, mọi thay đổi đều có nhật ký.
    - Các bất thường là đặc tính thật của dữ liệu (biến động FDI, ngoại lai, gián đoạn năm 2020)
      không sửa ở đây mà xử lý khi thiết kế mô hình (xem notebooks/01_eda_base.ipynb).

Đầu ra (data/processed/)
    panel_clean.csv          bảng panel tỉnh × năm sau làm sạch, thêm cột corrected_vars
    observations_clean.csv   số liệu dạng dài (toàn bộ chỉ tiêu) sau làm sạch, thêm cột is_corrected
    corrections.csv          nhật ký: mỗi dòng là một ô đã sửa, bỏ trống, tính lại hoặc đã xem xét và giữ nguyên

Chạy:  python scripts/03_clean.py
"""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data" / "interim" / "base"
OUT = ROOT / "data" / "processed"

# Công thức biến phái sinh lấy từ scripts/01_build_base.py để hai bước dùng chung một công thức
_spec = importlib.util.spec_from_file_location("build_base", ROOT / "scripts" / "01_build_base.py")
_build_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_build_base)
DERIVED = _build_base.DERIVED

GRDP_PARTS = ["grdp_current_agri", "grdp_current_ind", "grdp_current_serv", "grdp_current_tax"]
EMPLOY_PARTS = ["employed_agri", "employed_ind", "employed_serv"]

# ----------------------------------------------------------------------------- danh sách xử lý
# Rút ra từ notebooks/01_eda_base.ipynb và đối chiếu PDF.
#   action: "sửa"        thay bằng value
#           "bỏ trống"   thay bằng ô trống
#           "giữ nguyên" không đổi, chỉ ghi vào nhật ký để biết đã được xem xét
# Biến phái sinh ở các dòng bị sửa được tính lại tự động (ghi nhật ký với action "tính lại").
FIXES = [
    {
        "province": "Vinh Long", "years": [2024], "var": "grdp_current_serv",
        "action": "sửa", "value": 33206.1,
        "reason": "Ấn phẩm in 332060,1 (thừa một chữ số 0, gấp 4 lần tổng GRDP của tỉnh). "
                  "Giá trị đúng suy ra từ đẳng thức GRDP = 3 khu vực + thuế: "
                  "83678,8 − 32698,1 − 13561,4 − 4213,2 = 33206,1",
        "evidence": "GSO_Sach_63_tinh_2020_24.pdf, tr. 904",
    },
    {
        "province": "Ha Nam", "years": [2015, 2016, 2017, 2018], "var": "grdp_current_tax",
        "action": "bỏ trống", "value": None,
        "reason": "Ấn phẩm 2015-2021 không tách dòng thuế sản phẩm; 3 khu vực của ấn phẩm này đã cộng đủ "
                  "tổng GRDP. Số thuế đang có lấy từ ấn phẩm 2015-2018 nên bị tính trùng (tổng thành phần "
                  "vượt GRDP 6-8%). Ở các năm này thuế đã nằm trong các khu vực của ấn phẩm 2015-2021",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 149; GSO_Sach_63_tinh_2015_18.pdf, tr. 128",
    },
    {
        "province": "Bac Giang", "years": [2016], "var": "grdp_current_tax",
        "action": "bỏ trống", "value": None,
        "reason": "Cùng nguyên nhân như Hà Nam: thuế lấy từ ấn phẩm 2015-2018 trong khi 3 khu vực của "
                  "ấn phẩm 2015-2021 đã cộng đủ tổng GRDP",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 330",
    },
    {
        "province": "Gia Lai", "years": [2019], "var": "fixed_phone_subs",
        "action": "sửa", "value": 33.0,
        "reason": "Ấn phẩm 2015-2021 in đổi chỗ dòng thuê bao cố định và di động (cố định 1323, di động 33). "
                  "Ấn phẩm 2020-2024 xác nhận thứ tự đúng (năm 2020: cố định 31,5, di động 1355,1). Đổi lại chỗ",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 689; GSO_Sach_63_tinh_2020_24.pdf, tr. 689",
    },
    {
        "province": "Gia Lai", "years": [2019], "var": "mobile_subs",
        "action": "sửa", "value": 1323.0,
        "reason": "Đổi chỗ với thuê bao cố định (xem dòng trên)",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 689; GSO_Sach_63_tinh_2020_24.pdf, tr. 689",
    },
    {
        "province": "Dien Bien", "years": [2023], "var": "schools",
        "action": "bỏ trống", "value": None,
        "reason": "Ấn phẩm in 5173 trường trong khi các năm lân cận 295-296; không suy ra được giá trị đúng",
        "evidence": "GSO_Sach_63_tinh_2020_24.pdf, tr. 382",
    },
    {
        "province": "Binh Phuoc", "years": [2023], "var": "schools",
        "action": "bỏ trống", "value": None,
        "reason": "Ấn phẩm in 6442 trường trong khi các năm lân cận 265-267; không suy ra được giá trị đúng",
        "evidence": "GSO_Sach_63_tinh_2020_24.pdf, tr. 753",
    },
    {
        "province": "Ho Chi Minh City", "years": [2018], "var": "students",
        "action": "bỏ trống", "value": None,
        "reason": "Cả hai ấn phẩm in 2.934.937 học sinh, gấp 2,3 lần các năm lân cận (1,23-1,32 triệu); "
                  "không suy ra được giá trị đúng",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 835",
    },
    {
        "province": "Vinh Long", "years": [2018], "var": "students",
        "action": "bỏ trống", "value": None,
        "reason": "Cả hai ấn phẩm in khoảng 283 nghìn học sinh, gấp 1,7 lần các năm lân cận (168-172 nghìn), "
                  "trùng với số của Tiền Giang; không suy ra được giá trị đúng",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 919",
    },
    {
        "province": "Tay Ninh", "years": [2019], "var": "employed",
        "action": "giữ nguyên", "value": None,
        "reason": "Lao động 3 khu vực cộng lại (696,5) lệch tổng (678,6) 2,6%. Ấn phẩm in đúng như vậy, "
                  "không xác định được số nào sai",
        "evidence": "GSO_Sach_63_tinh_2016_21.pdf, tr. 756",
    },
    {
        "province": "Dong Thap", "years": [2022], "var": "employed",
        "action": "giữ nguyên", "value": None,
        "reason": "Lao động 3 khu vực cộng lại (840,2) lệch tổng (867,3) 3,1%. Ấn phẩm in đúng như vậy, "
                  "không xác định được số nào sai",
        "evidence": "GSO_Sach_63_tinh_2020_24.pdf, tr. 921",
    },
]

# Số học sinh phổ thông năm 2018 ở 7 tỉnh Đồng bằng sông Cửu Long: ấn phẩm 2015-2021 in lệch số giữa
# các tỉnh (ví dụ số của Kiên Giang 197.184 chính là số của Cần Thơ). Ấn phẩm 2015-2018 in số khớp với
# các năm lân cận nên dùng số của ấn phẩm này (đơn vị nghìn học sinh). Tỉnh: (giá trị, trang ấn phẩm 2015-2021)
STUDENTS_2018 = {
    "Tien Giang": (282.3, 868), "Dong Thap": (283.1, 936), "Kien Giang": (296.9, 970),
    "Can Tho": (197.2, 986), "Hau Giang": (132.0, 1002), "Soc Trang": (219.0, 1018), "Bac Lieu": (139.0, 1033),
}
FIXES += [
    {
        "province": prov, "years": [2018], "var": "students", "action": "sửa", "value": value,
        "reason": "Ấn phẩm 2015-2021 in lệch số học sinh năm 2018 giữa các tỉnh Đồng bằng sông Cửu Long; "
                  "thay bằng số của ấn phẩm 2015-2018, khớp với các năm 2017 và 2019",
        "evidence": f"GSO_Sach_63_tinh_2016_21.pdf, tr. {page}; đối chiếu GSO_Sach_63_tinh_2015_18.pdf",
    }
    for prov, (value, page) in STUDENTS_2018.items()
]


# ----------------------------------------------------------------------------- xử lý
def apply_fixes(panel, indicator_of):
    """Áp dụng FIXES và tính lại biến phái sinh. Trả về panel đã sửa và bảng nhật ký."""
    panel = panel.copy()
    log, corrected = [], {}
    for fix in FIXES:
        # grdp_real_growth tính từ grdp_real ở bước tạo dữ liệu nền, không tính lại ở đây
        assert fix["var"] != "grdp_real", "Sửa grdp_real cần bổ sung bước tính lại grdp_real_growth"
        for year in fix["years"]:
            hit = panel.index[(panel.province_name == fix["province"]) & (panel.year == year)]
            assert len(hit) == 1, f"Không tìm thấy {fix['province']} {year}"
            i = hit[0]
            before = panel.at[i, fix["var"]]
            if fix["action"] == "sửa":
                panel.at[i, fix["var"]] = fix["value"]
            elif fix["action"] == "bỏ trống":
                panel.at[i, fix["var"]] = np.nan
            if fix["action"] != "giữ nguyên":
                corrected.setdefault(i, []).append(fix["var"])
            log.append({
                "province_id": panel.at[i, "province_id"], "province_name": fix["province"], "year": year,
                "variable": fix["var"], "indicator_id": indicator_of[fix["var"]],
                "value_before": before, "value_after": panel.at[i, fix["var"]],
                "action": fix["action"], "reason": fix["reason"], "evidence": fix["evidence"],
            })

    # Tính lại biến phái sinh theo đúng công thức của bước tạo dữ liệu nền
    for name, formula, _, _ in DERIVED:
        new = formula(panel).round(4)
        diff = ~((new == panel[name]) | (new.isna() & panel[name].isna()))
        for i in panel.index[diff]:
            corrected.setdefault(i, []).append(name)
            log.append({
                "province_id": panel.at[i, "province_id"], "province_name": panel.at[i, "province_name"],
                "year": panel.at[i, "year"], "variable": name, "indicator_id": None,
                "value_before": panel.at[i, name], "value_after": new[i], "action": "tính lại",
                "reason": "Biến phái sinh, tính lại sau khi sửa biến đầu vào cùng dòng", "evidence": None,
            })
        panel[name] = new

    # Cột mới: danh sách biến đã bị sửa, bỏ trống hoặc tính lại ở dòng đó (giống cách ghi của break_vars)
    panel["corrected_vars"] = pd.Series({i: ";".join(v) for i, v in corrected.items()})
    return panel, pd.DataFrame(log)


def clean_observations(obs, log):
    """Áp dụng các thay đổi trong nhật ký (sửa, bỏ trống) cho observations.csv."""
    obs = obs.copy()
    obs["is_corrected"] = 0
    key = obs.set_index(["indicator_id", "province_id", "year"]).index
    for r in log[log.action.isin(["sửa", "bỏ trống"])].itertuples():
        hit = key == (r.indicator_id, r.province_id, r.year)
        assert hit.sum() == 1, f"Không tìm thấy {r.indicator_id} {r.province_name} {r.year}"
        obs.loc[hit, "value"] = r.value_after
        obs.loc[hit, "is_corrected"] = 1
    return obs


# ----------------------------------------------------------------------------- tự kiểm tra
def check(base, clean, log, obs_base, obs_clean, indicator_of):
    """Tự kiểm tra kết quả. Trả về số lỗi."""
    results = []

    def add(name, ok, detail=""):
        results.append(ok)
        print(f"{'ĐẠT' if ok else 'LỖI'}  {name}" + (f"  ({detail})" if detail else ""))

    add("Giữ nguyên 630 dòng và khóa (tỉnh, năm)",
        len(clean) == len(base) and clean[["province_id", "year"]].equals(base[["province_id", "year"]]))

    # Chỉ những ô có trong nhật ký (sửa, bỏ trống, tính lại) được phép thay đổi
    changed = log[log.action != "giữ nguyên"]
    allowed = set(zip(changed.province_id, changed.year, changed.variable))
    cols = [c for c in base.columns if c != "break_vars"]
    diff = (base[cols] != clean[cols]) & ~(base[cols].isna() & clean[cols].isna())
    row_idx, col_idx = np.where(diff)
    actual = {(base.province_id[r], base.year[r], cols[c]) for r, c in zip(row_idx, col_idx)}
    add("Chỉ các ô trong nhật ký bị thay đổi", actual == allowed, f"{len(actual)} ô")

    # Biến phái sinh khớp công thức ở mọi dòng
    bad = sum(int((~np.isclose(f(clean).round(4), clean[n], equal_nan=True)).sum()) for n, f, _, _ in DERIVED)
    add("Biến phái sinh khớp công thức", bad == 0, f"{bad} ô lệch")

    # Tổng 4 thành phần GRDP khớp tổng GRDP (lệch ≤ 1%) ở mọi dòng đủ 4 thành phần
    parts = clean[GRDP_PARTS].sum(axis=1, min_count=4)
    bad = int(((parts / clean.grdp_current - 1).abs() > 0.01).sum())
    add("Thành phần GRDP giá hiện hành khớp tổng (lệch ≤ 1%)", bad == 0, f"{bad} dòng lệch")

    # Dòng bị bỏ trống thuế: 3 khu vực phải cộng đủ tổng GRDP (căn cứ của việc bỏ trống)
    blank = changed[changed.variable == "grdp_current_tax"]
    rows = clean.set_index(["province_id", "year"]).loc[list(zip(blank.province_id, blank.year))]
    gap = (rows[GRDP_PARTS[:3]].sum(axis=1) / rows.grdp_current - 1).abs().max()
    add("Các dòng bỏ trống thuế: 3 khu vực cộng đủ tổng GRDP", gap < 0.001, f"lệch lớn nhất {gap:.4%}")

    # Số học sinh đã sửa nằm trong khoảng ±10% trung bình năm 2017 và 2019 của cùng tỉnh
    st = clean.pivot(index="province_name", columns="year", values="students").loc[list(STUDENTS_2018)]
    dev = (st[2018] / ((st[2017] + st[2019]) / 2) - 1).abs().max()
    add("Số học sinh 2018 đã sửa khớp các năm lân cận (±10%)", dev < 0.10, f"lệch lớn nhất {dev:.1%}")

    # Lao động: chỉ còn đúng các dòng đã ghi "giữ nguyên"
    emp = (clean[EMPLOY_PARTS].sum(axis=1) / clean.employed - 1).abs() > 0.01
    kept = log[(log.action == "giữ nguyên") & (log.variable == "employed")]
    add("Lao động 3 khu vực lệch tổng chỉ còn ở các dòng đã ghi nhận",
        set(zip(clean.province_name[emp], clean.year[emp])) == set(zip(kept.province_name, kept.year)))

    # observations_clean: giữ nguyên số dòng, chỉ các dòng is_corrected = 1 bị đổi giá trị
    same = (obs_base.value == obs_clean.value) | (obs_base.value.isna() & obs_clean.value.isna())
    add("observations_clean: giữ nguyên số dòng, chỉ các dòng is_corrected bị đổi",
        len(obs_clean) == len(obs_base) and (~same).equals(obs_clean.is_corrected == 1),
        f"{int(obs_clean.is_corrected.sum())} dòng")

    # Hai bảng sạch nhất quán: mọi biến lấy trực tiếp trong panel_clean bằng đúng giá trị ở observations_clean
    long = clean.melt(id_vars=["province_id", "year"], value_vars=list(indicator_of.index), var_name="var")
    long["indicator_id"] = long["var"].map(indicator_of)
    m = long.merge(obs_clean, on=["indicator_id", "province_id", "year"], suffixes=("_panel", "_obs"))
    match = (m.value_panel == m.value_obs) | (m.value_panel.isna() & m.value_obs.isna())
    add("panel_clean và observations_clean khớp nhau", match.all(), f"{len(m):,} ô đã so")

    n_fail = results.count(False)
    print(f"\nTổng: {len(results)} kiểm tra, {n_fail} lỗi")
    return n_fail


def main():
    base = pd.read_csv(BASE / "panel_wide.csv")
    indicators = pd.read_csv(BASE / "indicators.csv")
    indicator_of = indicators.dropna(subset=["panel_var"]).set_index("panel_var").indicator_id

    clean, log = apply_fixes(base, indicator_of)
    obs_base = pd.read_csv(BASE / "observations.csv")
    obs_clean = clean_observations(obs_base, log)

    OUT.mkdir(parents=True, exist_ok=True)
    clean.to_csv(OUT / "panel_clean.csv", index=False, encoding="utf-8-sig")
    obs_clean.to_csv(OUT / "observations_clean.csv", index=False, encoding="utf-8-sig")
    log.to_csv(OUT / "corrections.csv", index=False, encoding="utf-8-sig")

    print(f"panel_clean.csv: {clean.shape[0]} dòng × {clean.shape[1]} cột")
    print(f"observations_clean.csv: {obs_clean.shape[0]:,} dòng × {obs_clean.shape[1]} cột")
    print("corrections.csv:", log.action.value_counts().to_dict(), "\n")
    return check(base, clean, log, obs_base, obs_clean, indicator_of)


if __name__ == "__main__":
    raise SystemExit(1 if main() else 0)
