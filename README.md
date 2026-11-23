# NCKH 2027

Dự án nghiên cứu sử dụng dữ liệu bảng (panel) kinh tế – xã hội của 63 tỉnh, thành phố Việt Nam giai đoạn 2015–2024, phương pháp chính là causal forest.

## Cấu trúc thư mục

```
NCKH 2027/
├── README.md                  giới thiệu dự án (file này)
├── requirements.txt           thư viện Python và phiên bản
├── data/
│   ├── raw/                   3 ấn phẩm PDF của Tổng cục Thống kê – không sửa
│   ├── interim/
│   │   ├── extracted/         gso_extracted.xlsx – số liệu trích xuất từ 3 PDF
│   │   └── base/              bộ dữ liệu nền: provinces, indicators, observations, panel_wide
│   ├── processed/             panel_clean, observations_clean (đã làm sạch) + corrections (nhật ký sửa)
│   └── model/                 dữ liệu đầu vào cho mô hình
├── src/                       hàm dùng chung (eda.py: hàm cho hai notebook EDA)
├── scripts/                   code tạo dữ liệu, chạy theo số thứ tự
├── notebooks/                 01_eda_base, 02_eda_processed (cùng cấu trúc); sau này: mô hình cơ sở, causal forest
├── outputs/                   kết quả mô hình đã chạy (không đưa lên Git)
├── reports/
│   ├── figures/               biểu đồ đưa vào bài
│   └── tables/                bảng kết quả đưa vào bài
└── docs/
    ├── references/            tài liệu tham khảo
    ├── proposal/              đề cương
    └── manuscript/            bản thảo
```

## Luồng dữ liệu

```
data/raw (PDF) → data/interim/extracted → data/interim/base → data/processed → data/model
```

Mỗi bước được tạo từ bước trước bằng script; không sửa tay dữ liệu ở bất kỳ bước nào. Muốn thay đổi thì sửa script rồi chạy lại.

## Cài đặt

```
pip install -r requirements.txt
```

## Chạy lại từ đầu

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1 | `python scripts/01_build_base.py` | Tạo bộ dữ liệu nền trong `data/interim/base/` |
| 2 | `python scripts/02_check_base.py --pdf` | Kiểm tra bộ dữ liệu nền và đối chiếu ngẫu nhiên với PDF |
| 3 | `python scripts/03_clean.py` | Làm sạch, ghi `panel_clean.csv`, `observations_clean.csv` và nhật ký `corrections.csv` vào `data/processed/` |

## Nguồn dữ liệu

Tổng cục Thống kê. *Tư liệu kinh tế – xã hội 63 tỉnh, thành phố trực thuộc trung ương*, các ấn phẩm giai đoạn 2015–2018, 2015–2021 và 2020–2024. Hà Nội: NXB Thống kê.
