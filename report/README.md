# Báo cáo Lab 3 — LaTeX hai cột

- Tác giả: **Le Khoi Nguyen**.
- Giảng viên hướng dẫn: **Phan Thi Thu Hong**.
- Subject: **Lab3-ObjectDetection-Varroa/DPL302m**.
- Nguồn: `report.tex` và `references.bib`; bản đọc: `report.pdf`.

Report tiếng Việt dùng bố cục article A4 hai cột theo phong cách bài nghiên cứu.
Đây không phải tuyên bố tuân thủ template chính thức của một hội nghị cụ thể.
Nội dung bao gồm problem, dataset audit, model selection, implementation, training,
evaluation, inference, comparison, discussion, limitations và reproducibility.

## Biên dịch

Cần XeLaTeX, BibTeX và các font Noto Serif/Noto Sans/Noto Sans Mono.

```bash
cd report
make
```

Hoặc chạy `xelatex report.tex`, `bibtex report`, rồi `xelatex report.tex` hai lần.
Hình/bảng được lưu sẵn nên biên dịch từ clean clone không cần weights hoặc dataset.

## Sinh lại hình và bảng từ artifact

```bash
python -m pip install matplotlib numpy Pillow
python build_figures.py
```

Script đọc JSON/CSV trong `../output/`, sinh PDF vector cho biểu đồ và bảng LaTeX
trong `generated/`. Phần ảnh định tính cần `../dataset/varroa_ready/images/test/`.
Nếu không có ảnh local, các ảnh minh họa đã commit vẫn có thể dùng để biên dịch.
Không chỉnh số liệu thủ công trong bảng sinh tự động.

`generated/artifact_sha256.json` xác định bộ JSON/CSV đã dùng; các đường dẫn tương
đối với root Lab 3. `generated/qualitative_manifest.json` ghi ID, path và quy tắc
chọn ảnh định tính: ảnh đầu tiên trong thứ tự test gốc thỏa từng loại tình huống.
Ảnh mẫu không đại diện tần suất lỗi; các kết luận định lượng dùng toàn bộ test.

Các số liệu chính được kiểm tra bằng prediction/annotation đã tải. Báo cáo phân biệt
run AMP Faster R-CNN thất bại và run FP32 thành công; không khẳng định đã xác định
phép toán gây NaN/Inf trong run cũ. Các đề xuất cải tiến là hướng phát triển, chưa có
thực nghiệm mới để chứng minh.
