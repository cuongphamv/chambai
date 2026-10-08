import streamlit as st
import nbformat
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
import re

st.title("🎓 HỆ THỐNG CHẤM BÀI TỰ ĐỘNG - CHI TIẾT TỪNG CÂU")
st.write("Hệ thống kiểm tra mã lệnh chuyên biệt cho 10 câu hỏi hồi quy logistic.")

uploaded_file = st.file_uploader("Chọn file bài nộp (.ipynb)", type=["ipynb"])

if uploaded_file is not None:
    try:
        nb = nbformat.reads(uploaded_file.read().decode('utf-8'), as_version=4)
        
        mssv_tim_duoc = "Không xác định"
        ho_ten_tim_duoc = "Không xác định"
        
        # 1. Trích xuất thông tin sinh viên
        for cell in nb.cells:
            source_text = cell.get('source', '')
            match_mssv = re.search(r'MSSV\s*=\s*[\"\'](.*?)[\"\']', source_text)
            if not match_mssv:
                match_mssv = re.search(r'MSSV\s*=\s*([0-9]+)', source_text)
            if match_mssv and mssv_tim_duoc == "Không xác định":
                mssv_tim_duoc = match_mssv.group(1).strip()
                
            match_hoten = re.search(r'HO_TEN\s*=\s*[\"\'](.*?)[\"\']', source_text)
            if match_hoten and ho_ten_tim_duoc == "Không xác định":
                ho_ten_tim_duoc = match_hoten.group(1).strip()
        
        st.success(f"Đã nhận diện bài làm của: **{ho_ten_tim_duoc}** - MSSV: **{mssv_tim_duoc}**")
        
        # 2. Gom nhóm các ô code và markdown theo thứ tự xuất hiện trong bài
        code_cells = [cell.get('source', '') for cell in nb.cells if cell.cell_type == 'code'][1:] # Bỏ qua ô đầu tiên (khởi tạo dữ liệu)
        markdown_cells = [cell.get('source', '') for cell in nb.cells if cell.cell_type == 'markdown'][1:]
        
        chi_tiet_cham = []
        tong_diem = 0
        
        # Kiểm tra Câu 1: Thống kê mô tả (0.5đ code + 0.5đ nhận xét)
        c1_code = code_cells[0] if len(code_cells) > 0 else ""
        diem_c1_code = 0.5 if ('describe' in c1_code or 'value_counts' in c1_code or 'mean' in c1_code) else 0
        tong_diem += diem_c1_code
        chi_tiet_cham.append(f"- **Câu 1 (Thống kê mô tả):** {'✅ Đạt' if diem_c1_code > 0 else '❌ Thiếu lệnh thống kê mô tả (`describe` hoặc `value_counts`)'} (+{diem_diem_c1_code if 'diem_diem_c1_code' in locals() else diem_c1_code}đ)")

        # Kiểm tra Câu 3 & 4: Hồi quy đơn/đa biến (2.5đ)
        c34_code = "".join(code_cells[2:4]) if len(code_cells) > 3 else ""
        diem_c34 = 2.5 if 'logit' in c34_code else 0
        tong_diem += diem_c34
        chi_tiet_cham.append(f"- **Câu 3 & 4 (Mô hình Hồi quy):** {'✅ Đã áp dụng hàm `logit`' if diem_c34 > 0 else '❌ Không tìm thấy lệnh gọi mô hình hồi quy logistic'} (+{diem_c34}đ)")

        # Kiểm tra Câu 8: Ma trận nhầm lẫn (1.0đ)
        c8_code = code_cells[7] if len(code_cells) > 7 else ""
        diem_c8 = 1.0 if ('confusion_matrix' in c8_code or 'predict' in c8_code) else 0
        tong_diem += diem_c8
        chi_tiet_cham.append(f"- **Câu 8 (Confusion Matrix):** {'✅ Có lệnh ma trận nhầm lẫn' if diem_c8 > 0 else '❌ Chưa viết lệnh tính ma trận nhầm lẫn'} (+{diem_c8}đ)")

        # Kiểm tra Câu 9: ROC và AUC (1.0đ)
        c9_code = code_cells[8] if len(code_cells) > 8 else ""
        diem_c9 = 1.0 if ('roc_curve' in c9_code or 'auc' in c9_code) else 0
        tong_diem += diem_c9
        chi_tiet_cham.append(f"- **Câu 9 (ROC & AUC):** {'✅ Đã viết lệnh vẽ ROC/AUC' if diem_c9 > 0 else '❌ Thiếu lệnh vẽ ROC hoặc tính AUC'} (+{diem_c9}đ)")

        # Điểm đánh giá phần giải thích văn bản cho các câu còn lại
        so_cau_giai_thich_thuc_te = sum(1 for text in markdown_cells if len(text.strip()) > 35 and "Nhập câu trả lời" not in text)
        diem_giai_thich = min(5.0, so_cau_giai_thich_thuc_te * 0.5)
        tong_diem += diem_giai_thich
        chi_tiet_cham.append(f"- **Phần giải thích/Nhận xét (Tổng hợp):** Ghi nhận {so_cau_giai_thich_thuc_te} phần trả lời chi tiết đạt yêu cầu (+{diem_giai_thich}đ)")

        st.write("---")
        st.subheader("📋 Bảng điểm chi tiết theo tiêu chí:")
        for item in chi_tiet_cham:
            st.write(item)
            
        st.metric(label="Tổng điểm đánh giá tự động", value=f"{round(tong_diem, 1)} / 10.0")

    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {str(e)}")
