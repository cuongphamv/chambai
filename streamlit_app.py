import streamlit as st
import nbformat
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc
import re

st.title("🎓 HỆ THỐNG CHẤM BÀI TỰ ĐỘNG - HỒI QUY LOGISTIC")
st.write("Hệ thống kiểm tra mã nguồn thực thi và nội dung giải thích của sinh viên.")

uploaded_file = st.file_uploader("Chọn file bài nộp (.ipynb)", type=["ipynb"])

if uploaded_file is not None:
    try:
        nb = nbformat.reads(uploaded_file.read().decode('utf-8'), as_version=4)
        
        mssv_tim_duoc = "Không xác định"
        ho_ten_tim_duoc = "Không xác định"
        
        # 1. Trích xuất MSSV và Họ tên
        for cell in nb.cells:
            source_text = ""
            if cell.cell_type == 'code':
                source_text = cell.get('source', '')
            elif cell.cell_type == 'markdown':
                source_text = cell.get('source', '')
                
            match_mssv = re.search(r'MSSV\s*=\s*[\"\'](.*?)[\"\']', source_text)
            if not match_mssv:
                match_mssv = re.search(r'MSSV\s*=\s*([0-9]+)', source_text)
            if match_mssv and mssv_tim_duoc == "Không xác định":
                mssv_tim_duoc = match_mssv.group(1).strip()
                
            match_hoten = re.search(r'HO_TEN\s*=\s*[\"\'](.*?)[\"\']', source_text)
            if match_hoten and ho_ten_tim_duoc == "Không xác định":
                ho_ten_tim_duoc = match_hoten.group(1).strip()
        
        if mssv_tim_duoc == "Không xác định" and len(nb.cells) > 0:
            first_code = nb.cells[0].get('source', '')
            numbers = re.findall(r'\b\d{5,8}\b', first_code)
            if numbers:
                mssv_tim_duoc = numbers[0]

        st.success(f"Đã nhận diện bài làm của: **{ho_ten_tim_duoc}** - MSSV: **{mssv_tim_duoc}**")
        
        # 2. Tái tạo dữ liệu chuẩn ngầm định của sinh viên này để đối chiếu
        diem_chi_tiet = 0
        nhan_xet_chi_tiet = []
        
        if mssv_tim_duoc != "Không xác định":
            seed_val = abs(hash(str(mssv_tim_duoc).strip().upper())) % (2**32)
            np.random.seed(seed_val)
            
            n_samples = np.random.randint(650, 1001)
            tuoi = np.clip(np.random.normal(loc=45, scale=10, size=n_samples).astype(int), 20, 80)
            bmi = np.clip(np.random.normal(loc=23.5, scale=3.5, size=n_samples), 15, 40)
            hut_thuoc = np.random.binomial(n=1, p=np.random.uniform(0.3, 0.65), size=n_samples)
            tap_the_duc = np.random.binomial(n=1, p=np.random.uniform(0.2, 0.75), size=n_samples)
            gioi_tinh = np.random.binomial(n=1, p=0.52, size=n_samples)
            benh_dong_mac = np.random.binomial(n=1, p=np.random.uniform(0.1, 0.45), size=n_samples)
            
            z = (-7.5 + np.random.uniform(0.01, 0.12) * tuoi + 
                 np.random.uniform(0.1, 0.2) * bmi + 
                 np.random.uniform(0.34, 1.2) * hut_thuoc - 
                 np.random.uniform(0.3, 1.3) * tap_the_duc + 
                 0.50 * gioi_tinh + 
                 np.random.uniform(0.8, 2.1) * benh_dong_mac + 
                 np.random.normal(0, 1, size=n_samples))
            prob = 1 / (1 + np.exp(-z))
            tang_huyet_ap = np.random.binomial(n=1, p=prob)
            
            df_chuan = pd.DataFrame({
                'Tuoi': tuoi, 'BMI': np.round(bmi, 1), 'GioiTinh': gioi_tinh,
                'HutThuoc': hut_thuoc, 'TapTheDuc': tap_the_duc,
                'BenhDongMac': benh_dong_mac, 'TangHuyetAp': tang_huyet_ap
            })
            
            model = smf.logit('TangHuyetAp ~ Tuoi + BMI + GioiTinh + HutThuoc + TapTheDuc + BenhDongMac', data=df_chuan).fit(disp=0)
            y_pred_prob = model.predict(df_chuan)
            fpr, tpr, _ = roc_curve(df_chuan['TangHuyetAp'], y_pred_prob)
            roc_auc = auc(fpr, tpr)

            # 3. Quét kiểm tra chất lượng thực tế từng câu trong notebook của sinh viên
            code_cells = [cell for cell in nb.cells if cell.cell_type == 'code']
            markdown_cells = [cell for cell in nb.cells if cell.cell_type == 'markdown']
            
            # Kiểm tra xem sinh viên có thực hiện viết lệnh hồi quy hay không (tìm từ khóa smf.logit hoặc logit)
            has_logit_code = any('logit' in cell.get('source', '') for cell in code_cells)
            if has_logit_code:
                diem_chi_tiet += 4.0
                nhan_xet_chi_tiet.append("✅ Đã viết lệnh xây dựng mô hình hồi quy logistic.")
            else:
                nhan_xet_chi_tiet.append("❌ Chưa tìm thấy câu lệnh chạy mô hình hồi quy (`logit`) trong bài.")

            # Kiểm tra xem sinh viên có viết phần giải thích văn bản thực chất không (loại bỏ các cell chỉ chứa placeholder)
            valid_markdowns = 0
            for cell in markdown_cells:
                text = cell.get('source', '').strip()
                # Kiểm tra nếu text có độ dài kha khá và không phải là văn bản hướng dẫn mặc định
                if len(text) > 30 and "Nhập câu trả lời" not in text and "HƯỚNG DẪN" not in text:
                    valid_markdowns += 1
            
            if valid_markdowns >= 4:
                diem_chi_tiet += 6.0
                nhan_xet_chi_tiet.append(f"✅ Tìm thấy {valid_markdowns} phần giải thích/nhận xét chi tiết.")
            else:
                diem_chi_tiet += float(valid_markdowns) * 1.2
                nhan_xet_chi_tiet.append(⚠️ f" Chỉ tìm thấy {valid_markdowns} phần giải thích hợp lệ (cần viết chi tiết hơn ở các câu nhận xét).")

        st.write("---")
        st.subheader("📊 Kết quả kiểm định chi tiết bài làm:")
        for note in nhan_xet_chi_tiet:
            st.write(note)
            
        st.metric(label="Điểm đánh giá thực chất", value=f"{round(diem_chi_tiet, 1)} / 10.0")
        
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {str(e)}")
