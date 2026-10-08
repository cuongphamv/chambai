import streamlit as st
import nbformat
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.metrics import roc_curve, auc

st.title("🎓 HỆ THỐNG CHẤM BÀI TỰ ĐỘNG - HỒI QUY LOGISTIC")
st.write("Tải lên file bài tập `.ipynb` của bạn để hệ thống tự động kiểm tra và chấm điểm.")

# Khung upload file từ sinh viên
uploaded_file = st.file_uploader("Chọn file bài nộp (.ipynb)", type=["ipynb"])

if uploaded_file is not None:
    try:
        # Đọc nội dung file notebook
        nb = nbformat.reads(uploaded_file.read().decode('utf-8'), as_version=4)
        
        mssv_tim_duoc = "Không xác định"
        ho_ten_tim_duoc = "Không xác định"
        
        # Trích xuất thông tin từ ô code đầu tiên
        if len(nb.cells) > 0:
            ma_nguon_dau = nb.cells[0].get('source', '')
            for line in ma_nguon_dau.split('\n'):
                if 'MSSV' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        mssv_tim_duoc = parts[1].strip().strip('"\'')
                if 'HO_TEN' in line and '=' in line:
                    parts = line.split('=')
                    if len(parts) > 1:
                        ho_ten_tim_duoc = parts[1].strip().strip('"\'')
        
        st.success(f"Đã nhận diện bài làm của: **{ho_ten_tim_duoc}** - MSSV: **{mssv_tim_duoc}**")
        
        # Kiểm tra khối lượng hoàn thành bài làm
        so_cell_code = sum(1 for cell in nb.cells if cell.cell_type == 'code' and len(cell.get('source', '').strip()) > 0)
        so_cell_markdown = sum(1 for cell in nb.cells if cell.cell_type == 'markdown' and len(cell.get('source', '').strip()) > 50)
        
        # Tính điểm sơ bộ cấu trúc
        diem_cau_truc = 0
        if so_cell_code >= 5: diem_cau_truc += 5.0
        if so_cell_markdown >= 5: diem_cau_truc += 5.0
        
        st.write("---")
        st.subheader("📊 Kết quả kiểm tra sơ bộ:")
        st.write(f"- Số ô code đã viết lệnh: {so_cell_code}")
        st.write(f"- Số ô giải thích/nhận xét: {so_cell_markdown}")
        st.metric(label="Điểm đánh giá sơ bộ", value=f"{diem_cau_truc} / 10.0")
        
        # --- PHẦN TÍNH TOÁN ĐÁP ÁN CHUẨN NGẦM (Dựa theo MSSV của sinh viên) ---
        if mssv_tim_duoc != "Không xác định":
            seed_val = abs(hash(str(mssv_tim_duoc).strip().upper())) % (2**32)
            np.random.seed(seed_val)
            
            # Tái tạo dữ liệu chuẩn của riêng sinh viên đó
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
            
            st.info(f"💡 **Thông tin đối chiếu ngầm:** Bộ dữ liệu cá nhân của bạn có {n_samples} quan sát. Giá trị AUC chuẩn của mô hình là: **{round(roc_auc, 3)}**. Hãy kiểm tra lại kết quả trong notebook của bạn xem đã khớp chưa!")
            
    except Exception as e:
        st.error(f"Lỗi khi xử lý file: {str(e)}")
