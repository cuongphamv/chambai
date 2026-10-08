import re

# Khởi tạo giá trị mặc định
mssv_tim_duoc = "Không xác định"
ho_ten_tim_duoc = "Không xác định"

# Quét qua toàn bộ các cell trong notebook để tìm MSSV và Họ tên
for cell in nb.cells:
    source_text = ""
    if cell.cell_type == 'code':
        source_text = cell.get('source', '')
    elif cell.cell_type == 'markdown':
        source_text = cell.get('source', '')
        
    # Dùng RegEx tìm biến MSSV = "..." hoặc biến số
    match_mssv = re.search(r'MSSV\s*=\s*[\"\'](.*?)[\"\']', source_text)
    if not match_mssv:
        match_mssv = re.search(r'MSSV\s*=\s*([0-9]+)', source_text)
    if match_mssv and mssv_tim_duoc == "Không xác định":
        mssv_tim_duoc = match_mssv.group(1).strip()
        
    # Dùng RegEx tìm biến HO_TEN = "..."
    match_hoten = re.search(r'HO_TEN\s*=\s*[\"\'](.*?)[\"\']', source_text)
    if match_hoten and ho_ten_tim_duoc == "Không xác định":
        ho_ten_tim_duoc = match_hoten.group(1).strip()

# Nếu vẫn chưa tìm thấy, thử tìm bất kỳ chuỗi số nào trông giống MSSV trong ô code đầu tiên
if mssv_tim_duoc == "Không xác định" and len(nb.cells) > 0:
    first_code = nb.cells[0].get('source', '')
    numbers = re.findall(r'\b\d{5,8}\b', first_code) # Tìm các chuỗi từ 5 đến 8 chữ số
    if numbers:
        mssv_tim_duoc = numbers[0]
