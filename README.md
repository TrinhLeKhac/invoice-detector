## 1. Cài đặt và chạy ứng dụng

### 1.1 Cài đặt 
Cài python 3.10  
Tạo môi trường và cài thư viện
``` bash
cd api
python3.10 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ./requirements.txt
```

### 1.2. Chạy demo quá trình xử lý hoá đơn
Mở terminal mới, start backend (FastAPI)
``` bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
Mở terminal mới, start Frontend (Vite+React)
```bash
npm run dev
```

### 1.3. Check API 
Mở trang document FastAPI [http://0.0.0.0:8000/docs](http://0.0.0.0:8000/docs)

## 2. Tài liệu API

### 2.1. Mô tả
API `/api/invoice_detector` được sử dụng để nhận diện thông tin hóa đơn từ hình ảnh được mã hóa dưới dạng Base64 và mã cửa hàng.

### 2.2. Endpoint
Phương thức: POST 
Endpoint: `/api/invoice_detector`

#### 2.2.1. Đầu vào (Request Body)
| Tham số  | Kiểu dữ liệu   | Mô tả |
|------------|--------|-------------|
| `image`    | string (base64) | Hình ảnh hóa đơn dưới dạng chuỗi Base64 |
| `shop_code` | string | Mã cửa hàng |

#### 2.2.2. Đầu ra (Response Body)
| Tham số  | Kiểu dữ liệu   | Mô tả |
|------------|--------|-------------|
| `shop_code`    | string | Mã cửa hàng từ request |
| `name` | string | Tên khách hàng trên hóa đơn |
| `phone`    | string | Số điện thoại khách hàng |
| `address` | string | Địa chỉ khách hàng|
| `commune`    | string | Xã/Phường của địa chỉ |
| `district` | string | Quận/Huyện của địa chỉ |
| `province`    | string| Tỉnh/Thành phố của địa chỉ |
| `phone_checked` | int | Kiểm tra có số điện thoại trong API hay không(1: có, 0: không) |
| `name_checked`    | int | Kiểm tra có đúng tên ứng với số điện thoại hay không(1: đúng, 0: sai) |
| `address_checked` | int | Kiểm tra có đúng địa chỉ ứng với số điện thoại hay không (1: đúng, 0: sai) |
| `total_quantity`    | int  | Tổng số lượng sản phẩm trên hóa đơn |
| `total_amount` | float | Tổng tiền trên hóa đơn |
| `discount`    | float  | Số tiền giảm giá |
| `monetary` | float | Tổng tiền sau giảm giá |

### 2.3. Ví dụ
#### 2.3.1. Ví dụ Request
```json
{
  "image": "<base64_encoded_image>",
  "shop_code": "123"
}
```

#### 2.3.2. Ví dụ Response
```json
{
    "shop_code": "SHOP123",
    "name": "Nguyen Van A",
    "phone": "0987654321",
    "address": "123 Đường ABC, Phường XYZ",
    "commune": "Phường XYZ",
    "district": "Quận 1",
    "province": "TP. Hồ Chí Minh",
    "phone_checked": 1,
    "name_checked": 1,
    "address_checked": 1,
    "total_quantity": 5,
    "total_amount": 1000000,
    "discount": 50000,
    "monetary": 950000
}
```

## 3. Lưu ý
- Trang web convert base64 sang image và ngược lại ([https://base64.guru/converter/encode/image](https://base64.guru/converter/encode/image), [https://www.base64-image.de/](https://www.base64-image.de/))
- Đảm bảo image được mã hóa đúng định dạng Base64
- Kết quả trả về có thể thay đổi tùy theo thông tin hóa đơn được nhận diện