## Cài đặt và chạy ứng dụng
``` bash
cd api
python3.10 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r ./requirements.txt
flask run
```

## Tài liệu API

### 1. Mô tả
API `/image` được sử dụng để nhận diện thông tin hóa đơn từ hình ảnh được mã hóa dưới dạng Base64 và mã cửa hàng.

### 2. Endpoint
POST /image

#### 2.1. Đầu vào (Request Body)
| Tham số  | Kiểu dữ liệu   | Mô tả |
|------------|--------|-------------|
| `image`    | string (base64) | Hình ảnh hóa đơn dưới dạng chuỗi Base64 |
| `shop_code` | string | Mã cửa hàng |

#### 2.2. Đầu ra (Response Body)
| Tham số  | Kiểu dữ liệu   | Mô tả |
|------------|--------|-------------|
| `shop_code`    | string | Mã cửa hàng từ request |
| `name` | string | Tên khách hàng trên hóa đơn |
| `phone`    | string) | Số điện thoại khách hàng |
| `address` | string | Địa chỉ khách hàng|
| `commune`    | string | Xã/Phường của địa chỉ |
| `district` | string | Quận/Huyện của địa chỉ |
| `province`    | string| Tỉnh/Thành phố của địa chỉ |
| `phone_checked` | int | Kiểm tra số điện thoại (1: đúng, 0: sai) |
| `name_checked`    | int | Kiểm tra tên (1: đúng, 0: sai) |
| `address_checked` | int | Kiểm tra địa chỉ (1: đúng, 0: sai) |
| `total_quantity`    | int  | Tổng số lượng sản phẩm trên hóa đơn |
| `total_amount` | float | Tổng tiền trên hóa đơn |
| `discount`    | float  | Số tiền giảm giá |
| `monetary` | float | Tổng tiền sau giảm giá |

### 3. Ví dụ
#### 3.1. Ví dụ Request
```json
{
  "image": "<base64_encoded_image>",
  "shop_code": "123"
}
```

#### 3.2. Ví dụ Response
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

## 4. Lưu ý
- Đảm bảo image được mã hóa đúng định dạng Base64
- Kết quả trả về có thể thay đổi tùy theo thông tin hóa đơn được nhận diện