## API Documentation

### `/image`

#### Description
This API endpoint receives an image and a shop code to detect invoice details.

#### Request Format
| Parameter  | Type   | Description |
|------------|--------|-------------|
| `image`    | string (base64) | The image data in base64 encoding. |
| `shop_code` | string | The shop code associated with the invoice. |

#### Example Request (JSON)
```json
{
  "image": "<base64_encoded_image>",
  "shop_code": "SHOP123"
}
```

#### Example Response (JSON)
```json
{
  "shop_code": "SHOP123",
  "invoice_details": {
    "total_amount": 500000,
    "customer_name": "John Doe",
    "customer_phone": "0987654321"
  }
}
```

#### Notes
- Ensure the `image` is properly encoded in base64 format.
- The `shop_code` must be a valid identifier for invoice detection.
- Response may vary depending on the detected invoice details.