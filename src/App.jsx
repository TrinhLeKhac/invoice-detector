import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedImages, setSelectedImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(null);
  const [base64, setBase64] = useState([]);
  const [response, setresponse] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (currentIndex !== null) {
      sendImageToAPI(base64[currentIndex]);
    }
  }, [currentIndex]);

  const handleImageChange = (event) => {
    const files = event.target.files;
    const images = [];
    const base64s = [];

    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();
      reader.readAsDataURL(files[i]);
      reader.onload = (e) => {
        images.push(e.target.result);
        base64s.push(reader.result);
        if (images.length === files.length) {
          setSelectedImages(images);
          setCurrentIndex(0);
          setBase64(base64s);
        }
      };
    }
  };

  const handlePrev = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex > 0 ? prevIndex - 1 : selectedImages.length - 1
    );
  };

  const handleNext = () => {
    setCurrentIndex((prevIndex) =>
      prevIndex < selectedImages.length - 1 ? prevIndex + 1 : 0
    );
  };

  const sendImageToAPI = (image) => {
    setLoading(true);
    fetch('api/invoice_detector', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: image }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log(data.order_details);
        setresponse(data);
      })
      .catch((error) => {
        console.error('Error sending image to API:', error);
        setresponse(null);
      })
      .finally(() => {
        setLoading(false);
      });
  };

  return (
    <>
      <div className='image-container'>
        <div>
          <input
            className='input-btn'
            type='file'
            accept='image/*'
            multiple
            onChange={handleImageChange}
          />
          {selectedImages.length > 0 && (
            <div className='selected-images'>
              <button onClick={handlePrev}>Prev</button>
              <img src={selectedImages[currentIndex]} alt='Selected' />
              <button onClick={handleNext}>Next</button>
            </div>
          )}
        </div>

        {loading && (
          <div className='loading'>
            <div className='spinner'></div>
          </div>
        )}

        {!loading && response && (
          <div className='api-images'>
            {response?.gray && <img src={response.gray} alt='Gray' />}
            {response?.binary && <img src={response.binary} alt='Binary' />}
            {response?.cropped && <img src={response.cropped} alt='Cropped' />}
          </div>
        )}
      </div>

      {!loading && response && (
        <div className='tables'>
          {response?.invoice_information && (
            <table>
              <thead>
                <tr>
                  <th>Thông tin hoá đơn</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>
                    {response.invoice_information
                      .split('\n')
                      .map((line, index) => (
                        <span key={index}>
                          {line}
                          <br />
                        </span>
                      ))}
                  </td>
                </tr>
              </tbody>
            </table>
          )}

          {response?.profile && (
            <table>
              <thead>
                <tr>
                  <th colSpan={2}>Thông tin profile</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Thời gian xuất hoá đơn</td>
                  <td>{response.profile.created_time}</td>
                </tr>
                <tr>
                  <td>Shop</td>
                  <td>{response.profile.shop_name}</td>
                </tr>
                <tr>
                  <td>Hotline</td>
                  <td>{response.profile.hotline.join(' - ')}</td>
                </tr>
                <tr>
                  <td>Nhân viên bán hàng</td>
                  <td>{response.profile.employee_name}</td>
                </tr>
                <tr>
                  <td>Khách hàng</td>
                  <td>{response.profile.customer_name}</td>
                </tr>
                <tr>
                  <td>SDT khách hàng</td>
                  <td>{response.profile.customer_phone}</td>
                </tr>
                <tr>
                  <td>Địa chỉ</td>
                  <td>{response.profile.address}</td>
                </tr>
                <tr>
                  <td>Khu vực</td>
                  <td>{response.profile.region}</td>
                </tr>
                <tr>
                  <td>Thời gian giao hàng</td>
                  <td>{response.profile.shipping_time}</td>
                </tr>
              </tbody>
            </table>
          )}

          {response?.order_details && (
            <table>
              <thead>
                <tr>
                  <th>Tên sản phẩm</th>
                  <th>Số lượng</th>
                  <th>Đơn giá</th>
                  <th>Thành tiền</th>
                </tr>
              </thead>
              <tbody>
                {response.order_details.map((order, index) => (
                  <tr key={index}>
                    <td>{order.product_name}</td>
                    <td>{order.quantity}</td>
                    <td>{order.unit_price}</td>
                    <td>{order.total_price}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          {response?.order_summary && (
            <table>
              <thead>
                <tr>
                  <th colSpan={2}>Thông kê đơn hàng</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Tổng số lượng</td>
                  <td>{response.order_summary.total_quantity}</td>
                </tr>
                <tr>
                  <td>Tổng tiền hàng</td>
                  <td>{response.order_summary.total_amount}</td>
                </tr>
                <tr>
                  <td>Chiết khấu</td>
                  <td>{response.order_summary.discount}</td>
                </tr>
                <tr>
                  <td>Tổng tiền phải thu</td>
                  <td>{response.order_summary.monetary}</td>
                </tr>
              </tbody>
            </table>
          )}
        </div>
      )}
    </>
  );
}

export default App;
