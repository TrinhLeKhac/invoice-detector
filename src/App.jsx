import { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [selectedImages, setSelectedImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(null);
  const [base64, setBase64] = useState([]);

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
    fetch('api/invoice_detector', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ image: image }),
    })
      .then((res) => res.json())
      .then((data) => {
        console.log('Response from API:', data);
      })
      .catch((error) => {
        console.error('Error sending image to API:', error);
      });
  };

  return (
    <>
      <input
        className='input-btn'
        type='file'
        accept='image/*'
        multiple
        onChange={handleImageChange}
      />
      {selectedImages.length > 0 && (
        <div className='container'>
          <button onClick={handlePrev}>Prev</button>
          <img
            src={selectedImages[currentIndex]}
            alt='Selected'
            style={{ maxWidth: '800px', height: 'auto' }}
          />
          <button onClick={handleNext}>Next</button>
        </div>
      )}
    </>
  );
}

export default App;
