import { useState, useEffect } from 'react';
import reactLogo from './assets/react.svg';
import viteLogo from '/vite.svg';
import './App.css';

function App() {
  const [accuracy, setAccuracy] = useState(0);
  const [selectedImages, setSelectedImages] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    fetch('api/ml')
      .then((res) => res.json())
      .then((data) => {
        setAccuracy(data.accuracy);
      });
  }, []);

  const handleImageChange = (event) => {
    const files = event.target.files;
    const images = [];
    for (let i = 0; i < files.length; i++) {
      const reader = new FileReader();
      reader.onload = (e) => {
        images.push(e.target.result);
        if (images.length === files.length) {
          setSelectedImages(images);
          setCurrentIndex(0);
        }
      };
      reader.readAsDataURL(files[i]);
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

  return (
    <>
      <div className='card'>Output: {accuracy}</div>
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
