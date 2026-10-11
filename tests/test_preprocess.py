import numpy as np
from PIL import Image

from app.services.predictor import preprocessing

MEAN=np.full((3,1,1), 0.5, dtype=np.float32)
STD= np.full((3,1,1), 0.5, dtype=np.float32)

def test_preprocess_shape():
    image=Image.new("RGB", (64,64), (255,0,0))
    x=preprocessing(image, MEAN, STD)

    assert x.shape== (1,3,64,64)
    assert x.dtype == np.float32
    assert np.allclose(x[0, 0], 1.0)   
    assert np.allclose(x[0, 1], -1.0)  
    assert np.allclose(x[0, 2], -1.0) 
