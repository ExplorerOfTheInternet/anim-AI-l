import numpy as np
from PIL import Image
from io import BytesIO

async def prepare_image(file):
    image = Image.open(BytesIO(await file.read()))
    image = image.resize((128, 128))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

