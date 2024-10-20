from fastapi import FastAPI,Form,UploadFile,File
from fastapi.responses import FileResponse
from typing import Annotated
import torch
import torch.nn.functional as F
import logging
from .class_mr import TumorModel,device
from starlette.responses import PlainTextResponse
from PIL import Image
from io import BytesIO
import numpy as np


app = FastAPI()

class Classify_IMg:
    def __init__(self):
        pass

    @app.get("/")
    def root():
        return {"CNN":"Brain tumor classification system"}

    @app.post("/classify_upload/")
    async def upload_identify(file: UploadFile = File(...)):
        try:

            # Loading the pre-trained model
            model = TumorModel().to(device)
            model.load_state_dict(torch.load("../model/brain_tumor_mri_model.pth"))
            model.eval()

            # Processing the uploaded image
            contents = await file.read()
            image = Image.open(BytesIO(contents)).convert("L")
            image = image.resize((130, 130))
            arr = np.array(image).reshape(1, 1, 130, 130)
            tensor = torch.tensor(arr, device=device).float()

            # Making the  prediction
            with torch.no_grad():
                output = model(tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()

            result = "Malignant (Brain Tumor)" if predicted_class == 1 else "Non-Malignant (Healthy)"
            confidence = probabilities[0][predicted_class].item()
            logging.info("Image identified and classified")

            return {"prediction": result, "confidence": f"{confidence:.2f}"}
        except Exception as error:
            logging.error(str(error))


def main():
    class_ = Classify_IMg()
    class_.root()
    class_.upload_identify()

if __name__=="__main__":
    main()




