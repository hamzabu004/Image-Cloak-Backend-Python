import uuid
from time import sleep

from fastapi import FastAPI, File, UploadFile, Form
from typing import Annotated
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
import os
import io
import cloudinary
import cloudinary.uploader
import cloudinary.api
import requests
from fastapi.responses import JSONResponse
from Stegnographer.steganography_advanced import encode, decode

# Configure Cloudinary - add your credentials
cloudinary.config(
    cloud_name="du5e7ngxi",
    api_key="593292333232947",
    api_secret="UBlLTDV3Fh5fdHGjCfhrR_HtfRE"
)

app = FastAPI()

@app.post("/encrypt-image/")
async def encrypt_image(image_url: str = Form(...), password: str = Form(...), iv: bytes = Form(...),
                        mode: int = Form(...), isSteg: bool = Form(...), stegImage: UploadFile = None) :
      # Fetch the image from Cloudinary URL

    response = requests.get(image_url)
    if response.status_code != 200:
        return JSONResponse(content={"error": "Failed to fetch image from URL"}, status_code=403)

    # Process the image
    image_data = response.content
    nparr = np.frombuffer(image_data, np.uint8)
    imageOrig = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    rowOrig, columnOrig, depthOrig = imageOrig.shape

    print("rowOrig: ", rowOrig)
    print("columnOrig: ", columnOrig)
    print("depthOrig: ", depthOrig)
    # Convert original image data to bytes
    imageOrigBytes = imageOrig.tobytes()

    # Encrypt
    key = password.encode('utf-8')
    cipher = AES.new(key, AES.MODE_CBC, iv)
    imageOrigBytesPadded = pad(imageOrigBytes, AES.block_size)
    ciphertext = cipher.encrypt(imageOrigBytesPadded)

    # Convert ciphertext bytes to encrypted image data
    paddedSize = len(imageOrigBytesPadded) - len(imageOrigBytes)
    void = columnOrig * depthOrig - len(iv) - paddedSize
    ivCiphertextVoid = iv + ciphertext + bytes(void)
    imageEncrypted = np.frombuffer(ivCiphertextVoid, dtype=imageOrig.dtype).reshape(rowOrig + 1, columnOrig, depthOrig)

    temp_filename = f"encrypted_image_{os.urandom(8).hex()}.bmp"
    # Save the encrypted image to a BMP file in memory
    success = cv2.imwrite(temp_filename, imageEncrypted)
        # Create a temporary file to upload to Cloudinary
    if not success:
        return JSONResponse(content={"error": "Failed to save encrypted image"}, status_code=401)

    try:
        # Upload the encrypted image to Cloudinary
        upload_result = cloudinary.uploader.upload(
          temp_filename,
          folder="encrypted_images",
          resource_type="image"
        )
        response_to_send = {
            "encrypted_image_url": upload_result["secure_url"],
        }
        if isSteg:
            # Hide decrypted image in carrier image
            try :

                # Process the image
                stegImageUnprocessed = await stegImage.read()
                stegImageUnprocessedPath = f"steg_image_{os.urandom(8).hex()}.png"
                with open(stegImageUnprocessedPath, "wb") as f:
                    f.write(stegImageUnprocessed)
                # use path for encoding
                # read encrypted image
                global encryptedImage
                with open(temp_filename, "rb") as f:
                    encryptedImage = f.read()
                # debug encode data


                steg_img = encode(stegImageUnprocessedPath, encryptedImage, n_bits=4)
                if steg_img is None:
                    return JSONResponse(content={"error": "Failed to hide encrypted image in carrier image"}, status_code=500)
                temp_filename_steg = f"steg_image_{uuid.uuid4()}.png"
                cv2.imwrite(temp_filename_steg, steg_img)
                upload_result_steg = cloudinary.uploader.upload(
                    temp_filename_steg,
                    folder="steg_images",
                    resource_type="image"
                )
                response_to_send["steg_image_url"] = upload_result_steg["secure_url"]

            #     clean up
            #     os.unlink(temp_filename_steg)
            #     os.unlink(stegImageUnprocessedPath)
            except Exception as e2:
                print(e2)
                return JSONResponse(content={"error": f"Failed to hide encrypted image in carrier image: {str(e2)}"}, status_code=500)
            finally:
                os.unlink(encryptedImage)
        # Return the new Cloudinary UR
        # L
        return JSONResponse(content=response_to_send)
    except Exception as e:
        print(e)
        return JSONResponse(content={"error": f"Failed to upload to Cloudinary: {str(e)}"}, status_code=500)
    finally:
        pass
        # Clean up the temporary file
        # if os.path.exists(temp_filename):
        #     os.remove(temp_filename)

from pydantic import BaseModel

class DecryptForm(BaseModel):
    image: UploadFile = File(...)
    password: str = Form(...)
    iv: str = Form(...)
    format: str = Form(...)


@app.post("/decrypt-img/")
async def d_img(
    image: UploadFile = File(...),
    password: str = Form(...),
    iv: str = Form(...),
    format: str = Form(...)
    ):
    # Read the encrypted image
    return JSONResponse({
        "image": image.filename
    })

@app.post("/decrypt-image/")
async def decrypt_image(
    image: UploadFile = File(...),
    password: str = Form(...),
    iv: str = Form(...),
    mode: str = Form(...),
    format: str = Form(...),
    isSteg: bool = Form(...)
    ):
    # Fetch the image from URL
    # response = requests.get(image_url)
    # if response.status_code != 200:
    #     return JSONResponse(content={"error": "Failed to fetch image from URL"}, status_code=400)
    #
    mode = AES.MODE_CBC
    ivSize = AES.block_size if mode == AES.MODE_CBC else 0

    global image_data, imageEncrypted
    # Process the downloaded image
    try :
        image_data = await image.read()
        # do processing for steg image
        if isSteg == True:

            #     store image in disk
            steg_unprocessed = "steg_image" + os.urandom(8).hex() + ".png"
            with open(steg_unprocessed, "wb") as f:
                f.write(image_data)
            #     pass path to decode function
            decoded_image = decode(steg_unprocessed, n_bits=4, in_bytes=True)
            if decoded_image is None:
                return JSONResponse(content={"error": "Failed to decode the image"}, status_code=500)
            temp_filename = f"decrypted_image_{os.urandom(8).hex()}.bmp"
            with open(temp_filename, "wb") as f:
                f.write(decoded_image)
            imageEncrypted = cv2.imread(temp_filename, flags=cv2.IMREAD_ANYCOLOR)

        else:
            nparr = np.frombuffer(image_data, np.uint8)
            imageEncrypted = cv2.imdecode(nparr, cv2.IMREAD_ANYCOLOR)

    except Exception as e:
        print(e)
        return JSONResponse(content={"error": f"Failed to read image: {str(e)}"}, status_code=500)





    rowEncrypted, columnOrig, depthOrig = imageEncrypted.shape
    rowOrig = rowEncrypted - 1
    encryptedBytes = imageEncrypted.tobytes()
    iv = encryptedBytes[:ivSize]
    imageOrigBytesSize = rowOrig * columnOrig * depthOrig
    paddedSize = (imageOrigBytesSize // AES.block_size + 1) * AES.block_size - imageOrigBytesSize
    encrypted = encryptedBytes[ivSize: ivSize + imageOrigBytesSize + paddedSize]

    password = str.encode(password)
    # iv = str.encode(iv)

    cipher = AES.new(password, AES.MODE_CBC, iv) if mode == AES.MODE_CBC else AES.new(password, AES.MODE_ECB)
    decryptedImageBytesPadded = cipher.decrypt(encrypted)
    # decryptedImageBytes = decryptedImageBytesPadded[:len(decryptedImageBytesPadded) - 8]
    decryptedImageBytes = unpad(decryptedImageBytesPadded, AES.block_size)

    # Convert bytes to decrypted image data
    decryptedImage = np.frombuffer(decryptedImageBytes, imageEncrypted.dtype).reshape(rowOrig, columnOrig, depthOrig)

    # Save the decrypted image to a BMP file in memory

    print(format.split("/")[-1])
    is_success, buffer = cv2.imencode("."+format.split("/")[1], decryptedImage)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type=format)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)