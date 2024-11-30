import gradio as gr
from PIL import Image
import requests
import io
import zipfile

def find_close_movies(image):
    if image is None:
        return "No image provided on the gradio interface!"
    
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="JPEG")

    # Send request to the API
    response = requests.post("http://flask-api-service:5000/predict_close_movies", data=img_binary.getvalue())
    if response.status_code == 200:
        img_io = io.BytesIO(response.content) 
        with zipfile.ZipFile(img_io, 'r') as zipf:
            zip_images = zipf.namelist()           
            images = []
            for zip_image in zip_images:
                with zipf.open(zip_image) as img_file:
                    img_to_display = Image.open(img_file)
                    img_to_display.load()  
                    images.append(img_to_display)  
            return images
    else:
        return "Error during the recommendation!"

if __name__ == '__main__':
    gr.Interface(fn=find_close_movies, 
                 inputs="image", 
                 outputs="gallery",
                 live=True,
                 description="Drop a movie poster to see the closest movies!",
                 ).launch(server_name="0.0.0.0", debug=True, share=True)