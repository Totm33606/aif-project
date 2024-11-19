import gradio as gr
from PIL import Image
import requests
import io

def find_close_movies(image):
    if image is None:
        return "No image provided on the gradio interface!"
    
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="JPEG")

    # Send request to the API
    response = requests.post("http://flask-api-service:5000/predict_close_movies", data=img_binary.getvalue())
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return "Error during the recommandation!"

if __name__=='__main__':

    gr.Interface(fn=find_close_movies, 
                inputs="image", 
                outputs='image',
                live=True,
                description="Drop a movie poster to see the closest movie!",
                ).launch(server_name="0.0.0.0", debug=True, share=True)