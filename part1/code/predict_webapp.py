import gradio as gr
from PIL import Image
import requests
import io

def find_genre(image):
    if image is None:
        return "No image provided on the gradio interface!"
    
    image = Image.fromarray(image.astype('uint8'))
    img_binary = io.BytesIO()
    image.save(img_binary, format="JPEG")

    # Send request to the API
    response = requests.post("http://flask-api-service:5000/predict", data=img_binary.getvalue())
    if response.status_code == 200:
        return response.text
    else:
        return "Error during the prediction!"

if __name__=='__main__':

    gr.Interface(fn=find_genre, 
                inputs=gr.Image(type="numpy"), 
                outputs=gr.Textbox(),
                live=True,
                description="Drop a poster to know the genre of the movie.",
                ).launch(server_name="0.0.0.0", debug=True, share=True)