import requests
import gradio as gr

def find_close_movies(text):
    if text is None or text.strip() == "":
        return "No text provided!"

    # Send text to API
    response = requests.post("http://flask-api-service:5000/predict_close_movies", data=text.encode('utf-8'))
    
    if response.status_code == 200:
        recommended_titles = response.json()
        return recommended_titles
    else:
        return "Error during the recommendation!"

# Gradio Interface
if __name__ == '__main__':
    gr.Interface(fn=find_close_movies, 
                 inputs="text",
                 outputs="text",  
                 live=True,
                 description="Enter a movie description to see the closest movies!",
                 ).launch(server_name="0.0.0.0", debug=True, share=True)
