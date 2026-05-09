from flask import Flask, render_template, request
from transformers import pipeline

app = Flask(__name__)

# NLP Detector: BERT model specialized for spotting fakes
detector = pipeline("text-classification", model="mrm8488/bert-tiny-finetuned-fake-news-detection")

# Gen AI Generator: GPT-2 model for creating text
generator = pipeline('text-generation', model='gpt2')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    fake_story = None
    
    if request.method == 'POST':
        user_text = request.form.get('input_text')
        action = request.form.get('action')
        
        if action == "Detect":
            # NLP Logic
            res = detector(user_text[:512])[0]
            label = "REAL" if res['label'] == 'LABEL_1' else "FAKE"
            prediction = f"This looks {label} ({round(res['score']*100, 2)}% confidence)"
            
        elif action == "Generate":
            # Gen AI Logic
            gen = generator(user_text, max_length=100, num_return_sequences=1)
            fake_story = gen[0]['generated_text']
            
    return render_template('index.html', prediction=prediction, fake_story=fake_story)

if __name__ == '__main__':
    app.run(debug=True)