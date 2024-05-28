from flask import Flask, render_template, request
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import random
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import logging

app = Flask(__name__, template_folder='C:/Users/vikas/OneDrive/Desktop/ai/templates')

# Set up logging
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Load your product dataset
# Replace 'path/to/your/product_data.csv' with your file path
try:
    product_data = pd.read_csv('C:/Users/vikas/OneDrive/Desktop/ai/p.csv')
except Exception as e:
    logging.error(f"Error loading product data: {e}")
    def error_loading_product_data():
        return "Error loading product data. Please check the file path."
    error_loading_product_data()

# Define a function to generate recommendations based on the specified scenario
def generate_recommendations(scenario, product_data):
    try:
        # Convert the scenario to lowercase for case-insensitive matching
        scenario_lower = scenario.lower()

        # Create a TF-IDF matrix for the product descriptions
        tfidf = TfidfVectorizer()
        tfidf_matrix = tfidf.fit_transform(product_data['Description'])

        # Compute the cosine similarity between the scenario and the product descriptions
        scenario_vector = tfidf.transform([scenario_lower])
        cosine_sim = cosine_similarity(scenario_vector, tfidf_matrix).flatten()

        # Find the index of the product with the highest cosine similarity
        most_similar_idx = cosine_sim.argsort()[-1]

        # Select the most similar product
        selected_product = product_data.iloc[most_similar_idx]

        return selected_product
    except Exception as e:
        logging.error(f"Error generating recommendations: {e}")
        return None

@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        if request.method == 'POST':
            scenario = request.form['scenario']

            # Generate recommendations based on the scenario
            selected_product = generate_recommendations(scenario, product_data)

            if selected_product is not None:
                # Load the fine-tuned GPT-2 model and tokenizer
                model_name = "gpt2"  # Use the pre-trained GPT-2 model
                fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_name)
                tokenizer = GPT2Tokenizer.from_pretrained(model_name)

                max_length = 150  # Increase the max_length for more detailed descriptions

                # Initialize the text generation pipeline with the updated max_length
                text_generator = pipeline("text-generation", model=fine_tuned_model, tokenizer=tokenizer, max_length=max_length)

                # Define a custom template for the AI description
                custom_template = "Discover the perfect {product_category} from {product_brand}. Crafted with {product_material}, this {product_description} in the stunning {product_color} color is a true standout. At just {product_price}, it's a steal that will elevate your style and keep you comfortable all day long. Available in size {product_size}, this piece is a must-have for your wardrobe."

                # Generate a custom AI description based on the product details and the template
                product_category = selected_product['Category']
                product_description = selected_product['Description']
                product_brand = selected_product['Brand']
                product_price = selected_product['Price']
                product_color = selected_product['Color']
                product_size = selected_product['Size']
                product_material = selected_product['Material']

                prompt_text = custom_template.format(
                    product_category=product_category,
                    product_description=product_description,
                    product_brand=product_brand,
                    product_price=product_price,
                    product_color=product_color,
                    product_size=product_size,
                    product_material=product_material
                )

                generated_text = text_generator(prompt_text, num_return_sequences=1)[0]['generated_text']

                ai_description = f"{prompt_text} {generated_text}"

                # Display the product details and AI description in a web page
                product_details = pd.DataFrame({
                    'Product ID': [selected_product['Product ID']],
                    'Category': [selected_product['Category']],
                    'Description': [selected_product['Description']],
                    'Brand': [selected_product['Brand']],
                    'Price': [selected_product['Price']],
                    'Color': [selected_product['Color']],
                    'Size': [selected_product['Size']],
                    'Material': [selected_product['Material']]
                })

                return render_template('index.html', scenario=scenario, product_details=product_details.to_html(index=False), ai_description=ai_description)
            else:
                return render_template('index.html', scenario=scenario, product_details=None, ai_description=None)

        return render_template('index.html')
    except Exception as e:
        logging.error(f"Error in index route: {e}")
        return "An error occurred. Please check the server logs for more information."

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)