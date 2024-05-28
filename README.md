# Product Recommendation System with AI-Generated Descriptions

This Flask web application generates product recommendations based on user-provided scenarios and creates custom AI-generated product descriptions using a fine-tuned GPT-2 model, providing users with a more detailed and engaging experience.

## Key Features

1. **Product Recommendation**: The application takes a user-provided scenario as input and generates a product recommendation based on the similarity between the scenario and the product descriptions in the dataset.
2. **AI-Generated Product Description**: The application uses a fine-tuned GPT-2 model to generate a custom product description that incorporates the key details of the recommended product, such as category, brand, material, color, size, and price, enhancing the overall recommendation experience.

## Requirements

- Python 3.7 or higher
- Flask
- Pandas
- Transformers (Hugging Face)
- Scikit-learn
- OpenCV
- Matplotlib

## Usage

1. Clone the repository and navigate to the project directory.
2. Install the required packages using `pip install -r requirements.txt`.
3. Run the Flask application with `python app.py`.
4. Open your web browser and go to `http://localhost:5000/` to access the application.
5. Enter a scenario in the input field and click the "Get Recommendation" button to see the product recommendation and the AI-generated product description.

## Customization

- **Product Dataset**: Replace the `'C:/Users/vikas/OneDrive/Desktop/ai/p.csv'` file path with the location of your own product dataset.
- **GPT-2 Model**: Fine-tune the pre-trained GPT-2 model on your own dataset or use a different language model.
- **UI Design**: Modify the HTML template located in `'C:/Users/vikas/OneDrive/Desktop/ai/templates/index.html'` to change the appearance and layout of the web application.

## Logging

The application uses the `logging` module to log any errors that occur during the execution of the code. The log file is saved as `'app.log'` in the project directory.

## Potential Improvements

- **Personalization**: Incorporate user preferences or browsing history to tailor the generated descriptions to individual users.
- **Multimodal Generation**: Integrate image generation models to generate product images alongside the textual descriptions.
- **Contextual Awareness**: Enhance the description generation by incorporating additional context, such as user intent or product category.
- **Iterative Refinement**: Implement a feedback mechanism to allow users to provide ratings or comments, enabling the model to learn and improve over time.

By continuously improving the description generation feature, the product recommendation system can provide users with an even more engaging and informative experience, leading to better product discovery and increased customer satisfaction.
