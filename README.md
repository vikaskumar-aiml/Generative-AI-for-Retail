# Product Recommendation System

This is a Flask web application that generates product recommendations based on a user-provided scenario and generates a custom AI-generated product description using a fine-tuned GPT-2 model.

## Features

1. **Product Recommendation**: The application takes a user-provided scenario as input and generates a product recommendation based on the similarity between the scenario and the product descriptions in the dataset.
2. **AI-Generated Product Description**: The application generates a custom AI-generated product description using a fine-tuned GPT-2 model, which is then displayed along with the product details.

## Requirements

- Python 3.7 or higher
- Flask
- Pandas
- Transformers (Hugging Face)
- Scikit-learn
- OpenCV
- Matplotlib

You can install the required packages using the following command:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```
git clone https://github.com/your-username/product-recommendation-system.git
```

2. Navigate to the project directory:

```
cd product-recommendation-system
```

3. Run the Flask application:

```
python app.py
```

4. Open your web browser and go to `http://localhost:5000/` to access the application.

5. Enter a scenario in the input field and click the "Get Recommendation" button to see the product recommendation and the AI-generated product description.

## Dataset

The application uses a product dataset stored in a CSV file located at `'C:/Users/vikas/OneDrive/Desktop/ai/p.csv'`. You can replace this file path with the location of your own product dataset.

## Customization

You can customize the following aspects of the application:

1. **Product Dataset**: Replace the `'C:/Users/vikas/OneDrive/Desktop/ai/p.csv'` file path with the location of your own product dataset.
2. **GPT-2 Model**: The application uses a pre-trained GPT-2 model for generating the product description. You can fine-tune the model on your own dataset or use a different language model.
3. **UI Design**: The HTML template located in `'C:/Users/vikas/OneDrive/Desktop/ai/templates/index.html'` can be modified to change the appearance and layout of the web application.

## Logging

The application uses the `logging` module to log any errors that occur during the execution of the code. The log file is saved as `'app.log'` in the project directory.


