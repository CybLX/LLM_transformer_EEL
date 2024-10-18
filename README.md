# Text Generation Model

## Overview
This project implements a text generation model using a Transformer architecture, specifically focusing on the decoder layers. The model is trained to understand and generate human-like text based on a provided prompt. Unfortunately, after including a Pytorch dataset and dataloader, the model has been suffering from overfitting. If you want to see the original inspiration for the model, follow the [link](https://github.com/wingedsheep/transformer). The overfitting will be fixed soon.

## Dataset Features
The first dataset contains comments from students at EEL USP regarding their professors and subjects from 2018 to 2022. However, for legal reasons, this dataset will not be disclosed, but rather the final product.
- **Comentarios:** Processed words from the original text.

The second datasets contain customer reviews, tweets, and other user-generated content, and they have been preprocessed to include several features:
- **Tokens (stemmed and lemmatized):** Processed words from the original text.
- **POS Tags:** Part-of-speech tagging to identify the syntactic roles of words.
- **Syntactic Dependency:** Information about how words are related to each other in the sentence.
- **Polarity:** Sentiment analysis to classify whether the comments are positive or negative.

**IMPORTANT:** Due to the dataset consisting of 1.6 GB of comments, it was compressed into a .zip file along with preprocessing/cleaning scripts, i will set up Git Large File Storage (LFS) soon. Below is a description of each dataset used and their links. Feel free to download them and run the preprocessing scripts:

### B2W:
- **Description**: B2W-Review01 contains over 130,000 customer reviews collected from Americanas.com between January and May 2018.
- **Source**: [B2W Reviews01](https://github.com/americanas-tech/b2w-reviews01)

### Buscapé:
- **Description**: A dataset with over 80,000 product reviews tracked in 2013.
- **Source**: [Buscapé Reviews](https://drive.google.com/file/d/1IZJuvt1uxQ4oPGAvGQQxQ_h_ZiV-Be72/view)

### OLIST:
- **Description**: A dataset containing 100,000 orders from 2016 to 2018 across various marketplaces.
- **Source**: [Brazilian E-commerce Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_order_items_dataset.csv)

### Twitter:
- **Description**: A dataset of 890,000 tweets, both topic-based and general, collected between August and October 2018.
- **Source**: [Portuguese Tweets for Sentiment Analysis](https://www.kaggle.com/datasets/augustop/portuguese-tweets-for-sentiment-analysis)

### UTL Corpus:
- **Description**: A corpus with 2,881,589 reviews on movies and smartphone apps.
- **Source**: [UTL Corpus](https://github.com/RogerFig/UTLCorpus)

In this case, we only use the tokens.

## Project Goals
The main objectives of this project are:
- To develop a text generation model capable of producing coherent and contextually relevant text based on an initial prompt.
- To implement and understand the various components of a Transformer model, including token embeddings, positional encoding, self-attention mechanisms, and layer normalization.
- To demonstrate the model's ability to generate text through an autoregressive process.

## Tools Used
- **matplotlib:** A plotting library for creating static, interactive, and animated visualizations in Python.
- **nltk:** The Natural Language Toolkit, used for working with human language data (text) and performing various NLP tasks.
- **numpy:** A fundamental package for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices.
- **pandas:** A powerful data manipulation and analysis library that provides data structures like DataFrames for handling structured data.
- **spacy:** An NLP library designed for performance and ease of use, providing tools for tasks like tokenization, part-of-speech tagging, and named entity recognition.
- **symspellpy:** A fast spell checking library that provides efficient methods for finding and correcting spelling errors in text.
- **torch:** The core library for building and training neural networks in Python, providing support for tensors and GPU acceleration.
- **tqdm:** A library for creating progress bars in Python, making it easier to monitor the progress of loops and long-running tasks.
- **Unidecode:** A library that helps in converting Unicode text to ASCII, making it useful for processing text with special characters.

## How to Use
1. **Model Training:**
   - Create a tokenizer to convert your dataset into tokens.
   - Initialize the autoregressive language model with the desired maximum sequence length (e.g., 20).
   - Prepare your dataset by splitting the training text into sequences and padding them appropriately.
   - Train the model for a specified number of epochs (e.g., 50) with a chosen batch size (e.g., 8).
   - Monitor the loss value to ensure the model is learning.

2. **Text Generation:**
   - Switch the model to evaluation mode to disable dropout.
   - Use the `Generator` class to generate text by providing a prompt (e.g., "elephants").
   - The model will continue generating tokens autoregressively until it reaches the maximum character limit or encounters an end-of-sequence (eos) token.

3. **Saving and Loading the Model:**
   - Implement functions within the `LanguageModel` class to save the trained model to disk and load it for future use without retraining.

## For More Information
For further details on the model architecture and implementation, please refer to the code comments within the `DecoderLayer`, `DecoderStack`, and `LanguageModel` classes. Additionally, feel free to reach out with any questions or feedback regarding the project.

For more information, codes, tutorials, and exciting projects, visit the links below:

- Email: alves_lucasoliveira@usp.br
- GitHub: [cyblx](https://github.com/cyblx)
- LinkedIn: [Cyblx](https://www.linkedin.com/in/cyblx)
