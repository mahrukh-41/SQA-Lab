# AI Intern Support Chatbot 🤖

## 📌 Project Overview

The **AI Intern Support Chatbot** is an NLP-based chatbot designed to provide quick and automated support to interns. It answers common intern queries using a predefined FAQ dataset and helps reduce the need for manual support.

The project uses **Natural Language Processing (NLP)** techniques to understand user questions and provide relevant responses.

## 🎯 Objective

The main objective of this project is to:

* Automate responses to common intern queries.
* Provide quick and consistent support.
* Reduce the workload of internship support teams.
* Improve the overall intern support experience.
* Use NLP techniques to match user questions with appropriate answers.

## 🛠️ Technologies Used

* Python
* Natural Language Processing (NLP)
* Pandas
* Scikit-learn
* TF-IDF Vectorization
* Cosine Similarity
* Google Colab
* CSV Dataset

## 📂 Dataset

The project uses an FAQ dataset containing common questions and their corresponding answers.

Example:

| Question                               | Answer                                                                                |
| -------------------------------------- | ------------------------------------------------------------------------------------- |
| How can I submit my internship report? | You can submit your report through the assigned submission portal.                    |
| How can I contact my mentor?           | You can contact your assigned mentor through the provided communication channel.      |
| When will I receive my certificate?    | Certificates are provided after successful completion of the internship requirements. |

The dataset can be expanded by adding more questions and answers.

## ⚙️ How It Works

The chatbot follows these basic steps:

1. Load the FAQ dataset.
2. Clean and prepare the questions.
3. Convert questions into numerical vectors using **TF-IDF**.
4. Take the user's query as input.
5. Convert the query into a TF-IDF vector.
6. Calculate similarity between the user's query and FAQ questions using **Cosine Similarity**.
7. Find the most relevant question.
8. Return the corresponding answer.
9. If the similarity is below the defined threshold, the chatbot informs the user that it could not find a suitable answer.

## 🔄 Workflow

```text
User Query
     ↓
Text Preprocessing
     ↓
TF-IDF Vectorization
     ↓
Cosine Similarity
     ↓
Find Most Relevant FAQ
     ↓
Generate Response
     ↓
Display Answer
```

## 🚀 Installation

Install the required Python libraries:

```bash
pip install pandas scikit-learn
```

## ▶️ Running the Project

### Step 1: Open Google Colab

Upload the project dataset to Google Colab.

### Step 2: Import Required Libraries

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```

### Step 3: Load the Dataset

```python
df = pd.read_csv("intern_faq_dataset.csv")
```

### Step 4: Train the TF-IDF Model

```python
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(df["question"])
```

### Step 5: Create the Chatbot

```python
def chatbot(user_query):

    query_vector = vectorizer.transform([user_query])

    similarity = cosine_similarity(query_vector, faq_vectors)

    best_match = similarity.argmax()
    score = similarity[0][best_match]

    if score < 0.2:
        return "Sorry, I could not find a suitable answer to your question."

    return df.iloc[best_match]["answer"]
```

### Step 6: Ask Questions

```python
while True:

    user_query = input("You: ")

    if user_query.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    response = chatbot(user_query)

    print("Chatbot:", response)
```

## 💡 Example

```text
You: How can I submit my internship report?

Chatbot: You can submit your report through the assigned submission portal.
```

Another example:

```text
You: How do I contact my mentor?

Chatbot: You can contact your assigned mentor through the provided communication channel.
```

## 📊 Key Features

* FAQ-based automated support
* NLP-based question matching
* TF-IDF text representation
* Cosine similarity for finding relevant answers
* Fast response generation
* Easy-to-update FAQ dataset
* Unknown-query handling
* Simple and lightweight implementation

## 📈 Future Improvements

The chatbot can be improved by:

* Using advanced Transformer models.
* Integrating Hugging Face Transformers.
* Adding Rasa for conversational AI.
* Supporting multiple languages.
* Adding voice input and output.
* Connecting the chatbot to a web interface.
* Integrating
