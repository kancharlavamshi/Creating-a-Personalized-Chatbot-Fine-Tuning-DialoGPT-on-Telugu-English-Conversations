# Creating-a-Personalized-Chatbot-Fine-Tuning-DialoGPT-on-Telugu-English-Conversations![chat](https://github.com/user-attachments/assets/efa549e9-16b8-4065-85b3-2eb02d91bb11)

This project demonstrates the development of a custom chatbot capable of engaging in personalized conversations in a hybrid language style—Telugu written in English. Using Microsoft’s DialoGPT as the foundation, we fine-tune the model on a handcrafted dataset to create a chatbot that delivers context-aware, natural, and meaningful interactions.

## 2. Data Collection & Preprocessing
The success of any conversational AI lies in the quality of its dataset. For this project:
  Manual Dataset Creation:
     We curated a dataset of Telugu-English conversations to simulate realistic dialogues. 
  ### Example:
  ```
  Input: "Ela unnav?" (How are you?)
  Response: "Nenu bagunnanu, nuvvu ela unnaru?" (I’m fine, how are you?)
  ```
  Dataset Cleaning:  
  Removed irrelevant and incomplete entries.
  Normalized punctuation and ensured proper alignment of inputs and responses.
  This process resulted in a high-quality dataset tailored for the chatbot’s intended use case.

## 3. Model Training
    To train the chatbot:   
    Model Choice:  
    We selected DialoGPT (small version) for its efficiency and specialization in dialogue generation tasks.
    
    Fine-Tuning Process:
    Leveraged Hugging Face's AutoTokenizer and AutoModelForCausalLM.
    Custom tokens were added to preserve Telugu-English words (e.g., "Ela" as a single token instead of splitting into "E" and "la").
    Training was conducted using the Trainer API, with hyperparameters optimized for loss and conversational performance.
    
    Output:    
    The fine-tuned model was saved locally, incorporating updated vocabulary for smooth deployment.
    
## 4. Custom Tokenization
    Why Custom Tokenization Matters:
    Telugu words written in English can lose meaning if split incorrectly.
    
    Example:
    ```
    Without Custom Tokens: "Ela" → "E", "la" (Loses meaning: “How”)
    With Custom Tokens: "Ela" → Retained as "Ela" (Meaning preserved: “How”)
    ```
    Custom tokens ensure:    
    Accurate interpretation of Telugu-English phrases.
    Improved contextual understanding.
    Consistent and meaningful responses.
## 5. Unique Features
  This chatbot stands out due to:
  Hybrid Language Support:
  Specifically designed for Telugu-English conversations, a linguistic style rarely addressed by existing chatbots.
  
  Personalized Interactions:
  Trained on a handcrafted dataset, ensuring responses align with specific user preferences and conversational nuances.
  
  Contextual Awareness:
  Maintains conversational flow by remembering prior exchanges and adapting responses accordingly.

## 6. Deployment
To make the chatbot accessible and user-friendly:
Streamlit Frontend:
A web interface for real-time user interaction with the chatbot.

Flask API:
Enables integration with other applications, ensuring versatility and scalability.
