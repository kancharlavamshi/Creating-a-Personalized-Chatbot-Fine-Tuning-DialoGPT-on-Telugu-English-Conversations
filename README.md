# Creating-a-Personalized-Chatbot-Fine-Tuning-DialoGPT-on-Telugu-English-Conversations![chat](https://github.com/user-attachments/assets/efa549e9-16b8-4065-85b3-2eb02d91bb11)

This project demonstrates the development of a custom chatbot capable of engaging in personalized conversations in a hybrid language style—Telugu written in English. Using Microsoft’s DialoGPT as the foundation, we fine-tune the model on a handcrafted dataset to create a chatbot that delivers context-aware, natural, and meaningful interactions.

✨✨ The detailed description of this project has been posted on Medium blogs; [please take a look](https://medium.com/@vamshi.kancharla461/training-a-custom-chatbot-on-whatsapp-data-and-deploying-with-streamlit-9fcb5997a054)

## 2. Data Collection & Preprocessing
The success of any conversational AI lies in the quality of its dataset. For this project:
  Manual Dataset Creation:
     We curated a dataset of Telugu-English conversations to simulate realistic dialogues. 
  ### Example:
  ```
  Input: "Ela unnav?" (How are you?)
  Response: "Nenu bagunnanu, nuvvu ela unnaru?" (I’m fine, how are you?)
  ```
  ![chat1](https://github.com/user-attachments/assets/b892aef5-ba36-4f9c-8da0-84418d44f9a2)

  Dataset Cleaning:  
  Removed irrelevant and incomplete entries.
  Normalized punctuation and ensured proper alignment of inputs and responses.
  This process resulted in a high-quality dataset tailored for the chatbot’s intended use case.
  

## 3. Model Training
    To train the chatbot:   
    -- Model Choice  
    We selected DialoGPT (small version) for its efficiency and specialization in dialogue generation tasks.
    
    Fine-Tuning Process:
    Leveraged Hugging Face's AutoTokenizer and AutoModelForCausalLM.
    Custom tokens were added to preserve Telugu-English words (e.g., "Ela" as a single token instead of splitting into "E" and "la").
    Training was conducted using the Trainer API, with hyperparameters optimized for loss and conversational performance.
    
    Output:    
    The fine-tuned model was saved locally, incorporating updated vocabulary for smooth deployment.

    ```
    #Use the following command to start training:
    python train_chatbot.py --data_path data/chatbot_data.json --output_dir results/ --epochs 50 --batch_size 16
    ```
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

  ![chat2](https://github.com/user-attachments/assets/1fd40c75-2411-46a7-855b-010031c00bc6)
  
## 5. Unique Features
  This chatbot stands out due to:
  Hybrid Language Support:
  Specifically designed for Telugu-English conversations, a linguistic style rarely addressed by existing chatbots.
  
  Personalized Interactions:
  Trained on a handcrafted dataset, ensuring responses align with specific user preferences and conversational nuances.
  
  Contextual Awareness:
  Maintains conversational flow by remembering prior exchanges and adapting responses accordingly.
## 6. Inference Script
    ```
    #Start the chatbot using
    python run_inference.py --model_path results/ --max_length 128
    ```

## 7. Deployment
To make the chatbot accessible and user-friendly:
Streamlit Frontend:
A web interface for real-time user interaction with the chatbot.

Flask API:
Enables integration with other applications, ensuring versatility and scalability.

## [Youtube](https://youtu.be/0JLXgKYl3fo)

### Disclaimer:
This project was developed purely for educational purposes to showcase the capabilities of fine-tuning a language model using custom datasets and deploying it in real-world applications. The chatbot was trained on anonymized and fictional WhatsApp chat data and is not intended for any commercial use or to infringe upon the rights or privacy of any individuals or platforms. I do not claim ownership of any real-world data or conversations used in this project, and all data used for training is purely for educational experimentation.
