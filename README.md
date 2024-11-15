Internship Report: Emotion Detection Model with ResNet
1. Introduction
During my internship at NullClass, I worked on the development of an emotion detection model aimed at recognizing various emotions in images. This project involved implementing a custom deep learning model using a ResNet18 architecture, training it on the CK+ dataset, and deploying it for real-time emotion recognition. The main goal was to build a robust, efficient model that could accurately detect emotions such as happiness, sadness, anger, and surprise, as well as deploy this model for real-time testing and usage.

2. Background
The need for emotion recognition systems has been growing across various industries, including security, customer service, and healthcare. Emotion detection can provide insights into human behavior and emotions, which can be used to improve interaction between machines and humans. For this internship, the focus was on utilizing deep learning techniques for classifying emotions from facial expressions in images. I worked with the CK+ dataset and developed a custom model using ResNet18 architecture to identify emotions in real-time.

3. Learning Objectives
My main objectives during this internship were:

To gain hands-on experience in deep learning, particularly in the area of emotion detection.
To develop and fine-tune a custom deep learning model using ResNet18.
To deploy the trained model using a GPU for efficient testing and prediction.
To integrate a label encoding mechanism for handling the multi-class output in emotion detection tasks.
To solve real-world challenges such as image preprocessing, model deployment, and performance optimization.
4. Activities and Tasks
Data Preprocessing: I started by preparing and preprocessing the CK+ dataset. This involved image resizing, normalization, and organizing the dataset into training and test sets.

Model Architecture: I developed a custom emotion recognition model based on ResNet18, removing pre-trained weights to ensure the model was trained from scratch on the given dataset.

Training: Using the preprocessed images, I trained the model on a GPU using PyTorch and FastAI. The training involved careful tuning of hyperparameters and regular monitoring of performance to avoid overfitting.

Testing and Evaluation: After training, I tested the model’s performance on a separate test set to assess accuracy and reliability.

Deployment: I developed a script to deploy the trained model, ensuring it worked effectively on new input data. Additionally, I used a label encoder to map the predicted outputs back to the original emotion classes.

Integration: The model was integrated into a functional application for real-time emotion detection. I ensured the application could process input images and display the predicted emotion.

5. Skills and Competencies
Throughout this internship, I developed a range of technical and soft skills:

Deep Learning: Gained proficiency in building and training deep learning models using PyTorch and FastAI.
Computer Vision: Improved my understanding of image preprocessing, feature extraction, and model architecture.
Programming: Enhanced my Python programming skills, particularly in working with libraries such as torch, pickle, and PIL.
Model Deployment: Learned how to deploy trained models and handle challenges such as GPU availability and batch processing.
Problem-Solving: Sharpened my ability to solve technical issues such as handling data imbalances and optimizing model performance.
6. Feedback and Evidence
Throughout the internship, I received continuous feedback from my supervisor at NullClass, which helped me improve the quality and efficiency of my work. Regular check-ins ensured I stayed on track with deadlines and provided valuable insights into the practical challenges of deploying AI models.

Evidence of my work includes:

The source code of the emotion detection model.
A trained model saved in pkl format and a label encoder.
Documentation and results of model testing, including accuracy on the test dataset.
7. Challenges and Solutions
Several challenges arose during the project, including:

GPU Availability: At times, I faced issues with GPU availability on local machines, which caused delays in model training. I addressed this by utilizing cloud-based environments like Google Colab, where GPU support is more readily available.

Data Preprocessing: One challenge was ensuring the dataset was properly preprocessed for use in a deep learning model. I had to experiment with different image normalization techniques to optimize model performance.

Model Performance: Initially, the model struggled with accuracy, but after fine-tuning the learning rate and batch size, I was able to achieve better results. I also implemented techniques such as early stopping to prevent overfitting.

8. Outcomes and Impact
By the end of the internship, I successfully developed an emotion detection model that achieved a high accuracy on the CK+ test dataset. The model was integrated into a simple Python script capable of classifying emotions in images. This project helped me build a deeper understanding of machine learning workflows and how to handle real-world challenges such as data preprocessing, model deployment, and evaluation.

The project has potential applications in various domains such as customer feedback analysis, mental health monitoring, and AI-driven interactive systems.

9. Conclusion
In conclusion, this internship provided me with a solid foundation in emotion recognition using deep learning. I developed important skills in model training, testing, and deployment, and I gained experience in solving technical challenges associated with machine learning projects. The project not only improved my understanding of deep learning and computer vision but also helped me grow as a problem solver and team player.

I am confident that the knowledge gained during this internship will be beneficial in my future career, particularly in AI and machine learning fields.
