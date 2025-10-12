# 🩺 Pneumonia Detector — AI-Powered Chest X-Ray Analysis

An AI-powered web app that detects **Pneumonia** from **chest X-ray images** using a **Convolutional Neural Network (CNN)** built with **TensorFlow** and **Keras**, wrapped inside an interactive **Streamlit** interface.  
It also provides **Grad-CAM heatmaps** to visualize which parts of the lungs influenced the model’s prediction.

---

## 🚀 Features
- 🔍 Real-time **Pneumonia detection** from X-ray images  
- 🧠 Custom **CNN model** trained on Kaggle’s Chest X-Ray dataset  
- 🌈 **Grad-CAM visualization** for explainable AI (XAI)  
- 💻 Interactive **Streamlit web app** for easy usage  
- 🧾 Upload your own image or test with built-in sample images  
- ⚡ Lightweight, fast, and runs fully **locally**

---

## 🧩 Tech Stack
| Category | Tools |
|-----------|-------|
| **Frameworks** | TensorFlow · Keras |
| **Frontend** | Streamlit |
| **Languages** | Python |
| **Visualization** | Matplotlib · OpenCV · Grad-CAM |
| **Deployment** | Streamlit / Localhost |

---

## 📁 Folder Structure
Pneumonia_Detector/
│
├── app.py # Streamlit web app
├── utils.py # Image preprocessing, prediction & Grad-CAM logic
├── pneumonia_cnn_model.h5 # Trained CNN model
├── requirements.txt # Dependencies list
├── sample_images/ # Example chest X-rays
│ ├── NORMAL/
│ └── PNEUMONIA/
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/Pneumonia_Detector.git
cd Pneumonia_Detector
2️⃣ Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate     # (Windows)
# or
source venv/bin/activate  # (Mac/Linux)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the Streamlit app
streamlit run app.py
Then open your browser at 👉 http://localhost:8501

🧠 Model Overview
The CNN model architecture includes:

3 Convolutional layers with ReLU activation

MaxPooling2D layers for downsampling

Dropout layers for regularization

Dense output layer with sigmoid activation for binary classification

Trained to distinguish Normal vs Pneumonia chest X-rays

🔬 Grad-CAM Visualization
Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the lung regions that most influenced the CNN’s decision.
This helps make the AI predictions interpretable and medically meaningful.

🩸 Red/Yellow = regions of higher pneumonia probability
💙 Blue = less significant regions

🧾 License
This project is open-sourced under the MIT License.

👤 Author
Fateh Mohammed Shariff
🎓 B.E. in Artificial Intelligence & Machine Learning
🔗 LinkedIn · GitHub