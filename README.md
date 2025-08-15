# Formula 1 Track Classifier 🏎️

An AI-powered tool that recognizes Formula 1 circuits from images using deep learning and provides fun facts about each track.  
Built on the NVIDIA Jetson Orin Nano, with a Flask-based web interface, and optionally paired with Ollama for AI-generated track trivia.

<img width="992" height="676" alt="Screenshot 2025-08-14 101358" src="https://github.com/user-attachments/assets/f670e333-3bab-4bce-b83c-5a156dfaa0dd" />


---

## 📌 The Algorithm

This project uses a **Convolutional Neural Network (CNN)** trained on images of Formula 1 track layouts.  

**Workflow:**
1. **Image Upload** – The user uploads a track layout image through the web interface.
2. **Preprocessing** – The image is resized to `224x224` pixels, normalized, and converted to an array.
3. **Prediction** – The CNN outputs the most probable track name from the trained dataset.
4. **Fun Facts (Optional)** – Ollama generates an interesting fact about the circuit.

**Model Details:**
- **Framework:** TensorFlow / Keras
- **Layers:** Conv2D + MaxPooling → Dense → Output softmax layer
- **Training Data:** Custom dataset of F1 track layouts (augmented with rotation, flips, zoom, and brightness adjustments)
- **Accuracy:** ~90% on validation set

**Tech Stack:**
- **Hardware:** NVIDIA Jetson Orin Nano
- **Backend:** Python + Flask
- **Model:** TensorFlow/Keras CNN
- **Extras:** Ollama for generating fun facts
- **Dataset:** Custom-collected images of Formula 1 track layouts

**Dependencies:**
```bash
tensorflow
flask
numpy
Pillow
ollama        # optional, only for track facts
opencv-python # optional, if you use OpenCV for preprocessing



```

DEMO VIDEO: https://youtu.be/frBz3SLUoXc 
 
