# Module 1 – Intro to Deep Learning & MNIST

## **Big ideas (3 bullets max)**

* Deep learning differs from traditional programming by learning patterns from inputs and outputs rather than following explicit rules.
* GPUs excel at AI tasks because they can perform massive parallel matrix operations efficiently.
* Neural networks can be applied far beyond tech—into geopolitics, economics, and complex social modeling.

---

## **Terms & formulas (quick defs)**

* **Von Neumann Architecture** – Computing design where data and program instructions share the same memory space; foundational for modern computers.
* **Deep Blue** – IBM’s chess-playing computer that beat the world champion in 1997, showing the power of brute-force search in AI.
* **Backpropagation** – Method of adjusting neural network weights by propagating the error backward from output to input layers.
* **Parallel Processing** – Performing multiple calculations at once; key to GPU efficiency in AI training.
* **Computer Vision** – AI’s ability to interpret and process visual information from the world.
* **Natural Language Processing (NLP)** – AI’s ability to interpret, generate, and manipulate human language.
* **Reinforcement Learning** – Approach where agents learn by trial and error, guided by rewards or penalties.
* **Classical Programming** – Traditional method where explicit rules are given to the computer rather than learned from data.
* **Hidden Layer** – Layers of neurons between input and output that capture complex, abstract features.
* **Training Dataset** – Data used to fit the model.
* **Validation Dataset** – Data used for unbiased evaluation while tuning hyperparameters.
* **Test Dataset** – Data used for final unbiased evaluation after training is complete.
* **Epoch** – One complete pass through a dataset.
* **Tensor** – An n-dimensional array used to store numerical data, e.g., pixels of an image.

---

## **Lab steps I ran (ordered)**

1. Opened Jupyter Notebook, shut down unused kernels to free GPU memory.
2. Set device to CUDA for PyTorch (with CPU fallback if unavailable).
3. Loaded MNIST dataset: 60,000 training images, 10,000 validation images.
4. Converted images to tensors using `ToTensor()` and normalized to float32.
5. Explored dataset: verified shape `(1, 28, 28)` for grayscale images.
6. Applied transformations and batching (batch size = 32).
7. Defined neural network architecture:

   * Flatten layer to convert image tensor to vector.
   * Input layer, hidden layer (with ReLU activation), and output layer (10 classes).
8. Set up loss function and optimizer.
9. Trained for 5 epochs, tracking accuracy on training and validation sets.
10. Used `argmax` to get predicted class from model output.
11. Cleared GPU memory after run.

---

## **Code snippets I want to reuse**

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Flatten layer in PyTorch
nn.Flatten()

# Example model architecture
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, output_size)
).to(device)

---

## **My creative takeaways**

* This framework is highly adaptable: today it classifies digits, tomorrow it could classify political events, infrastructure vulnerabilities, or propaganda narratives.
* The real power lies not just in predicting outcomes but in dynamically adapting as new data changes the “rules” in real time.
* Reinforcement learning could be applied to political simulations, where the AI learns optimal policies under evolving constraints.

---

## **Possible applications in political science**

* **Conflict Escalation Prediction** – Model historical patterns of military buildups, sanctions, and diplomatic breakdowns to predict when crises may turn into armed conflict.
* **Regime Stability Forecasting** – Use economic, social, and political indicators to classify regimes as “stable,” “at risk,” or “transitioning.”
* **Migration Flow Modeling** – Predict refugee and migration flows during crises using economic data, climate signals, and satellite imagery.
* **Alliance Network Analysis** – Classify and map the shifting strength of alliances using trade, defense spending, and diplomatic events data.
* **Cyberattack Attribution** – Train a classifier on technical and geopolitical signatures to identify likely state or non-state actors behind cyber operations.
* **Propaganda Detection** – NLP models trained on political speeches and media coverage to detect narrative shifts, disinformation, and coordinated influence campaigns.
* **Aid Effectiveness Forecasting** – Predict which forms of foreign aid (military, economic, humanitarian) will achieve stated policy goals in specific contexts.

---

## **Ideas for future projects**

* Build a **geopolitical sanctions AI** that models historical sanctions and predicts their impact on trade flows, inflation, and regime change likelihood.
* Develop a **conflict early warning system** combining satellite imagery, trade data, and sentiment analysis from news/social media.
* Create a **political speech analyzer** that detects narrative changes, escalation tones, or hidden policy shifts over time.
* Implement a **migration risk index** that forecasts refugee flows for humanitarian planning.

