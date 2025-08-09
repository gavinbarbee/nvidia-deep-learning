# Project Journal

---

## Day 1 – MNIST Intro & Neural Networks

### Summary
Today was my first deep dive into the NVIDIA course, starting with the MNIST dataset and building a neural network in PyTorch. 
I learned the foundational workflow for moving data to GPU, processing it into tensors, constructing input → hidden → output layers, and training/validating the model. 
I reached nearly 100% accuracy after five epochs and practiced cleaning up GPU memory after runs.

I also reinforced key AI history milestones:  
- 1950s: Von Neumann architecture & early computing concepts.  
- 1990s: Deep Blue defeating a chess champion; rise of more data-driven approaches like backgammon neural networks.  
- Shift from rule-based to trial-and-error learning.  
- The role of GPU parallel processing for matrix-heavy tasks like deep learning.  

This wasn’t just technical — I also connected how deep learning differs from traditional programming and where each approach excels.

---

### Key Takeaways
- **Training, Validation, Test Sets**: Clear understanding of their roles and why they must be split properly.  
- **Tensor Basics**: n-dimensional arrays for numerical image representation.  
- **Relu Activation**: Why it’s used to improve decision boundaries.  
- **Layers**: Flatten → Input → Hidden → Output → Loss function & optimizer.  
- **GPU Acceleration**: `.cuda()` moves work to GPU, massively speeding training.  

---

### What Stood Out Most
- The *hidden layer* analogy — it’s the “rules” that are too complex to explicitly code but are discovered through training.  
- Seeing the model approach near-perfect accuracy was a satisfying proof of concept.  
- The simplicity of the MNIST dataset made it perfect for understanding architecture before moving to harder problems.

---

### Ideas for Future Projects
- **Economic Sanctions AI**: Integrating structured trade datasets and unstructured text/news to flag possible sanction evasion.  
- **Political Sentiment Mapping**: Real-time tracking of public opinion via social media + NLP models.  
- **Satellite Image Analysis**: Detecting infrastructure development or military movements.  
- **Hybrid Forecast System**: Combine economic, military, and sentiment data to predict shifts in alliances or conflict escalation.  

---

**End of Day 1.** Tomorrow: Continue building on PyTorch fundamentals and move toward more complex datasets.
