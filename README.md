# Meal-State Gap Optimization (MSV) Recommendation System

> Zomathon Submission – Cart Super Add-On (CSAO) Rail  
> Target: <200ms Latency | +15% AOV Lift | Production-Ready  

---

## Overview

Traditional recommendation systems rely on “Frequently Bought Together” logic.  
This project introduces **Meal-State Gap Optimization (MSV)** — a state-aware recommendation engine that treats the cart as a dynamic, incomplete meal.

Instead of recommending popular items, we compute the gap between the current cart state and an ideal balanced meal state.

This enables intelligent suggestions that:

- Complete flavor balance
- Improve texture diversity
- Balance temperature
- Increase cart value
- Preserve delivery time constraints

---

## Core Innovation

### Meal-State Vector (MSV)

Each food item is encoded as a multi-dimensional vector representing:

- Flavor Profile (Spicy, Sweet, Tangy, Savory)
- Texture (Crunchy, Liquid, Soft)
- Temperature (Hot, Cold)
- Category (Main, Side, Beverage, Dessert)
- Heaviness (Light, Heavy)

### Meal Gap Score
Meal Gap Score = Ideal Balanced Meal Vector - Current Cart Vector


The recommender selects items that minimize this gap.

---

## System Architecture

### Model Layer
- LightGCN (Graph Neural Network)
- Bayesian Personalized Ranking (BPR) Loss
- CPU-optimized inference

### State Layer
- Redis for:
  - Session Cart Storage
  - Event Streaming
  - Real-time updates

### API Layer
- FastAPI deployment
- <200ms latency target

---

## Project Structure
├── train_msv_model.ipynb # Training pipeline (LightGCN + BPR)
├── app.py # FastAPI + Redis deployment
├── lightgcn.pth # Trained model weights
├── msv_vectors.pkl # MSV feature vectors
├── requirements.txt
├── LICENSE
└── README.md


---

## Installation

### Clone Repository
git clone https://github.com/subhamsinhadev/Meal-State-Gap-Optimization-MSV-Recommendation-System.git

cd msv-recommender


### Install Dependencies
pip install -r requirements.txt

Or manually:
pip install torch scipy pandas numpy scikit-learn fastapi uvicorn redis

### Start Redis
redis-server

---

## Train Model
train_msv_model.ipynb

This will generate:
- `lightgcn.pth`
- `msv_vectors.pkl`

---

## Run API
uvicorn app:app --reload

API endpoint:
POST /add_to_cart/?user_id=1&item_id=45

Example Response:

```json
{
  "recommended_items": [12, 78, 5, 199, 34]
}
```
Model Details
| Component      | Value  |
|----------------|--------|
| Embedding Size | 64     |
| Layers         | 3      |
| Loss           | BPR    |
| Optimizer      | Adam   |
| Target Latency | <200ms |

Business KPIs
- Attach Rate Improvement
- +15% AOV Lift Target
- Low Infrastructure Cost (< ₹0.05 per recommendation)
- Cold-Start Ready

How It Works
1. User adds item to cart
2. Redis stores cart state
3. MSV computes gap vector
4. LightGCN retrieves candidate embeddings
5. Gap-minimizing items returned
6. CSAO rail updated instantly

Tech Stack
- PyTorch
- LightGCN
- Redis
- FastAPI
- NumPy / Pandas
- Scikit-learn

Future Improvements
- Redis Vector Search integration
- LambdaMART ranking layer
- Real-time A/B testing framework
- Multi-restaurant graph modeling
- GPU acceleration option
