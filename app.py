import torch
import pickle
import numpy as np
import redis
from fastapi import FastAPI

app = FastAPI()
r = redis.Redis(host='localhost', port=6379, db=0)

NUM_USERS = 1000
NUM_ITEMS = 300

# Load model
from train_msv_model import LightGCN
model = LightGCN(NUM_USERS, NUM_ITEMS)
model.load_state_dict(torch.load("lightgcn.pth"))
model.eval()

users_final, items_final = model()

# Load MSV
with open("msv_vectors.pkl","rb") as f:
    food_vec, ideal_meal = pickle.load(f)

def compute_cart_vector(cart):
    return food_vec[cart].mean(axis=0)

@app.post("/add_to_cart/")
def add_to_cart(user_id: int, item_id: int):

    r.rpush(f"user:{user_id}:cart", item_id)

    cart = r.lrange(f"user:{user_id}:cart", 0, -1)
    cart = [int(i) for i in cart]

    cart_vec = compute_cart_vector(cart)
    gap = ideal_meal - cart_vec

    scores = food_vec @ gap

    topk = np.argsort(scores)[-5:].tolist()

    return {
        "recommended_items": topk
    }