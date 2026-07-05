# 🚖 SmartCabbie: Autonomous Taxi Navigation via Q-Learning

An interactive, production-ready Deep Reinforcement Learning project that trains an autonomous taxi agent to safely pick up and drop off passengers in a grid world simulator. Features a custom **Graphical Emoji Dashboard** built with Streamlit to visualize the agent's real-time decisions, rewards, and environment telemetry.

---

## 👤 Internship Demographics
* **Full Name:** Gopal Goswami 
* **Intern ID:** CITS3236
* **Project Name:**SmartCabbie: Autonomous Taxi Navigation via Q-Learning
* **Domain:**  Machine Learning
* **Duration:** 4 Weeks

---

## 🚀 Features

* **Tabular Q-Learning Implementation:** Trains the taxi driver agent from scratch using the classic Bellman Optimality Equation over 15,000 learning episodes.
* **Graphical Emoji Map Handler:** Transforms OpenAI Gymnasium's raw text-based layout into a clean, human-readable UI utilizing live emojis (`🚖`, `👤`, `🎯`).
* **Live Telemetry & Telemetry Tracking:** Displays real-time calculations of steps taken, individual trip rewards, and a session-wide cumulative performance counter.
* **Deterministic Deployment:** Saves the optimal policy matrix into a standalone `.npy` binary file for low-latency edge deployment on the frontend dashboard.

---
## 📂 Images
  [Dashboard_UI][Images/dashboard.png]
  [Trip_MAP][Images/trip1_visual_map.png]
---

## 📂 Project Architecture

```text
SmartCabbie_RL_Project/
├── app.py              # Streamlit graphical frontend UI dashboard
├── README.md           # Documentation for the repository
├── requirements.txt    # Strict version control for required libraries
├── train_taxi.py       # Main reinforcement learning training loop script
└── trained_q_table.npy # Serialized NumPy binary of the agent's learned brain
