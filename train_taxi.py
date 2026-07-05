import gymnasium as gym
import numpy as np
import random

def train_agent():
    # 1. Environment Setup 
    env = gym.make("Taxi-v4", render_mode="ansi")
    
    # Q-Table Initialize 
   
    state_size = env.observation_space.n
    action_size = env.action_space.n
    q_table = np.zeros([state_size, action_size])
    
    # 2. Hyperparameters
    total_episodes = 15000       
    alpha = 0.1                 
    gamma = 0.6                
    
    # Exploration parameters
    epsilon = 1.0               
    max_epsilon = 1.0
    min_epsilon = 0.01
    decay_rate = 0.0005         
    
    print("🤖 Autonomous Taxi Driver ki training shuru ho rahi hai...")
    print("Please wait, background mein automatic loop chal raha hai...\n")
    
    # 3.  TRAINING LOOP
    for episode in range(total_episodes):
        state, info = env.reset()
        done = False
        
        while not done:
            # Exploration vs Exploitation trade-off
            exp_exp_tradeoff = random.uniform(0, 1)
            
           
            if exp_exp_tradeoff < epsilon:
                action = env.action_space.sample()
            # Varna BEST predicted action uthao (Exploitation)
            else:
                action = np.argmax(q_table[state])
                
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Bellman Equation 
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            
            
            state = next_state
            
        
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
        
       
        if (episode + 1) % 3000 == 0:
            print(f"✅ Trip {episode + 1}/{total_episodes} completed. AI is learning...")

    print("\n🎉 Training Saffal Rahi! AI Taxi Driver ab expert ho chuka hai.")
    
    # 4. Save the Q-Table (Numpy Matrix File)
    # Yeh file hamare frontend/Streamlit mein kaam aayegi
    np.save("trained_q_table.npy", q_table)
    print("💾 Trained Brain saved as 'trained_q_table.npy'.")
    
    env.close()

if __name__ == "__main__":
    train_agent()