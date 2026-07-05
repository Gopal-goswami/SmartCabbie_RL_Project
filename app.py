import streamlit as st
import gymnasium as gym
import numpy as np
import time
import re

# Page configuration
st.set_page_config(page_title="Smart Cabbie Dashboard", layout="wide")

st.title("🚖 SMART CABBIE | Graphical Autonomous Taxi Dashboard")
st.write("Layout Handling: Raw ANSI string converted to Visual Emoji Map with Strict Line Breaks!")

# 🛠️ LAYOUT HANDLER FUNCTION: Yeh text ko vertically align karega aur Emojis lagaye ka
# 🛠️ FULLY FIXED LAYOUT HANDLER FUNCTION (No more [42m or [43m junk)
def render_emoji_layout(raw_ansi_grid):
    lines = raw_ansi_grid.split('\n')
    new_lines = []
    
    for line in lines:
        # Layout spacing aur readability badhane ke liye dots replace karte hain
        line = line.replace(" : ", " . ")
        line = line.replace(":", ".")
        
        # 1. BINA PASSENGER WALI TAXI DETECT KARNA ([43m = Yellow background)
        if "[43m" in line:
            line = line.replace("[43m ", "🚖").replace("[43mR", "🚖").replace("[43mG", "🚖").replace("[43mY", "🚖").replace("[43mB", "🚖")
            
        # 2. PASSENGER KO BITHANE KE BAAD WALI TAXI DETECT KARNA ([42m = Green background)
        if "[42m" in line:
            line = line.replace("[42m ", "🚖").replace("[42mR", "🚖").replace("[42mG", "🚖").replace("[42mY", "🚖").replace("[42mB", "🚖").replace("[42m_", "🚖")

        # 3. PASSENGER DETECT KARNA (Bina taxi ke khada hai jab - [34;1m = Blue)
        if "[34;1m" in line:
            line = line.replace("[34;1mR", "👤").replace("[34;1mG", "👤").replace("[34;1mY", "👤").replace("[34;1mB", "👤")
            
        # 4. DESTINATION DETECT KARNA ([35m = Magenta)
        if "[35m" in line:
            line = line.replace("[35mR", "🎯").replace("[35mG", "🎯").replace("[35mY", "🎯").replace("[35mB", "🎯")
            
        # Saare bache-kuche ANSI junk strings ko bilkul clean kar do
        for junk in ["[0m", "[35m", "[34;1m", "[43m", "[42m", "\x1b"]:
            line = line.replace(junk, "")
            
        # Agar line khali nahi hai, toh hi add karo
        if line.strip():
            new_lines.append(line.strip())
        
    return "<br>".join(new_lines)

# 1. Load the trained Brain (Q-Table)
try:
    q_table = np.load("trained_q_table.npy")
    st.sidebar.success("💾 Trained Q-Table Loaded Successfully!")
except FileNotFoundError:
    st.sidebar.error("❌ 'trained_q_table.npy' nahi mili! Pehle terminal mein 'python train_taxi.py' chalayein.")
    st.stop()

# 2. Sidebar Controls
st.sidebar.header("Controls & Telemetry")
num_episodes = st.sidebar.slider("Kitne Passenger Trips dekhne hain?", 1, 10, 3)
delay = st.sidebar.slider("Animation Speed (Seconds)", 0.1, 1.0, 0.3)

# Session State for Overall Cumulative Reward tracking
if "overall_reward" not in st.session_state:
    st.session_state.overall_reward = 0

# Overall Score Counter
overall_metric_place = st.metric(label="🏆 Overall Cumulative Reward", value=st.session_state.overall_reward)

if st.sidebar.button("Launch Autonomous Taxi"):
    env = gym.make("Taxi-v4", render_mode="ansi")
    
    # Reset overall score on new launch
    st.session_state.overall_reward = 0
    overall_metric_place.metric(label="🏆 Overall Cumulative Reward", value=st.session_state.overall_reward)
    
    for ep in range(num_episodes):
        st.markdown(f"### 📍 Trip {ep + 1} Visual Map:")
        state, info = env.reset()
        done = False
        step_count = 0
        total_trip_reward = 0
        
        # Live HTML placeholders
        grid_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        while not done:
            # AI selects the best move
            action = np.argmax(q_table[state])
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_trip_reward += reward
            st.session_state.overall_reward += reward
            step_count += 1
            
            # Live master score update
            overall_metric_place.metric(label="🏆 Overall Cumulative Reward", value=st.session_state.overall_reward)
            
            # Action Names Mapping
            action_names = {0: "South ⬇️", 1: "North ⬆️", 2: "East ➡️", 3: "West ⬅️", 4: "Pickup 🧳", 5: "Dropoff 🏁"}
            current_action = action_names[action]
            
            # Get and handle layout
            raw_grid = env.render()
            emoji_grid_html = render_emoji_layout(raw_grid)
            
            # 🎯 STRICT VISUAL LAYOUT FRAMEWORK (Forcing Vertical Structure)
            html_layout = (
                "<div style='background-color: #1a1a2e; padding: 25px; border-radius: 12px; border: 2px solid #00fff0; width: fit-content; min-width: 280px; display: block; margin-bottom: 15px;'>"
                f"<div style=\"font-family: 'Courier New', Courier, monospace; font-size: 30px; font-weight: bold; color: #00ffcc; line-height: 1.6; display: block;\">{emoji_grid_html}</div>"
                "<hr style='border: 1px solid #333; margin: 15px 0;'>"
                "<p style='font-family: Arial, sans-serif; font-size: 18px; color: #ffffff; margin: 0; font-weight: bold;'>"
                f"🎬 Current Action: <span style='color: #ffaa00;'>{current_action}</span>"
                "</p>"
                "</div>"
            )
            
            grid_placeholder.markdown(html_layout, unsafe_allow_html=True)
            
            # Telemetry display below the box
            metrics_placeholder.markdown(f"""
            **Trip Telemetry:**
            * 🏁 Steps Taken: `{step_count}`
            * 💰 Step Reward: `{reward}`
            * 📊 Current Trip Score: `{total_trip_reward}`
            """)
            
            time.sleep(delay)
            state = next_state
            
        if total_trip_reward > 0:
            st.success(f"🎉 Success! Trip {ep+1} finished smoothly.")
        else:
            st.warning(f"⚠️ Trip {ep+1} finished, but taxi took some penalties.")
            
    env.close()