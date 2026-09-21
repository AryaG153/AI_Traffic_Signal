import streamlit as st
from q_learning import QLearningAgent
from traffic_simulation import TrafficJunction


st.set_page_config(
    page_title="AI Traffic Signal Control",
    page_icon="🚦",
    layout="wide"
)

st.title("🚦 AI-Based Multi-Agent Traffic Signal Control")
st.caption("Multi-Agent Q-Learning Simulation")


# -----------------------------
# SESSION STATE
# -----------------------------

if "junctions" not in st.session_state:

    st.session_state.junctions = [
        TrafficJunction("Junction 1"),
        TrafficJunction("Junction 2"),
        TrafficJunction("Junction 3")
    ]

    st.session_state.agents = [
        QLearningAgent(),
        QLearningAgent(),
        QLearningAgent()
    ]

    st.session_state.step = 0
    st.session_state.total_delay = [0, 0, 0]


# -----------------------------
# CONTROL BUTTONS
# -----------------------------

col1, col2, col3 = st.columns(3)

with col1:
    start = st.button(
        "▶ Run Traffic Step",
        use_container_width=True
    )

with col2:
    train = st.button(
        "🧠 Train AI",
        use_container_width=True
    )

with col3:
    reset = st.button(
        "🔄 Reset Simulation",
        use_container_width=True
    )


# -----------------------------
# RESET
# -----------------------------

if reset:

    st.session_state.junctions = [
        TrafficJunction("Junction 1"),
        TrafficJunction("Junction 2"),
        TrafficJunction("Junction 3")
    ]

    st.session_state.agents = [
        QLearningAgent(),
        QLearningAgent(),
        QLearningAgent()
    ]

    st.session_state.step = 0
    st.session_state.total_delay = [0, 0, 0]

    st.rerun()


# -----------------------------
# TRAIN AI
# -----------------------------

if train:

    progress = st.progress(0)

    for episode in range(100):

        for i in range(3):

            junction = st.session_state.junctions[i]
            agent = st.session_state.agents[i]

            state = junction.get_state()

            action = agent.choose_action(state)

            junction.add_vehicles()

            junction.allow_vehicles(action)

            next_state = junction.get_state()

            cost = junction.queue

            agent.update(
                state,
                action,
                cost,
                next_state
            )

        progress.progress((episode + 1) / 100)

    st.success("AI training completed successfully!")


# -----------------------------
# RUN ONE TRAFFIC STEP
# -----------------------------

if start:

    st.session_state.step += 1

    for i in range(3):

        junction = st.session_state.junctions[i]
        agent = st.session_state.agents[i]

        # Current traffic state
        state = junction.get_state()

        # New vehicles arrive
        junction.add_vehicles()

        # AI chooses signal duration
        action = agent.choose_action(state)

        # Vehicles pass
        junction.allow_vehicles(action)

        # New state
        next_state = junction.get_state()

        # Traffic cost
        cost = junction.queue

        # Update Q-table
        agent.update(
            state,
            action,
            cost,
            next_state
        )

        # Store delay
        st.session_state.total_delay[i] += cost


# -----------------------------
# SIMULATION STATUS
# -----------------------------

st.divider()

st.subheader(
    f"🚦 Live Traffic Simulation — Step {st.session_state.step}"
)


# -----------------------------
# JUNCTION DISPLAY
# -----------------------------

columns = st.columns(3)

for i in range(3):

    junction = st.session_state.junctions[i]
    agent = st.session_state.agents[i]

    state = junction.get_state()

    action = agent.choose_action(state)

    with columns[i]:

        st.markdown(
            f"## 🚦 {junction.name}"
        )

        st.metric(
            "Vehicles Waiting",
            junction.queue
        )

        if state == "LOW":
            st.success("🟢 LOW TRAFFIC")

        elif state == "MEDIUM":
            st.warning("🟡 MEDIUM TRAFFIC")

        else:
            st.error("🔴 HIGH TRAFFIC")

        st.info(
            f"AI Signal Duration: **{action} seconds**"
        )


# -----------------------------
# PERFORMANCE
# -----------------------------

st.divider()

st.subheader("📊 AI Performance")

total_delay = sum(
    st.session_state.total_delay
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Simulation Steps",
        st.session_state.step
    )

with col2:
    st.metric(
        "Total Waiting Vehicles",
        sum(
            j.queue
            for j in st.session_state.junctions
        )
    )

with col3:
    st.metric(
        "Accumulated Delay",
        total_delay
    )


# -----------------------------
# Q TABLE
# -----------------------------

st.divider()

st.subheader("🧠 Learned Q-Tables")

for i, agent in enumerate(
    st.session_state.agents
):

    st.write(
        f"### Junction {i + 1}"
    )

    st.dataframe(
        {
            "10 sec": agent.q_table[:, 0],
            "20 sec": agent.q_table[:, 1],
            "30 sec": agent.q_table[:, 2]
        },
        hide_index=False
    )