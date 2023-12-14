import MalmoPython as malmo
import cv2
import numpy as np
import tensorflow as tf
import random
import json


class MinecraftClient:
    def __init__(self):
        self.agent_host = malmo.AgentHost()
        # 初始化观察空间和动作空间
        self.observation_space = None  # 这将在任务开始后根据实际屏幕尺寸更新
        self.action_space = ["move 1", "move -1", "strafe 1", "strafe -1", "attack 1"]  # 分别代表前进、后退、向左和向右

    def connect(self, mission_xml):
        try:
            mission = malmo.MissionSpec(mission_xml, True)
            mission_record = malmo.MissionRecordSpec()
            self.agent_host.startMission(mission, mission_record)

            # 等待任务开始
            print("Waiting for the mission to start", end=' ')
            world_state = self.agent_host.peekWorldState()
            while not world_state.has_mission_begun:
                print(".", end="")
                world_state = self.agent_host.peekWorldState()

            print("\nMission started")
            # 在任务开始后更新观察空间
            if world_state.number_of_video_frames_since_last_state > 0:
                frame = world_state.video_frames[0]
                self.observation_space = (frame.width, frame.height, 3)  # 图像的宽度、高度和颜色通道数

        except Exception as e:
            print(f"Error starting mission: {e}")

    def get_observation(self):
        world_state = self.agent_host.peekWorldState()
        if world_state.number_of_video_frames_since_last_state > 0:
            frame = world_state.video_frames[0].pixels
            return np.array(frame).reshape((world_state.video_frames[0].height, world_state.video_frames[0].width, 3))
        return None

    def send_action(self, action):
        self.agent_host.sendCommand(action)
    
    def received_video_frame(self):
        world_state = self.agent_host.peekWorldState()
        return world_state.number_of_video_frames_since_last_state > 0

# Image processing function remains the same
def process_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    return thresh
# Create an Option-Critic model class
class OptionCriticModel:
    def __init__(self, observation_shape, action_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=observation_shape),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(action_size, activation='softmax')
        ])

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self, state):
        probabilities = self.model.predict(np.array([state]))
        return probabilities

    def update(self, experience):
        # Randomly sample a batch from the memory replay
        batch = memory.sample(batch_size)
        for experience in batch:
            state, action, reward, next_state, done = experience

            # Calculate the target Q-value
            probabilities = self.model.predict(np.array([state]))
            action_probabilities = probabilities[0][action]
            next_probabilities = self.model.predict(np.array([next_state]))

            if done:
                target_q = reward
            else:
                max_next_q = np.max(next_probabilities)
                target_q = reward + discount_factor * max_next_q

            # Train using mean squared error loss function
            target_probabilities = np.zeros_like(probabilities[0])
            target_probabilities[action] = target_q

            self.model.train_on_batch(np.array([state]), target_probabilities)

# Create a memory replay class
class MemoryReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    def add(self, experience):
        if len(self.memory) >= self.capacity:
            self.memory.pop(0)
        self.memory.append(experience)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

# Create a Minecraft client instance and connect to the game
client = MinecraftClient()
client.connect(missionXML)

while not client.received_video_frame():
    pass
# Create an Option-Critic model instance
model = OptionCriticModel(client.observation_space, len(client.action_space))

# Create a memory replay instance
memory = MemoryReplay(capacity=10000)

# Training parameters
learning_rate = 0.001
discount_factor = 0.99
epsilon = 1.0
epsilon_decay = 0.999
batch_size = 32

# Start training
for episode in range(num_episodes=1000):
    state = client.get_observation()
    state = process_image(state)
    done = False
    total_reward = 0

    while not done:
        # Use epsilon-greedy strategy to select actions
        if np.random.rand() < epsilon:
            action = np.random.choice(client.action_space)
        else:
            probabilities = model.predict(np.array([state]))
            action = np.argmax(probabilities)

        # Perform the action in Minecraft
        client.send_action(action)

        # Get the new state, reward, and done information
        next_state = client.get_observation()
        next_state = process_image(next_state)
        reward, done = 0, False  # Replace with actual reward and done information from the game

        # Store the experience
        memory.add((state, action, reward, next_state, done))

        # Update the state
        state = next_state
        total_reward += reward

        # Train the model with a batch sampled from the memory replay every action
        if len(memory) > batch_size:
            batch = memory.sample(batch_size)
            for experience in batch:
                state, action, reward, next_state, done = experience

                # Calculate the target Q-value
                probabilities = model.predict(np.array([state]))
                action_probabilities = probabilities[0][action]
                next_probabilities = model.predict(np.array([next_state]))

                if done:
                    target_q = reward
                else:
                    max_next_q = np.max(next_probabilities)
                    target_q = reward + discount_factor * max_next_q

                # Train using mean squared error loss function
                target_probabilities = np.zeros_like(probabilities[0])
                target_probabilities[action] = target_q

                model.update((state, action, reward, next_state, done))

        # Update epsilon value
        epsilon = max(epsilon_decay * epsilon, 0.1)

    print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")

# Save the trained model
model.model.save("option_critic_model.h5")