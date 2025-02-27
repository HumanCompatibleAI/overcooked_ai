#!/usr/bin/env python
import pickle
import os

class RandomAI:
    def action(self, state):
        import numpy as np
        return np.random.choice(['up', 'down', 'left', 'right', 'stay', 'interact'])
    
    def reset(self):
        pass

if __name__ == "__main__":
    # Create agent directory
    os.makedirs('/app/data/agents/random_agent/agent', exist_ok=True)
    
    # Save the agent
    with open('/app/data/agents/random_agent/agent/agent.pickle', 'wb') as f:
        pickle.dump(RandomAI(), f)
    
    print("Agent created successfully!") 