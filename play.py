"""
Play against the trained X-agent (you are O).
"""
from env import TicTacToeEnv
from agent import QLearningAgent

def play():
    env = TicTacToeEnv()
    agent = QLearningAgent()
    agent.load("x_agent.pkl")

    print("\n=== Tic-Tac-Toe vs AI ===")
    print("You are O. Enter positions 0-8 (top-left → bottom-right).\n")
    print("Position map:\n0 1 2\n3 4 5\n6 7 8\n")

    state = env.reset()

    while True:
        env.render()

        if env.current_player == 1:   # AI (X)
            actions = env.available_actions()
            action  = agent.best_action(state, actions)
            print(f"AI plays: {action}")
        else:                         # Human (O)
            while True:
                try:
                    action = int(input("Your move: "))
                    if action in env.available_actions():
                        break
                    print("Invalid move, try again.")
                except ValueError:
                    print("Enter a number 0-8.")

        state, reward, done = env.step(action)

        if done:
            env.render()
            if reward == 1.0:
                winner = "X (AI)" if env.board[action] == 1 else "O (You)"
                print(f"🏆  {winner} wins!")
            else:
                print("It's a draw!")
            break

if __name__ == "__main__":
    play()
