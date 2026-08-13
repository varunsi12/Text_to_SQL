"""Interactive CLI for the Text-to-SQL Agent. Run with: uv run cli (or python -m src.cli)"""
import sys
import pandas as pd

from src.agent import TextToSQLAgent


def main() -> None:
    print("=" * 70)
    print("  Fireworks Text-to-SQL CLI Agent (Chinook Database)")
    print("  Commands: '/reset' clears history | 'exit' or 'quit' exits")
    print("=" * 70 + "\n")

    # Initialize agent once so conversation history persists
    try:
        agent = TextToSQLAgent()
        print(f"Connected using model: {agent.model_id}\n")
    except Exception as e:
        print(f"Initialization Error: {e}")
        sys.exit(1)

    while True:
        try:
            user_input = input("Question > ").strip()

            if not user_input:
                continue

            # Exit commands
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break

            # Reset conversation memory
            if user_input.lower() == "/reset":
                agent.reset_conversation()
                print("--> Conversation context reset.\n")
                continue

            # Execute agent
            response = agent.query(user_input)

            print(f"\n[Generated SQL]\n{response['sql']}\n")

            if response["success"]:
                results = response["results"]

                if results:
                    df = pd.DataFrame(results)

                    print(f"[Query Results ({len(results)} rows)]")
                    print(df.to_string(index=False))
                else:
                    print(
                        "[Query Results] "
                        "Query executed successfully, but returned 0 rows."
                    )

                print(
                    f"\n[Telemetry] "
                    f"Retries Used: {response['retries']} | "
                    f"Latency: {response['latency_s']}s\n"
                )

            else:
                print(f"[Execution Error] {response['error']}\n")

        except (KeyboardInterrupt, EOFError):
            print("\nExiting CLI...")
            break


if __name__ == "__main__":
    main()