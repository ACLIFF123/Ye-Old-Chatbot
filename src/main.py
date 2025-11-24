from chatbot import Chatbot


def main():
    bot = Chatbot()

    print("===================================")
    print("Welcome to Your Chatbot!")
    print("===================================\n")

    
    print("System Prompt:")
    print(bot.system_prompt)
    print("-----------------------------------\n")

    while True:
        # Get user input
        user_input = input("User: ")

        # Exit condition
        if user_input.lower() in ["quit", "exit"]:
            print("\n Goodbye!")
            break

        # Get bot reply
        reply = bot.generate_reply(user_input)

        
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()