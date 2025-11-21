from chatbot import Chatbot

bot = Chatbot()
# encoded = bot.encode_prompt("Hello, how are you?")
# reply = bot.decode_reply([15496, 703, 345, 30]) # Pass in a string of generated token IDs here from your tokenizer

# print(encoded) 
# print(reply)


prompt = "What is the weather like today?"
reply = bot.generate_reply(prompt)
print(f"Prompt: {prompt}")
print(f"Reply: {reply}")