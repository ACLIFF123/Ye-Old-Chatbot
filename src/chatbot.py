from transformers import AutoTokenizer, AutoModelForCausalLM

class Chatbot:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_token=True)
    
    def generate_reply(self, prompt: str) -> str:
        prompt_with_new_line = prompt + "\n"

        inputs = self.encode_prompt(prompt_with_new_line)

        output = self.model.generate(**inputs, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
        temperature=0.8, # Higher = more randomness (range: ~0.7 to 1.2)
        top_p=0.5, # Nucleus sampling: picks from top tokens with cumulative probability <= top_p 
        top_k=20)[0] # Only consider the top 50 most likely tokens 

        decode = self.decode_reply(output)

        reply = decode[len(prompt_with_new_line):]

        return reply 




    
     