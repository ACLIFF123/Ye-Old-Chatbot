from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 


class Chatbot:
    def __init__(self):
        self.model_name = "microsoft/DialoGPT-small"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.chat_history_ids = None

    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt")
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_token=True)
    
    def generate_reply(self, prompt: str) -> str:
        prompt_with_new_line = prompt + "\n"

        encoded = self.encode_prompt(prompt_with_new_line)

        new_input = encoded["input_ids"]


        if self.chat_history_ids is None:
            input_ids = new_input
        else:
            input_ids = torch.cat([self.chat_history_ids, new_input], dim=-1)
        
        output = self.model.generate(input_ids, pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
        temperature=0.9, # Higher = more randomness (range: ~0.7 to 1.2)
        top_p=0.8, # Nucleus sampling: picks from top tokens with cumulative probability <= top_p 
        top_k=50)# Only consider the top 50 most likely tokens 

        self.chat_history_ids = output
        reply_id = output[0, input_ids.shape[1]:]
        reply = self.decode_reply(reply_id.tolist()).strip()
       
        return reply
    
    def reset_history(self):
        self.chat_history_ids = None 


    


    
     