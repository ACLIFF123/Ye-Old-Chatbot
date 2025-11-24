from transformers import AutoTokenizer, AutoModelForCausalLM
import torch 


class Chatbot:
    def __init__(self):
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.chat_history_ids = None
        self.attention_mask = None
        self.system_prompt = self.system_prompt = "<|system|>\nYou are a helpful assistant.<|end|>\n"
        self.system_prompt_added = False 
    def encode_prompt(self, prompt: str):
        return self.tokenizer(prompt, return_tensors="pt").to(self.device)
    
    def decode_reply(self, reply_ids: list[int]) -> str:
        return self.tokenizer.decode(reply_ids, skip_special_tokens=True)
    
    def generate_reply(self, prompt: str) -> str:
        prompt_with_new_line = prompt + "\n"


        if not self.system_prompt_added:
            prompt_with_new_line = self.system_prompt + prompt_with_new_line
            self.system_prompt_added = True

        encoded = self.encode_prompt(prompt_with_new_line)

        new_input_ids = encoded["input_ids"].to(self.device)
        new_attention_mask = encoded["attention_mask"].to(self.device)

        if self.chat_history_ids is None:
            input_ids = new_input_ids
            attention_mask = new_attention_mask
        else:
            input_ids = torch.cat([self.chat_history_ids, new_input_ids], dim=-1)
            attention_mask = torch.cat([self.attention_mask, new_attention_mask], dim=-1)
            
        output = self.model.generate(input_ids,attention_mask=torch.ones_like(input_ids), pad_token_id=self.tokenizer.eos_token_id, do_sample=True, 
        temperature=0.8, # Higher = more randomness (range: ~0.7 to 1.2)
        top_p=0.9, # Nucleus sampling: picks from top tokens with cumulative probability <= top_p 
        top_k=50)# Only consider the top 50 most likely tokens 

        self.chat_history_ids = output
        self.attention_mask = torch.ones_like(output).to(self.device)
        reply_id = output[0, input_ids.shape[1]:]
        reply = self.decode_reply(reply_id.tolist()).strip()
       
        return reply
    
    def reset_history(self):
        self.chat_history_ids = None 
        self.system_prompt_added = False
        self.attention_mask = None
        
        




    
     