#MAX_LEN is the allowed length for each sentence
MAX_LEN = 40
# #VOC_SIZE is the vocabulary size for our tokenizer
VOC_SIZE = 50265
class Question_pair():
  """
  the class used to encode the total information of a quora question pair
  """
  def __init__(self, id, question1, question2, is_duplicate):
    self.id = id
    self.question1 = question1
    self.question2 = question2
    self.is_duplicate = is_duplicate
    self.tokens1 = None
    self.tokens2 = None
    self.tokens_combined = None
  
  def tokenize_self(self, tokenizer, MAX_LEN):
    encoded_input1 = tokenizer(self.question1, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
    encoded_input2 = tokenizer(self.question2, max_length = MAX_LEN,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
    encoded_inputs = tokenizer(self.question1, self.question2, max_length = MAX_LEN*2,
                              truncation = True, padding = "max_length",
                              add_special_tokens = True, return_token_type_ids = False,\
                              return_attention_mask = False, return_tensors = 'pt')['input_ids'].flatten()
                          
    self.tokens1 = encoded_input1
    self.tokens2 = encoded_input2
    self.tokens_combined = encoded_inputs
