from transformers import BertTokenizer
import numpy as np
import torch
from torch import nn
from transformers import BertModel
from huggingface_hub import PyTorchModelHubMixin
from langchain.tools import tool


CLASS_TYPE_INFO = {
"1": ["The behavior of the epidemic curve does not match the expert's expectations.", {
"1": "Graph of the number of infected",
"2": "Graph of the number of susceptible",
"3": "Graph of the number of recovered",
"4": "Graph of the number of deaths"
}],
"2": ["Consideration of measures to counter the spread of the epidemic.", {
"1": "Temporal discrepancy in the effect of measures",
"2": "Incorrect reflection of the effects of measures",
"3": "No reflection of the effects of measures",
}],
"3": ["The forecast contains incorrect parameters or the temporal dynamics of the epidemiological process.", {
"1": "Time discrepancy",
"2": "Number discrepancy",
}],
"4": ["The position of the epidemic peak does not match Expert expectations.", {
"1": "Early/late peak",
"2": "Incorrect number of cases during peak",
}]
}

class BertClassifier(nn.Module, PyTorchModelHubMixin):

    def __init__(self, dropout=0.5):

        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained('bert-base-cased')
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, 15)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_id, mask):

        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.sigmoid(linear_output)

        return final_layer



def predict_text(model, tokenizer, text, device=None, top_k=2):
    """
    Predict top-k classes for input text

    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        text: Input text to classify
        device: Device to use (cuda/cpu)
        top_k: Number of top predictions to return

    Returns:
        Tuple of (top class indices, top class probabilities, all probabilities)
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    # Prepare input
    inputs = tokenizer(text,
                       padding='max_length',
                       max_length=128,
                       truncation=True,
                       return_tensors="pt")

    # Move to device
    input_ids = inputs['input_ids'].squeeze(1).to(device)
    attention_mask = inputs['attention_mask'].to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.sigmoid(outputs) if outputs.min(
        ) < 0 else outputs  # Apply sigmoid if logits

    # Convert to numpy
    probs = probs.cpu().numpy().flatten()

    # Get top-k predictions
    top_indices = np.argsort(probs)[-top_k:][::-1]  # Sort descending
    top_probs = probs[top_indices]

    return top_indices, top_probs, probs

def load_model():
    model = BertClassifier.from_pretrained(
        "Dinara777/epidemic_classificator_model")

    return model

def load_tokenizer():
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    return tokenizer


labels_level_main = {0: '1',
                     1: '1',
                     2: '2',
                     3: '3',
                     4: '4',
                     5: '2',
                     6: '1',
                     7: '2',
                     8: '3',
                     9: '3',
                     10: '1',
                     11: '2',
                     12: '4',
                     13: '1',
                     14: '2'
                     }

def predict_class_and_sub_class(text):
    # text = "The effect of quarantine is not reflected after day"
    model = load_model()
    tokenizer = load_tokenizer()
    top_indices, top_probs, all_probs = predict_text(model, tokenizer, text)
    print("Raw predictions:", top_indices)
    
    
    
    # Проверяем, что два топовых предсказания образуют пару класс-подкласс
    if len(top_indices) >= 2:
        pred1, pred2 = top_indices[0], top_indices[1]
        is_valid = validate_prediction(pred1, pred2)
        
        if not is_valid:
            print("ВНИМАНИЕ: Предсказанные классы не образуют корректную иерархию!")
            top_indices[0] = 1  # класс по умолчанию
            top_indices[1] = 1  # подкласс по умолчанию
        else:
            print("Предсказания образуют корректную иерархию классов")

    # Преобразуем индексы в метки уровней
    for i in range(len(top_indices)):
        top_indices[i] = labels_level_main[top_indices[i]]
    
    print("After level mapping:", top_indices)
    
    return top_indices, top_probs, is_valid

# Вспомогательные функции (добавьте их в ваш код)
class_hierarchy = {
    0: [1, 2, 3, 4],
    5: [6, 7, 8],
    9: [10, 11],
    12: [13, 14]
}

def check_class_subclass(pred1, pred2):
    """
    Проверяет, что два предсказанных значения составляют класс и подкласс
    """
    # Проверяем, является ли pred1 классом для pred2
    if pred1 in class_hierarchy and pred2 in class_hierarchy[pred1]:
        return True
    
    # Проверяем, является ли pred2 классом для pred1
    if pred2 in class_hierarchy and pred1 in class_hierarchy[pred2]:
        return True
    
    return False

def validate_prediction(pred1, pred2):
    """
    Проверяет валидность пары предсказаний и выводит ошибку при необходимости
    """
    if not check_class_subclass(pred1, pred2):
        print(f"ОШИБКА: значения {pred1} и {pred2} не образуют пару класс-подкласс")
        print(f"Допустимые пары:")
        for main_class, subclasses in class_hierarchy.items():
            for subclass in subclasses:
                print(f"  {main_class} -> {subclass}")
        return False
    else:
        print(f"✓ Корректная пара: {pred1} и {pred2} образуют класс и подкласс")
        return True
    
@tool(description="tool to get class,subclass names to expert comment")
def get_class_subclass_names(expert_comment) -> str:
    top_indices, top_probs, is_valid = predict_class_and_sub_class(expert_comment)
    comment_class = str(top_indices[0])
    comment_subclass = str(top_indices[1])

    if not is_valid:
        print("BAAAAAAAAAAD\nCan't determine class and subclass")
        return "Can't determine class and subclass"

    return f"expert_comment class is {CLASS_TYPE_INFO[comment_class][0]}, expert comment subclass is {CLASS_TYPE_INFO[comment_class][1][comment_subclass]}"