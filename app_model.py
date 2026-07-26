import sqlite3
import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer

class HybridT5Model(nn.Module):
    """
    Modelo T5 customizado com uma cabeça de classificação paralela
    para predição de Frames e geração de argumentos estruturados.
    """
    def __init__(self, model_name="t5-small", num_frames=50):
        super(HybridT5Model, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.hidden_size = self.t5.config.d_model
        self.frame_classifier = nn.Linear(self.hidden_size, num_frames)

    def forward(self, input_ids, attention_mask, labels=None, frame_labels=None):
        encoder_outputs = self.t5.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        frame_logits = self.frame_classifier(pooled_output)
        
        if labels is not None and frame_labels is not None:
            outputs = self.t5(encoder_outputs=encoder_outputs, attention_mask=attention_mask, labels=labels)
            loss_generation = outputs.loss
            loss_fn = nn.CrossEntropyLoss()
            loss_classification = loss_fn(frame_logits, frame_labels)
            loss = loss_generation + loss_classification
            return loss, frame_logits, outputs
            
        return frame_logits, encoder_outputs

def buscar_detalhes_framenet(db_path, frame_name):
    """Consulta o banco SQLite local da FrameNet para mapear elementos de sintaxe."""
    dados_linguisticos = {
        "acao": frame_name,
        "agente": "Não identificado",
        "objetos": [],
        "circunstancias": []
    }
    
    try:
        # Abre a conexão com o banco de dados local
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query buscando os elementos da tabela frame_elements
        cursor.execute("""
            SELECT fe_name, fe_type 
            FROM frame_elements 
            WHERE frame_name = ?
        """, (frame_name,))
        
        rows = cursor.fetchall()
        for name, fe_type in rows:
            name_lower = name.lower()
            if "agent" in name_lower or "protagonist" in name_lower:
                dados_linguisticos["agente"] = name
            elif fe_type == "Core" or "theme" in name_lower or "item" in name_lower:
                dados_linguisticos["objetos"].append(name)
            else:
                dados_linguisticos["circunstancias"].append(name)
                
        conn.close()
    except Exception as e:
        dados_linguisticos["notas"] = f"Erro ao acessar banco de dados: {str(e)}"
        
    return dados_linguisticos
