import sqlite3
import torch
import torch.nn as nn
import json
import re
from transformers import T5ForConditionalGeneration, T5Tokenizer

class HybridT5Model(nn.Module):
    """
    Modelo T5 customizado com uma cabeça de classificação paralela
    para predição de Frames e geração de argumentos estruturados.
    """
    def __init__(self, model_name="t5-small", num_frames=50):
        super(HybridT5Model, self).__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
        self.hidden_size = self.t5.config.d_model
        self.frame_classifier = nn.Linear(self.hidden_size, num_frames)
        
        # Mapeamento reverso dos IDs para os nomes reais de Frames
        self.id_para_frame = {i: "Sending" if i == 0 else f"Frame_{i}" for i in range(num_frames)}

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

    def predict(self, texto_input):
        """Executa a inferência real recebendo o texto formatado pelo NLTK."""
        self.eval()
        inputs = self.tokenizer(texto_input, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            frame_logits, encoder_outputs = self.forward(input_ids, attention_mask)
            frame_id = torch.argmax(frame_logits, dim=-1).item()
            frame_predito = self.id_para_frame.get(frame_id, "Unknown_Frame")
            
            outputs_gerados = self.t5.generate(
                encoder_outputs=encoder_outputs, 
                attention_mask=attention_mask,
                max_length=128,
                num_beams=2,
                early_stopping=True
            )
            
            # CORREÇÃO DEFINITIVA: Usa batch_decode e extrai o primeiro item da lista gerada pelo T5
            textos_decodificados = self.tokenizer.batch_decode(outputs_gerados, skip_special_tokens=True)
            texto_gerado = textos_decodificados[0] if textos_decodificados else "{}"
            
        # Força de forma redundante que a variável seja uma string limpa
        texto_gerado = str(texto_gerado).strip()
            
        try:
            dados_argumentos = json.loads(texto_gerado)
        except (json.JSONDecodeError, TypeError):
            # Fallback seguro caso o T5 não gere uma estrutura JSON válida na string decodificada
            dados_argumentos = {"Theme": [], "Recipient": []}
            theme_match = re.search(r"Theme\":\s*\[(.*?)\]", texto_gerado)
            recipient_match = re.search(r"Recipient\":\s*\[(.*?)\]", texto_gerado)
            
            if theme_match:
                dados_argumentos["Theme"] = [int(x) for x in theme_match.group(1).split(",") if x.strip().isdigit()]
            if recipient_match:
                dados_argumentos["Recipient"] = [int(x) for x in recipient_match.group(1).split(",") if x.strip().isdigit()]
                
        return {
            "frame": frame_predito,
            "arguments": dados_argumentos if isinstance(dados_argumentos, dict) else {}
        }

def buscar_detalhes_framenet(db_path, frame_name):
    """
    Consulta o banco SQLite adaptado à nova estrutura de tabelas:
    tabela 'frames' associada à tabela 'frame_elements'.
    """
    dados_linguisticos = {
        "acao": frame_name,
        "definicao": "Sem definição cadastrada.",
        "agente": "Não identificado",
        "objetos": [],
        "circunstancias": []
    }
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Busca a definição geral do Frame na tabela 'frames'
        cursor.execute("SELECT definicao FROM frames WHERE nome = ?", (frame_name,))
        row_frame = cursor.fetchone()
        if row_frame:
            dados_linguisticos["definicao"] = row_frame[0] # Pega a string de dentro da tupla
            
        # 2. Busca os Elementos do Frame (FEs) fazendo JOIN
        cursor.execute("""
            SELECT fe.nome_fe, fe.tipo_coreness 
            FROM frame_elements fe
            JOIN frames f ON fe.frame_id = f.id_frame
            WHERE f.nome = ?
        """, (frame_name,))
        
        rows_fe = cursor.fetchall()
        for nome_fe, tipo_coreness in rows_fe:
            nome_lower = nome_fe.lower()
            
            if "agent" in nome_lower or "protagonist" in nome_lower:
                dados_linguisticos["agente"] = nome_fe
            elif tipo_coreness == "Core" or "theme" in nome_lower or "item" in nome_lower:
                dados_linguisticos["objetos"].append(nome_fe)
            else:
                dados_linguisticos["circunstancias"].append(nome_fe)
                
        conn.close()
    except Exception as e:
        dados_linguisticos["notes"] = f"Erro ao acessar a estrutura do banco: {str(e)}"
        
    return dados_linguisticos
