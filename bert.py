import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
import networkx as nx
import spacy
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
from torch.utils.data._utils.collate import default_collate
from collections import Counter
import psutil  # NEW: For memory usage monitoring

# Dataset paths
DATASET_ROOT = Path(r"C:\Users\aasri\OneDrive\Desktop\NLP\legal_doc_extraction\dataset")
ANNOTATION_FILE = DATASET_ROOT / "annotations.json"

# Paths for IN-Abs dataset
PATHS = {
    'IN-Abs': {
        'test-data': {
            'judgement': DATASET_ROOT / "IN-Abs" / "test-data" / "judgement",
            'summary': DATASET_ROOT / "IN-Abs" / "test-data" / "summary"
        },
        'train-data': {
            'judgement': DATASET_ROOT / "IN-Abs" / "train-data" / "judgement",
            'summary': DATASET_ROOT / "IN-Abs" / "train-data" / "summary"
        }
    }
}

# Load spaCy
nlp = spacy.load("en_core_web_sm")

# Global T5 model and tokenizer
t5_model = None
t5_tokenizer = None

# NEW: Function to log memory usage
def log_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    print(f"Memory Usage: RSS={mem_info.rss / 1024**2:.2f} MB, VMS={mem_info.vms / 1024**2:.2f} MB")

# Generate sample annotations with forced balancing
def generate_sample_annotations():
    annotations = {}
    for split in ['train', 'test']:
        judgement_dir = PATHS['IN-Abs'][f'{split}-data']['judgement']
        files = sorted(os.listdir(judgement_dir))
        max_docs = 50 if split == 'train' else 100  # CHANGED: Reduced max_docs for train
        files = files[:max_docs]
        positive_files = []
        negative_files = []
        for filename in files:
            try:
                with open(judgement_dir / filename, 'r', encoding='utf-8') as f:
                    judgement = f.read().lower()
                doc = nlp(judgement)
                entities = [(ent.text.lower(), ent.label_) for ent in doc.ents][:10]
                relations = []
                person_indices = [i for i, e in enumerate(entities) if e[1] == 'PERSON']
                gpe_indices = [i for i, e in enumerate(entities) if e[1] == 'GPE']
                for p in person_indices[:2]:
                    for g in gpe_indices[:2]:
                        relations.append([p, g])
                label = 1 if any(k in judgement for k in ['granted', 'allowed']) and not any(k in judgement for k in ['dismissed', 'rejected']) else 0
                if label == 1:
                    positive_files.append((filename, entities, relations, label))
                else:
                    negative_files.append((filename, entities, relations, label))
            except Exception as e:
                print(f"Skipping file {filename}: {str(e)}")
                continue
        if split == 'train':
            max_neg = min(len(negative_files), 25)  # CHANGED: Adjusted for smaller dataset
            max_pos = min(len(positive_files), max_neg if max_neg < 25 else 25)
            selected = negative_files[:max_neg] + positive_files[:max_pos]
        else:
            selected = positive_files + negative_files
        for filename, entities, relations, label in selected:
            annotations[filename] = {
                'label': label,
                'entities': entities,
                'relations': relations[:5]
            }
    with open(ANNOTATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2)
    print(f"Generated sample annotations at {ANNOTATION_FILE}")

# Custom collate function
def custom_collate(batch):
    graphs = [item['graph'] for item in batch]
    judgement_texts = [item['judgement_text'] for item in batch]
    summary_texts = [item['summary_text'] for item in batch]
    gt_entities = [item['gt_entities'] for item in batch]
    gt_relations = [item['gt_relations'] for item in batch]
    other_data = [
        {k: v for k, v in item.items() if k not in ['graph', 'judgement_text', 'summary_text', 'gt_entities', 'gt_relations']}
        for item in batch
    ]
    collated = default_collate(other_data)
    collated['graph'] = graphs
    collated['judgement_text'] = judgement_texts
    collated['summary_text'] = summary_texts
    collated['gt_entities'] = gt_entities
    collated['gt_relations'] = gt_relations
    return collated

# Custom Dataset
class LegalDataset(Dataset):
    def __init__(self, split='train', max_docs=None):
        self.judgement_dir = PATHS['IN-Abs'][f'{split}-data']['judgement']
        self.summary_dir = PATHS['IN-Abs'][f'{split}-data']['summary']
        self.files = sorted(os.listdir(self.judgement_dir))
        if max_docs is not None:
            self.files = self.files[:max_docs]
        self.tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")
        self.spacy_cache = {}
        for filename in self.files:
            try:
                with open(self.judgement_dir / filename, 'r', encoding='utf-8') as f:
                    text = f.read()
                self.spacy_cache[filename] = nlp(text)
            except Exception as e:
                print(f"Skipping file {filename} in spacy_cache: {str(e)}")
                self.files.remove(filename)
        generate_sample_annotations()
        with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        labels = []
        relation_counts = []
        for filename in self.files:
            if filename in self.annotations:
                labels.append(self.annotations[filename].get('label', 1))
                relation_counts.append(len(self.annotations[filename].get('relations', [])))
            else:
                with open(self.summary_dir / filename, 'r', encoding='utf-8') as f:
                    summary = f.read()
                labels.append(1 if summary.strip() else 0)
                relation_counts.append(0)
        label_counts = Counter(labels)
        print(f"Label distribution for {split} (total {len(labels)}): {dict(label_counts)}")
        if len(label_counts) == 1:
            print("Warning: All labels are identical. Classification metrics may be unreliable.")
        avg_relations = np.mean(relation_counts) if relation_counts else 0
        print(f"Average relations per document: {avg_relations:.2f}")
        if avg_relations < 1:
            print("Warning: Few or no relations in annotations. RE metrics may be unreliable.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        with open(self.judgement_dir / filename, 'r', encoding='utf-8') as f:
            judgement = f.read()
        with open(self.summary_dir / filename, 'r', encoding='utf-8') as f:
            summary = f.read()

        inputs = self.tokenizer(judgement, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        t5_inputs = self.t5_tokenizer("summarize: " + judgement, truncation=True, padding='max_length', max_length=512, return_tensors='pt')
        t5_labels = self.t5_tokenizer(summary, truncation=True, padding='max_length', max_length=150, return_tensors='pt')

        graph, entity_features = self.create_graph(judgement, filename)

        if filename in self.annotations:
            gt_entities = self.annotations[filename].get('entities', [])
            gt_relations = self.annotations[filename].get('relations', [])
            label = self.annotations[filename].get('label', 1)
        else:
            judgement_doc = self.spacy_cache[filename]
            gt_entities = [(ent.text.lower(), ent.label_) for ent in judgement_doc.ents]
            gt_relations = [(i, i+1) for i in range(len(gt_entities)-1)]
            label = 1 if summary.strip() else 0

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            't5_input_ids': t5_inputs['input_ids'].squeeze(),
            't5_attention_mask': t5_inputs['attention_mask'].squeeze(),
            't5_labels': t5_labels['input_ids'].squeeze(),
            'graph': graph,
            'entity_features': torch.tensor(entity_features, dtype=torch.float),
            'label': torch.tensor(label, dtype=torch.long),
            'judgement_text': judgement,
            'summary_text': summary,
            'gt_entities': gt_entities,
            'gt_relations': gt_relations
        }

    def create_graph(self, text, filename):
        doc = self.spacy_cache[filename]
        G = nx.Graph()
        entities = [(ent.text.lower(), ent.label_) for ent in doc.ents]
        max_entities = 30  # CHANGED: Reduced to lower memory usage
        max_entity_tokens = 10
        entity_features = np.zeros((max_entities, max_entity_tokens))

        if not entities:
            G.add_node(0)
            return G, entity_features

        entities = entities[:max_entities]
        for i, (ent_text, _) in enumerate(entities):
            G.add_node(i, label=ent_text)
            tokens = self.tokenizer(ent_text, return_tensors='pt')['input_ids'].squeeze()
            tokens = tokens[:max_entity_tokens]
            if len(tokens) < max_entity_tokens:
                tokens = torch.cat([tokens, torch.zeros(max_entity_tokens - len(tokens), dtype=torch.long)])
            entity_features[i] = tokens.numpy()
        if filename in self.annotations and self.annotations[filename].get('relations'):
            for rel in self.annotations[filename]['relations']:
                if rel[0] < len(entities) and rel[1] < len(entities):
                    G.add_edge(rel[0], rel[1])
        else:
            for i in range(len(entities)-1):
                G.add_edge(i, i+1)

        return G, entity_features

# Model
class LegalExtractionModel(nn.Module):
    def __init__(self):
        super(LegalExtractionModel, self).__init__()
        self.bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.fc = nn.Linear(768 + 10, 512)
        self.classifier = nn.Linear(512, 2)
        self.re_head = nn.Linear(768 * 2, 1)

    def forward(self, input_ids, attention_mask, graph, entity_features, mode='classify'):
        bert_out = self.bert(input_ids, attention_mask=attention_mask)[0]
        bert_cls = bert_out[:, 0, :]

        batch_size = input_ids.size(0)
        graph_features = []
        for g in graph:
            degrees = [d for _, d in nx.degree(g)]
            if not degrees:
                degrees = [0]
            avg_degree = np.mean(degrees)
            graph_features.append([avg_degree] * 10)
        graph_features = torch.tensor(graph_features, dtype=torch.float).to(bert_cls.device)

        combined = torch.cat([bert_cls, graph_features], dim=-1)
        combined = F.relu(self.fc(combined))
        if mode == 're':
            re_scores = []
            for i, g in enumerate(graph):
                nodes = list(g.nodes)
                if len(nodes) < 2:
                    re_scores.append(torch.tensor([]).to(bert_cls.device))
                    continue
                nodes = nodes[:30]  # CHANGED: Enforce max_entities
                entity_embeds = bert_out[i, :len(nodes), :]
                pair_scores = []
                for j in range(len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        pair_embed = torch.cat([entity_embeds[j], entity_embeds[k]])
                        score = torch.sigmoid(self.re_head(pair_embed))
                        pair_scores.append(score)
                if pair_scores:
                    re_scores.append(torch.stack(pair_scores))
                else:
                    re_scores.append(torch.tensor([]).to(bert_cls.device))
            if not re_scores or all(s.numel() == 0 for s in re_scores):
                return torch.tensor([]).to(bert_cls.device)
            return torch.cat([s for s in re_scores if s.numel()]).squeeze(-1)
        logits = self.classifier(combined)
        return logits

# Generate summary
def generate_summary(t5_model, input_ids, attention_mask, device, tokenizer):
    inputs = t5_tokenizer("summarize: " + tokenizer.decode(input_ids[0], skip_special_tokens=True), return_tensors="pt", max_length=512, truncation=True).to(device)
    summary_ids = t5_model.generate(inputs["input_ids"], max_length=150, num_beams=4, early_stopping=True)
    return t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Visualize graphs
def visualize_ner_graph(graph, filename):
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color='lightblue', node_size=500, font_size=8)
    plt.title("NER Graph")
    plt.savefig(DATASET_ROOT / filename)
    plt.close()

def visualize_re_graph(graph, filename):
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'label')
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color='lightgreen', node_size=500, font_size=8)
    plt.title("Relation Extraction Graph")
    plt.savefig(DATASET_ROOT / filename)
    plt.close()

def visualize_sum_graph(summary_text, filename):
    sentences = summary_text.split('. ')
    G = nx.Graph()
    for i, sent in enumerate(sentences):
        if sent.strip():
            G.add_node(i, label=sent[:20])
    for i in range(len(sentences)-1):
        if sentences[i].strip() and sentences[i+1].strip():
            G.add_edge(i, i+1)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightcoral', node_size=500, font_size=8)
    plt.title("Summarization Graph")
    plt.savefig(DATASET_ROOT / filename)
    plt.close()

# Evaluation functions
def evaluate_ner(pred_entities, gt_entities):
    pred_set = set((ent[0].lower(), ent[1]) for ent in pred_entities)
    gt_set = set((ent[0].lower(), ent[1]) for ent in gt_entities)
    true_positives = len(pred_set & gt_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gt_set) if gt_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

def evaluate_re(pred_relations, gt_relations):
    pred_set = set(tuple(r) for r in pred_relations if r)
    gt_set = set(tuple(r) for r in gt_relations)
    true_positives = len(pred_set & gt_set)
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(gt_set) if gt_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall

def evaluate_summary(pred_summary, gt_summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(gt_summary, pred_summary)
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

# Save model with verification
def save_model(model, path):
    try:
        torch.save(model.state_dict(), path)
        if os.path.exists(path) and os.path.getsize(path) > 0:
            print(f"Model successfully saved to {path} (Size: {os.path.getsize(path) / 1024**2:.2f} MB)")
        else:
            print(f"Warning: Model file {path} is empty or does not exist")
    except Exception as e:
        print(f"Error saving model to {path}: {str(e)}")

# Load model for inference
def load_model(model_class, path, device):
    try:
        model = model_class().to(device)
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Model loaded from {path}")
        return model
    except Exception as e:
        print(f"Error loading model from {path}: {str(e)}")
        return None

# Zero-shot prediction
def zero_shot_predict(model, t5_model, test_loader, device, tokenizer):
    model.eval()
    t5_model.eval()
    cls_preds, cls_labels = [], []
    ner_metrics, re_metrics, sum_metrics = [], [], []
    for i, batch in enumerate(test_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        graph = batch['graph']
        entity_features = batch['entity_features'].to(device)
        labels = batch['label'].to(device)
        judgement_text = batch['judgement_text']
        summary_text = batch['summary_text']
        gt_entities = batch['gt_entities']
        gt_relations = batch['gt_relations']

        outputs = model(input_ids, attention_mask, graph, entity_features, mode='classify')
        pred = torch.argmax(outputs, dim=1)
        cls_preds.extend(pred.cpu().numpy())
        cls_labels.extend(labels.cpu().numpy())

        pred_entities = [(ent.text.lower(), ent.label_) for ent in nlp(judgement_text[0]).ents]
        ner_f1, ner_prec, ner_rec = evaluate_ner(pred_entities, gt_entities[0])
        ner_metrics.append((ner_f1, ner_prec, ner_rec))

        re_scores = model(input_ids, attention_mask, graph, entity_features, mode='re')
        nodes = list(graph[0].nodes)[:30]  # CHANGED: Enforce max_entities
        pred_relations = []
        if re_scores.numel():
            idx = 0
            for j in range(len(nodes)):
                for k in range(j + 1, len(nodes)):
                    if idx < len(re_scores) and re_scores[idx].item() > 0.5:
                        pred_relations.append((j, k))
                    idx += 1
        re_f1, re_prec, re_rec = evaluate_re(pred_relations, gt_relations[0])
        re_metrics.append((re_f1, re_prec, re_rec))

        pred_summary = generate_summary(t5_model, input_ids, attention_mask, device, tokenizer)
        sum_r1, sum_r2, sum_rl = evaluate_summary(pred_summary, summary_text[0])
        sum_metrics.append((sum_r1, sum_r2, sum_rl))

        if i == 0:
            visualize_ner_graph(graph[0], "ner_graph.png")
            visualize_re_graph(graph[0], "re_graph.png")
            visualize_sum_graph(pred_summary, "sum_graph.png")

        if i < 1:
            print(f"Sample {i}:")
            print("Predicted Summary:", pred_summary)
            print("Ground Truth Summary:", summary_text[0][:100], "...")
            print("Predicted Entities:", pred_entities[:5])
            print("Ground Truth Entities:", gt_entities[0][:5])
            print("Predicted Relations:", pred_relations[:5])
            print("Ground Truth Relations:", gt_relations[0][:5])

    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    cls_prec = precision_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    cls_rec = recall_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    ner_avg = np.mean(ner_metrics, axis=0)
    re_avg = np.mean(re_metrics, axis=0)
    sum_avg = np.mean(sum_metrics, axis=0)
    return (cls_f1, cls_prec, cls_rec), ner_avg, re_avg, sum_avg

# Training function
def train(model, t5_model, train_loader, optimizer, t5_optimizer, device, epochs=5):
    model.train()
    t5_model.train()
    class_weights = torch.tensor([1.0, 5.0]).to(device)
    accum_steps = 4  # NEW: Gradient accumulation steps
    for epoch in range(epochs):
        total_cls_loss = 0
        total_re_loss = 0
        total_sum_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            try:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                graph = batch['graph']
                entity_features = batch['entity_features'].to(device)
                labels = batch['label'].to(device)
                gt_relations = batch['gt_relations']
                t5_input_ids = batch['t5_input_ids'].to(device)
                t5_attention_mask = batch['t5_attention_mask'].to(device)
                t5_labels = batch['t5_labels'].to(device)

                optimizer.zero_grad(set_to_none=True)  # CHANGED: Optimize memory
                outputs = model(input_ids, attention_mask, graph, entity_features, mode='classify')
                cls_loss = F.cross_entropy(outputs, labels, weight=class_weights) / accum_steps
                cls_loss.backward()
                total_cls_loss += cls_loss.item() * accum_steps

                re_scores = model(input_ids, attention_mask, graph, entity_features, mode='re')
                nodes = list(graph[0].nodes)[:30]
                re_labels = []
                for j in range(len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        re_labels.append(1.0 if [j, k] in gt_relations[0] else 0.0)
                re_labels = torch.tensor(re_labels).to(device)
                if re_labels.numel() > 0 and re_scores.numel() == len(re_labels) and re_labels.sum() > 0:
                    print(f"Batch {batch_idx+1}: re_scores shape: {re_scores.shape}, re_labels shape: {re_labels.shape}")
                    re_loss = F.binary_cross_entropy(re_scores, re_labels) / accum_steps
                    re_loss.backward()
                    total_re_loss += re_loss.item() * accum_steps
                else:
                    print(f"Batch {batch_idx+1}: Skipping RE loss (re_labels.numel={re_labels.numel()}, re_scores.numel={re_scores.numel()}, re_labels.sum={re_labels.sum()})")
                    total_re_loss += 0.0

                t5_optimizer.zero_grad(set_to_none=True)
                t5_outputs = t5_model(
                    input_ids=t5_input_ids,
                    attention_mask=t5_attention_mask,
                    labels=t5_labels
                )
                sum_loss = t5_outputs.loss / accum_steps
                sum_loss.backward()
                total_sum_loss += sum_loss.item() * accum_steps

                if (batch_idx + 1) % accum_steps == 0:
                    optimizer.step()
                    t5_optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    t5_optimizer.zero_grad(set_to_none=True)

                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}, Cls Loss={cls_loss.item() * accum_steps:.4f}, RE Loss={(re_loss.item() * accum_steps) if re_labels.numel() > 0 and re_scores.numel() == len(re_labels) and re_labels.sum() > 0 else 0:.4f}, Sum Loss={sum_loss.item() * accum_steps:.4f}")
                log_memory_usage()  # NEW: Log memory after each batch
            except Exception as e:
                print(f"Error in batch {batch_idx+1}: {str(e)}")
                continue
        print(f"Epoch {epoch+1}: Avg Cls Loss={total_cls_loss / len(train_loader):.4f}, Avg RE Loss={total_re_loss / len(train_loader):.4f}, Avg Sum Loss={total_sum_loss / len(train_loader):.4f}")

# Evaluation function
def evaluate(model, t5_model, test_loader, device, tokenizer):
    model.eval()
    t5_model.eval()
    cls_preds, cls_labels = [], []
    ner_metrics, re_metrics, sum_metrics = [], [], []
    for i, batch in enumerate(test_loader):
        try:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            graph = batch['graph']
            entity_features = batch['entity_features'].to(device)
            labels = batch['label'].to(device)
            judgement_text = batch['judgement_text']
            summary_text = batch['summary_text']
            gt_entities = batch['gt_entities']
            gt_relations = batch['gt_relations']

            outputs = model(input_ids, attention_mask, graph, entity_features, mode='classify')
            pred = torch.argmax(outputs, dim=1)
            cls_preds.extend(pred.cpu().numpy())
            cls_labels.extend(labels.cpu().numpy())

            pred_entities = [(ent.text.lower(), ent.label_) for ent in nlp(judgement_text[0]).ents]
            ner_f1, ner_prec, ner_rec = evaluate_ner(pred_entities, gt_entities[0])
            ner_metrics.append((ner_f1, ner_prec, ner_rec))

            re_scores = model(input_ids, attention_mask, graph, entity_features, mode='re')
            nodes = list(graph[0].nodes)[:30]
            pred_relations = []
            if re_scores.numel():
                idx = 0
                for j in range(len(nodes)):
                    for k in range(j + 1, len(nodes)):
                        if idx < len(re_scores) and re_scores[idx].item() > 0.5:
                            pred_relations.append((j, k))
                        idx += 1
            re_f1, re_prec, re_rec = evaluate_re(pred_relations, gt_relations[0])
            re_metrics.append((re_f1, re_prec, re_rec))

            pred_summary = generate_summary(t5_model, input_ids, attention_mask, device, tokenizer)
            sum_r1, sum_r2, sum_rl = evaluate_summary(pred_summary, summary_text[0])
            sum_metrics.append((sum_r1, sum_r2, sum_rl))

            if i < 1:
                print(f"Sample {i}:")
                print("Predicted Summary:", pred_summary)
                print("Ground Truth Summary:", summary_text[0][:100], "...")
                print("Predicted Entities:", pred_entities[:5])
                print("Ground Truth Entities:", gt_entities[0][:5])
                print("Predicted Relations:", pred_relations[:5])
                print("Ground Truth Relations:", gt_relations[0][:5])
        except Exception as e:
            print(f"Error in evaluation batch {i}: {str(e)}")
            continue

    cls_f1 = f1_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    cls_prec = precision_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    cls_rec = recall_score(cls_labels, cls_preds, average='weighted', zero_division=0)
    ner_avg = np.mean(ner_metrics, axis=0)
    re_avg = np.mean(re_metrics, axis=0)
    sum_avg = np.mean(sum_metrics, axis=0)
    return (cls_f1, cls_prec, cls_rec), ner_avg, re_avg, sum_avg

# Main execution
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    global t5_model, t5_tokenizer
    t5_model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)
    t5_tokenizer = T5Tokenizer.from_pretrained("t5-small")

    supervised_train_dataset = LegalDataset(split='train', max_docs=50)  # CHANGED: Reduced max_docs
    few_shot_train_dataset = LegalDataset(split='train', max_docs=5)
    one_shot_train_dataset = LegalDataset(split='train', max_docs=1)
    test_dataset = LegalDataset(split='test')
    supervised_train_loader = DataLoader(supervised_train_dataset, batch_size=2, shuffle=True, collate_fn=custom_collate)
    few_shot_train_loader = DataLoader(few_shot_train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    one_shot_train_loader = DataLoader(one_shot_train_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=custom_collate)

    zero_shot_model = LegalExtractionModel().to(device)
    print("\nZero-Shot Evaluation:")
    cls_metrics_zs, ner_metrics_zs, re_metrics_zs, sum_metrics_zs = zero_shot_predict(zero_shot_model, t5_model, test_loader, device, tokenizer)
    print(f"Classification Metrics (F1, Precision, Recall): {cls_metrics_zs}")
    print(f"NER Metrics (F1, Precision, Recall): {ner_metrics_zs}")
    print(f"RE Metrics (F1, Precision, Recall): {re_metrics_zs}")
    print(f"Summarization Metrics (ROUGE-1, ROUGE-2, ROUGE-L): {sum_metrics_zs}")

    one_shot_model = LegalExtractionModel().to(device)
    one_shot_optimizer = torch.optim.Adam(one_shot_model.parameters(), lr=2e-5)
    t5_optimizer = torch.optim.Adam(t5_model.parameters(), lr=1e-4)
    print("\nOne-Shot Training:")
    train(one_shot_model, t5_model, one_shot_train_loader, one_shot_optimizer, t5_optimizer, device, epochs=5)
    print("\nOne-Shot Evaluation:")
    cls_metrics_os, ner_metrics_os, re_metrics_os, sum_metrics_os = evaluate(one_shot_model, t5_model, test_loader, device, tokenizer)
    print(f"Classification Metrics (F1, Precision, Recall): {cls_metrics_os}")
    print(f"NER Metrics (F1, Precision, Recall): {ner_metrics_os}")
    print(f"RE Metrics (F1, Precision, Recall): {re_metrics_os}")
    print(f"Summarization Metrics (ROUGE-1, ROUGE-2, ROUGE-L): {sum_metrics_os}")
    one_shot_model_path = DATASET_ROOT / "legal_extraction_one_shot.pt"
    save_model(one_shot_model, one_shot_model_path)
    t5_one_shot_path = DATASET_ROOT / "t5_finetuned_one_shot.pt"
    save_model(t5_model, t5_one_shot_path)

    few_shot_model = LegalExtractionModel().to(device)
    few_shot_optimizer = torch.optim.Adam(few_shot_model.parameters(), lr=2e-5)
    print("\nFew-Shot Training (5 documents):")
    train(few_shot_model, t5_model, few_shot_train_loader, few_shot_optimizer, t5_optimizer, device, epochs=5)
    print("\nFew-Shot Evaluation:")
    cls_metrics_fs, ner_metrics_fs, re_metrics_fs, sum_metrics_fs = evaluate(few_shot_model, t5_model, test_loader, device, tokenizer)
    print(f"Classification Metrics (F1, Precision, Recall): {cls_metrics_fs}")
    print(f"NER Metrics (F1, Precision, Recall): {ner_metrics_fs}")
    print(f"RE Metrics (F1, Precision, Recall): {re_metrics_fs}")
    print(f"Summarization Metrics (ROUGE-1, ROUGE-2, ROUGE-L): {sum_metrics_fs}")
    few_shot_model_path = DATASET_ROOT / "legal_extraction_few_shot.pt"
    save_model(few_shot_model, few_shot_model_path)
    t5_few_shot_path = DATASET_ROOT / "t5_finetuned_few_shot.pt"
    save_model(t5_model, t5_few_shot_path)

    supervised_model = LegalExtractionModel().to(device)
    supervised_optimizer = torch.optim.Adam(supervised_model.parameters(), lr=2e-5)
    print("\nSupervised Training (50 documents):")
    train(supervised_model, t5_model, supervised_train_loader, supervised_optimizer, t5_optimizer, device, epochs=5)  # CHANGED: Reduced epochs
    print("\nSupervised Evaluation:")
    cls_metrics_sup, ner_metrics_sup, re_metrics_sup, sum_metrics_sup = evaluate(supervised_model, t5_model, test_loader, device, tokenizer)
    print(f"Classification Metrics (F1, Precision, Recall): {cls_metrics_sup}")
    print(f"NER Metrics (F1, Precision, Recall): {ner_metrics_sup}")
    print(f"RE Metrics (F1, Precision, Recall): {re_metrics_sup}")
    print(f"Summarization Metrics (ROUGE-1, ROUGE-2, ROUGE-L): {sum_metrics_sup}")
    supervised_model_path = DATASET_ROOT / "legal_extraction_supervised.pt"
    save_model(supervised_model, supervised_model_path)
    t5_supervised_path = DATASET_ROOT / "t5_finetuned_supervised.pt"
    save_model(t5_model, t5_supervised_path)

    print("\nGraphs saved to:", DATASET_ROOT / "ner_graph.png", DATASET_ROOT / "re_graph.png", DATASET_ROOT / "sum_graph.png")

if __name__ == "__main__":
    main()