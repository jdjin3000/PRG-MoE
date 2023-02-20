import json
import torch
from transformers import BertTokenizer

def get_data(data_file, device, max_seq_len, contain_context=False):
    f = open(data_file)
    data = json.load(f)
    f.close()

    emotion_label_policy = {'angry': 0, 'anger': 0,
        'disgust': 1,
        'fear': 2,
        'happy': 3, 'happines': 3, 'happiness': 3, 'excited': 3,
        'sad': 4, 'sadness': 4, 'frustrated': 4,
        'surprise': 5, 'surprised': 5, 
        'neutral': 6}

    cause_label_policy = {'no-context':0, 'inter-personal':1, 'self-contagion':2, 'latent':3}

    if contain_context:
        preprocessed_utterance, max_doc_len, max_seq_len = load_utterance_with_context(data_file, device, max_seq_len)
    else:
        preprocessed_utterance, max_doc_len, max_seq_len = load_utterance(data_file, device, max_seq_len)


    doc_speaker, doc_emotion_label, doc_pair_cause_label, doc_pair_binary_cause_label = [list() for _ in range(4)]

    for doc_id, content in data.items():
        speaker, emotion_label, corresponding_cause_turn, corresponding_cause_span, corresponding_cause_label = [list() for _ in range(5)]
        content = content[0]

        pair_cause_label = torch.zeros((int(max_doc_len * (max_doc_len + 1) / 2), 4), dtype=torch.long) 
        pair_binary_cause_label = torch.zeros(int(max_doc_len * (max_doc_len + 1) / 2), dtype=torch.long)

        for turn_data in content:
            speaker.append(0 if turn_data["speaker"] == "A" else 1)

            emotion_label.append(emotion_label_policy[turn_data["emotion"]])

            corresponding_cause_label_by_turn = list()
            if "expanded emotion cause evidence" in turn_data.keys():

                corresponding_cause_per_turn = [_ - 1 if type(_) != str else -1 for _ in turn_data["expanded emotion cause evidence"]]
                corresponding_cause_turn.append(corresponding_cause_per_turn)

                for _ in corresponding_cause_per_turn: 
                    if _ == -1:
                        corresponding_cause_label_by_turn.append(cause_label_policy["latent"])
                    elif _ + 1 == turn_data["turn"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["no-context"])
                    elif content[_]["speaker"] == turn_data["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["self-contagion"])
                    elif content[_]["speaker"] != turn_data["speaker"]:
                        corresponding_cause_label_by_turn.append(cause_label_policy["inter-personal"])
                        
            corresponding_cause_label.append(corresponding_cause_label_by_turn)

        for idx, corresponding_cause_per_turn in enumerate(corresponding_cause_label):
            pair_idx = int(idx * (idx + 1) / 2)

            if corresponding_cause_per_turn:
                for cause_turn, cause in zip(content[idx]["expanded emotion cause evidence"], corresponding_cause_per_turn):
                    if type(cause_turn) == str:
                        continue
                    
                    cause_idx = int(cause_turn) - 1
                    pair_cause_label[pair_idx + cause_idx][cause] = 1
                    pair_binary_cause_label[pair_idx + cause_idx] = 1
        
        pair_cause_label[(torch.sum(pair_cause_label, dim=1) == False).nonzero(as_tuple=True)[0], 3] = 1

        doc_speaker.append(speaker)
        doc_emotion_label.append(emotion_label)
        doc_pair_cause_label.append(pair_cause_label)
        doc_pair_binary_cause_label.append(pair_binary_cause_label)
        
    out_speaker, out_emotion_label = [list() for _ in range(2)]
    out_pair_cause_label, out_pair_binary_cause_label = torch.stack(doc_pair_cause_label), torch.stack(doc_pair_binary_cause_label)

    for speaker, emotion_label in zip(doc_speaker, doc_emotion_label):
        speaker_t = torch.zeros(max_doc_len, dtype=torch.long)
        speaker_t[:len(speaker)] = torch.tensor(speaker)

        emotion_label_t = torch.zeros(max_doc_len, dtype=torch.long)
        emotion_label_t[:len(speaker)] = torch.tensor(emotion_label)

        out_speaker.append(speaker_t); out_emotion_label.append(emotion_label_t)

    out_speaker, out_emotion_label = torch.stack(out_speaker).type(torch.FloatTensor), torch.stack(out_emotion_label)

    return preprocessed_utterance, out_speaker.to(device), out_emotion_label.to(device), out_pair_cause_label.to(device), out_pair_binary_cause_label.to(device)

def load_utterance(data_file, device, max_seq_len):
    f = open(data_file)
    data = json.load(f)
    f.close()

    tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased')

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance = list()
    for doc_id, content in data.items():
        utterance = list()
        content = content[0]
        max_doc_len = max(len(content), max_doc_len)
            
        for turn_data in content:
            utterance.append(tokenizer_(turn_data["utterance"], padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))
            
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        padding_sequence = tokenizer_('', padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt")

        padding_sequence_t = [padding_sequence for _ in range(max_doc_len - len(utterance_t))]

        utterance_t = utterance_t + padding_sequence_t # shape: (max_doc_len, max_seq_len)
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        for _ in utterance_t:
            utterance_input_ids_t.append(_['input_ids'])
            utterance_attention_mask_t.append(_['attention_mask'])
            utterance_token_type_ids_t.append(_['token_type_ids'])

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)


    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return (out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device)), max_doc_len, max_seq_len

def load_utterance_with_context(data_file, device, max_seq_len):
    def make_context(utterance_list, start_t, end_t, max_seq_len):
        context = " ".join(utterance_list[start_t:end_t])

        if start_t > end_t:
            return ""

        if len(context.split()) + len(utterance_list[end_t].split()) > max_seq_len:
            context = make_context(utterance_list=utterance_list, start_t=start_t+1, end_t=end_t, max_seq_len=max_seq_len)
        else:
            return context
        
        return context


    f = open(data_file)
    data = json.load(f)
    f.close()

    tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased')

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance = list()

    for doc_id, content in data.items():
        single_utterances = list()
        utterance = list()
        content = content[0]
        max_doc_len = max(len(content), max_doc_len)

        for turn_data in content:
            single_utterances.append(turn_data["utterance"])

        for end_t in range(len(single_utterances)):
            context = make_context(utterance_list=single_utterances, start_t=0, end_t=end_t, max_seq_len=max_seq_len)

            utterance.append(tokenizer_(single_utterances[end_t], context, padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))
        
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        padding_sequence = tokenizer_('', padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt")

        padding_sequence_t = [padding_sequence for _ in range(max_doc_len - len(utterance_t))]

        utterance_t = utterance_t + padding_sequence_t
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        for _ in utterance_t:
            utterance_input_ids_t.append(_['input_ids'])
            utterance_attention_mask_t.append(_['attention_mask'])
            utterance_token_type_ids_t.append(_['token_type_ids'])

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)

    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return (out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device)), max_doc_len, max_seq_len

def tokenize_conversation(conversation, device, max_seq_len):
    tokenizer_ = BertTokenizer.from_pretrained('bert-base-cased')

    max_seq_len = max_seq_len
    max_doc_len = 0

    doc_utterance, speaker = [list() for _ in range(2)]
    for doc_id, content in conversation.items():
        utterance = list()
        content = content[0]
            
        for turn_data in content:
            utterance.append(tokenizer_(turn_data["utterance"], padding='max_length', max_length = max_seq_len, truncation=True, return_tensors="pt"))

            speaker.append(0 if turn_data["speaker"] == "A" else 1)
            
        doc_utterance.append(utterance)
        
    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = [list() for _ in range(3)]

    for utterance_t in doc_utterance:
        utterance_input_ids_t, utterance_attention_mask_t, utterance_token_type_ids_t = [list() for _ in range(3)]
        for _ in utterance_t:
            utterance_input_ids_t.append(_['input_ids'])
            utterance_attention_mask_t.append(_['attention_mask'])
            utterance_token_type_ids_t.append(_['token_type_ids'])

        utterance_input_ids_t = torch.vstack(utterance_input_ids_t)
        utterance_attention_mask_t = torch.vstack(utterance_attention_mask_t)
        utterance_token_type_ids_t = torch.vstack(utterance_token_type_ids_t)

        out_utterance_input_ids.append(utterance_input_ids_t)
        out_utterance_attention_mask.append(utterance_attention_mask_t)
        out_utterance_token_type_ids.append(utterance_token_type_ids_t)

    out_utterance_input_ids, out_utterance_attention_mask, out_utterance_token_type_ids = torch.stack(out_utterance_input_ids), torch.stack(out_utterance_attention_mask), torch.stack(out_utterance_token_type_ids)
    return out_utterance_input_ids.to(device), out_utterance_attention_mask.to(device), out_utterance_token_type_ids.to(device), torch.Tensor(speaker).to(device)