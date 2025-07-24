import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp
from transformers import AutoModel, AutoTokenizer
from typing import Optional, Dict
import math
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
from utils import Tokenizer
from einops.layers.torch import Rearrange
import esm

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Namespace:
    def __init__(self, argvs):
        for k, v in argvs.items():
            setattr(self, k, v)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()

        self.layer = nn.ModuleList([
            TransformerEncoderLayer(Namespace({
                'encoder_embed_dim': dim,
                'encoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'encoder_normalize_before': True,
                'encoder_ffn_embed_dim': ff_dim,
            })) for i in range(num_layer)
        ])
        
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, encoder_padding_mask=None):
        for layer in self.layer:
            x = layer(x, encoder_padding_mask)
        x = self.layer_norm(x)
        return x

class MolecularFeatureExtractor(nn.Module):
    """整合GCN和ChemBERTa的分子特征提取器"""
    def __init__(self, drug_features, hidden_dim, dropout=0.1):
        super().__init__()

        self.gcn_conv1 = GCNConv(drug_features, drug_features * 2)
        self.gcn_conv2 = GCNConv(drug_features * 2, drug_features * 3)
        self.gcn_conv3 = GCNConv(drug_features * 3, hidden_dim)
        self.gcn_relu = nn.ReLU()
        
        chemberta_path = "/data/MultiDTAGen-main/ChemBERTa-77M-MTR"
        self.chemberta = AutoModel.from_pretrained(chemberta_path, local_files_only=True)
        self.chemberta_tokenizer = AutoTokenizer.from_pretrained(chemberta_path, local_files_only=True)
        
        self.smiles_proj = nn.Sequential(
            nn.Linear(384, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout))
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.gcn_relu(self.gcn_conv1(x, edge_index))
        x = self.gcn_relu(self.gcn_conv2(x, edge_index))
        gcn_features = self.gcn_relu(self.gcn_conv3(x, edge_index))
        gcn_pooled = gmp(gcn_features, batch)
        
        smiles_list = getattr(data, 'smiles', [])
        
        if not smiles_list:
            batch_size = batch.max().item() + 1
            smiles_list = ["C" for _ in range(batch_size)]
        
        inputs = self.chemberta_tokenizer(
            smiles_list, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=128).to(x.device)
        chemberta_out = self.chemberta(**inputs).last_hidden_state[:, 0, :]
        chemberta_features = self.smiles_proj(chemberta_out)
        
        combined = torch.cat([gcn_pooled, chemberta_features], dim=1)
        fused_features = self.fusion(combined)
        return fused_features, gcn_features

class Encoder(torch.nn.Module):
    def __init__(self, drug_features, hidden_dim, dropout, Final_dim):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.mol_extractor = MolecularFeatureExtractor(drug_features, hidden_dim, dropout)
        
        self.mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.var = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim))
        
        self.cond = nn.Linear(hidden_dim, hidden_dim)
        self.pp_seg_encoding = nn.Parameter(torch.randn(hidden_dim))

    def process_p(self, node_features, num_nodes, batch_size):
        d_node_features = pad_sequence(torch.split(node_features, num_nodes.tolist()), 
                                     batch_first=False, padding_value=-999)
        padded_sequence = d_node_features.new_ones((d_node_features.shape[0], 
                                                  d_node_features.shape[1], 
                                                  d_node_features.shape[2])) * -999
        padded_sequence[:d_node_features.shape[0], :, :] = d_node_features
        padding_mask = (d_node_features[:, :, 0].T == -999).bool()
        padded_sequence_with_encoding = d_node_features + self.pp_seg_encoding
        return padded_sequence_with_encoding, padding_mask

    def reparameterize(self, z_mean, logvar, batch, con, a):
        z_log_var = -torch.abs(logvar)
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp()) * 0.00005       
        epsilon = torch.randn_like(z_mean).to(z_mean.device)
        z_ = z_mean + torch.exp(z_log_var / 2) * epsilon
        
        con_embedding = self.cond(con).unsqueeze(0)
        a = a.unsqueeze(0)
        
        z_ = z_ + con_embedding + a
        
        return z_, kl_loss

    def forward(self, data, con):
        mol_features, mol_nodes = self.mol_extractor(data)
        
        d_sequence, mask = self.process_p(mol_nodes, data.c_size, data.batch)
        
        mu = self.mean(d_sequence)
        logvar = self.var(d_sequence)
        amvo, kl_loss = self.reparameterize(mu, logvar, data.batch, con, data.y.view(-1, 1))
        
        return d_sequence, amvo, mask, mol_features, kl_loss

class Decoder(nn.Module):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__()
        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
            })) for _ in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask=None, x_padding_mask=None, mem_padding_mask=None):
        for layer in self.layer:
            x = layer(x, mem,
                     self_attn_mask=x_mask, 
                     self_attn_padding_mask=x_padding_mask,
                     encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

    @torch.jit.export
    def forward_one(self, x, mem, incremental_state=None, mem_padding_mask=None):
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state, 
                     encoder_padding_mask=mem_padding_mask)[0]
        x = self.layer_norm(x)
        return x

class ProteinFeatureExtractor(nn.Module):
    """蛋白质特征提取器 - 整合CNN和ESM模型"""
    def __init__(self, vocab_size, hidden_dim=128):
        super().__init__()
        embed_dim = 128
        num_filters = 32
        kernel_size = 8
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        self.conv1 = nn.Conv1d(embed_dim, num_filters, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv1d(num_filters * 2, num_filters * 3, kernel_size, padding=kernel_size//2)
        
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        self.cnn_proj = nn.Sequential(
            nn.Linear(num_filters * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2))
        
        self.esm_model, self.esm_alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_batch_converter = self.esm_alphabet.get_batch_converter()
        
        for param in self.esm_model.parameters():
            param.requires_grad = False
        
        self.esm_proj = nn.Sequential(
            nn.Linear(1280, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2))
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.3))
        
        self.relu = nn.ReLU()
        self.hidden_dim = hidden_dim

    def get_esm_features(self, protein_seqs, device):
        data = [(f"protein{i}", seq) for i, seq in enumerate(protein_seqs)]
        
        _, _, batch_tokens = self.esm_batch_converter(data)
        batch_tokens = batch_tokens.to(device)
        
        with torch.no_grad():
            results = self.esm_model(batch_tokens, repr_layers=[33], return_contacts=False)
        
        token_representations = results["representations"][33]
        features = []
        for i, (_, seq) in enumerate(data):
            seq_representation = token_representations[i, 1:len(seq)+1].mean(0)
            features.append(seq_representation)
        
        return torch.stack(features)

    def forward(self, protein_indices, protein_seqs):
        device = protein_indices.device

        x_cnn = self.embedding(protein_indices)

        x_cnn = x_cnn.transpose(1, 2)

        x_cnn = self.relu(self.conv1(x_cnn))
        x_cnn = self.relu(self.conv2(x_cnn))
        x_cnn = self.relu(self.conv3(x_cnn))

        x_cnn = self.pool(x_cnn)

        x_cnn = x_cnn.view(x_cnn.size(0), -1)

        cnn_features = self.cnn_proj(x_cnn)

        esm_features = self.get_esm_features(protein_seqs, device)
        esm_features = self.esm_proj(esm_features)

        combined = torch.cat([cnn_features, esm_features], dim=1)
        fused_features = self.fusion(combined)
        
        return fused_features

class FC(torch.nn.Module):
    def __init__(self, output_dim, n_output, dropout):
        super(FC, self).__init__()
        self.FC_layers = nn.Sequential(
            nn.Linear(output_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_output)
        )

    def forward(self, Drug_Features, Protein_Features):
        Combined = torch.cat((Drug_Features, Protein_Features), 1)
        Pridection = self.FC_layers(Combined)
        return Pridection

class MultiDTAGen(torch.nn.Module):
    def __init__(self, tokenizer):
        super(MultiDTAGen, self).__init__()
        self.hidden_dim = 256
        self.max_len = 128
        self.node_feature = 94
        self.output_dim = 128
        self.ff_dim = 512
        self.heads = 4
        self.layers = 4
        self.encoder_dropout = 0.2
        self.dropout = 0.3
        self.protein_f = 25

        self.encoder = Encoder(
            drug_features=self.node_feature, 
            hidden_dim=self.hidden_dim, 
            dropout=self.encoder_dropout, 
            Final_dim=self.output_dim
        )
        
        self.decoder = Decoder(
            dim=self.hidden_dim, 
            ff_dim=self.ff_dim, 
            num_head=self.heads, 
            num_layer=self.layers
        )
        
        self.dencoder = TransformerEncoder(
            dim=self.hidden_dim, 
            ff_dim=self.ff_dim, 
            num_head=self.heads, 
            num_layer=self.layers
        )
        
        self.pos_encoding = PositionalEncoding(self.hidden_dim, max_len=138)

        self.protein_extractor = ProteinFeatureExtractor(
            vocab_size=self.protein_f + 1,
            hidden_dim=self.hidden_dim
        )

        self.fc = FC(output_dim=self.output_dim, n_output=1, dropout=self.dropout)

        self.zz_seg_encoding = nn.Parameter(torch.randn(self.hidden_dim))

        vocab_size = len(tokenizer)
        self.word_pred = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.PReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, vocab_size)
        )
        torch.nn.init.zeros_(self.word_pred[3].bias)

        self.vocab_size = vocab_size
        self.sos_value = tokenizer.s2i['<sos>']
        self.eos_value = tokenizer.s2i['<eos>']
        self.pad_value = tokenizer.s2i['<pad>']
        self.word_embed = nn.Embedding(vocab_size, self.hidden_dim)
        self.unk_index = Tokenizer.SPECIAL_TOKENS.index('<unk>')

        self.expand = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            Rearrange('batch_size h -> 1 batch_size h')
        )
        
        self.drug_projection = nn.Linear(self.hidden_dim, self.output_dim)
        self.protein_projection = nn.Linear(self.hidden_dim, self.output_dim)

    def expand_then_fusing(self, z, pp_mask, vvs):
        zz = z
        zzs = zz + self.zz_seg_encoding

        full_mask = zz.new_zeros(zz.shape[1], zz.shape[0])
        full_mask = torch.cat((pp_mask, full_mask), dim=1)

        zzz = torch.cat((vvs, zzs), dim=0)

        zzz = self.dencoder(zzz, full_mask)
        
        return zzz, full_mask

    def sample(self, batch_size, device):
        z = torch.randn(1, self.hidden_dim).to(device)
        return z

    def forward(self, data):

        Protein_vector = self.protein_extractor(data.target, data.protein_seqs)

        con = Protein_vector

        vss, AMVO, mask, PMVO, kl_loss = self.encoder(data, con)

        PMVO_projected = self.drug_projection(PMVO)
        Protein_vector_projected = self.protein_projection(Protein_vector)

        zzz, encoder_mask = self.expand_then_fusing(AMVO, mask, vss)

        targets = data.target_seq
        _, target_length = targets.shape
        target_mask = torch.triu(torch.ones(target_length, target_length, dtype=torch.bool), diagonal=1).to(targets.device)
        target_embed = self.word_embed(targets)
        target_embed = self.pos_encoding(target_embed.permute(1, 0, 2).contiguous())

        output = self.decoder(target_embed, zzz, x_mask=target_mask, mem_padding_mask=encoder_mask).permute(1, 0, 2).contiguous()
        prediction_scores = self.word_pred(output)

        shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
        targets = targets[:, 1:].contiguous()
        batch_size, sequence_length, vocab_size = shifted_prediction_scores.size()
        shifted_prediction_scores = shifted_prediction_scores.view(-1, vocab_size)
        targets = targets.view(-1)

        Pridection = self.fc(PMVO_projected, Protein_vector_projected)
        lm_loss = F.cross_entropy(shifted_prediction_scores, targets, ignore_index=self.pad_value)
        
        return Pridection, prediction_scores, lm_loss, kl_loss

    def _generate(self, zzz, encoder_mask, random_sample, return_score=False):

        batch_size = zzz.shape[1]
        device = zzz.device

        token = torch.full((batch_size, self.max_len), self.pad_value, dtype=torch.long, device=device)
        token[:, 0] = self.sos_value

        text_pos = self.pos_encoding.pe

        text_embed = self.word_embed(token[:, 0])
        text_embed = text_embed + text_pos[0]
        text_embed = text_embed.unsqueeze(0)

        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[torch.Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {}),
        )

        if return_score:
            scores = []

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for t in range(1, self.max_len):
            one = self.decoder.forward_one(text_embed, zzz, incremental_state, mem_padding_mask=encoder_mask)
            one = one.squeeze(0)
            l = self.word_pred(one)

            if return_score:
                scores.append(l)

            if random_sample:
                k = torch.multinomial(torch.softmax(l, 1), 1).squeeze(1)
            else:
                k = torch.argmax(l, -1)

            token[:, t] = k

            finished |= k == self.eos_value
            if finished.all():
                break

            text_embed = self.word_embed(k)
            text_embed = text_embed + text_pos[t]
            text_embed = text_embed.unsqueeze(0)

        predict = token[:, 1:]

        if return_score:
            return predict, torch.stack(scores, dim=1)
        return predict

    def generate(self, data, random_sample=False, return_z=False):
        Protein_vector = self.protein_extractor(data.target, data.protein_seqs)
        con = Protein_vector

        vss, AMVO, mask, PMVO, kl_loss = self.encoder(data, con)

        z = self.sample(data.batch, device=vss.device)

        zzz, encoder_mask = self.expand_then_fusing(AMVO, mask, vss)

        predict = self._generate(zzz, encoder_mask, random_sample=random_sample, return_score=False)

        if return_z:
            return predict, z.detach().cpu().numpy()
        return predict