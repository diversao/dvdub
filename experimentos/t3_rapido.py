"""Laço de geração rápido para o T3 do Chatterbox multilíngue.

O laço original pede output_attentions=True em todas as camadas (força atenção "eager", lenta)
só para o verificador de alinhamento. Aqui: atenção SDPA, sem mapas de atenção, sem tqdm e sem
sincronizar com a CPU a cada passo; o fim da fala é checado a cada poucos passos.
Proteções no lugar do verificador: limite de tamanho pelo texto e corte de repetição.
"""
import torch
from transformers.generation.logits_process import (MinPLogitsWarper, RepetitionPenaltyLogitsProcessor,
                                                    TopPLogitsWarper)


def preparar(t3):
    """Cria o backend HF uma vez, com atenção SDPA."""
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend
    t3.tfmr.config._attn_implementation = "sdpa"
    t3.tfmr.config.output_attentions = False
    for camada in t3.tfmr.layers:
        if hasattr(camada.self_attn, "config"):
            camada.self_attn.config._attn_implementation = "sdpa"
    t3._rapido = T3HuggingfaceBackend(config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb,
                                       speech_head=t3.speech_head, alignment_stream_analyzer=None)


@torch.inference_mode()
def gerar(t3, t3_cond, text_tokens, n_letras, cfg_weight=0.5, temperature=0.8, top_p=1.0, min_p=0.05,
          repetition_penalty=2.0, ao_pedaco=None, pedaco=None):
    """Gera os tokens de fala. `ao_pedaco(tokens)` é chamado a cada `pedaco` tokens (para tocar enquanto gera)."""
    hp = t3.hp
    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
    inicial = hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=inicial,
                                        cfg_weight=cfg_weight)
    dev = embeds.device
    bos = torch.tensor([[hp.start_speech_token]], dtype=torch.long, device=dev)
    bos_e = t3.speech_emb(bos) + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_e = torch.cat([bos_e, bos_e])
    out = t3._rapido(inputs_embeds=torch.cat([embeds, bos_e], dim=1), past_key_values=None, use_cache=True,
                     output_attentions=False, output_hidden_states=True, return_dict=True)
    past = out.past_key_values

    rep = RepetitionPenaltyLogitsProcessor(penalty=float(repetition_penalty))
    minp = MinPLogitsWarper(min_p=min_p)
    topp = TopPLogitsWarper(top_p=top_p)
    gerados = bos.clone()
    # ~25 tokens por segundo de fala; teto generoso pelo tamanho do texto
    maximo = int(min(400, 60 + n_letras * 4))
    pred = []
    eos = hp.stop_speech_token
    for i in range(maximo):
        lg = out.logits[:, -1, :]
        cond, unc = lg[0:1], lg[1:2]
        lg = cond + cfg_weight * (cond - unc)
        lg = rep(gerados[:1], lg)
        if temperature != 1.0:
            lg = lg / temperature
        lg = minp(gerados[:1], lg)
        lg = topp(gerados[:1], lg)
        nxt = torch.multinomial(torch.softmax(lg, dim=-1), 1)
        pred.append(nxt)
        gerados = torch.cat([gerados, nxt], dim=1)
        if i % 4 == 3:  # checa o fim só de vez em quando (cada .item() espera a placa)
            ult = torch.cat(pred[-4:], dim=1)[0]
            if (ult == eos).any():
                break
            # repetição presa: o mesmo token 12 vezes seguidas
            if len(pred) >= 12 and bool((torch.cat(pred[-12:], dim=1) == nxt).all()):
                break
            if ao_pedaco and pedaco and len(pred) % pedaco == 0:
                ao_pedaco(torch.cat(pred, dim=1)[0])
        e = t3.speech_emb(nxt) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        out = t3._rapido(inputs_embeds=torch.cat([e, e]), past_key_values=past, use_cache=True,
                         output_attentions=False, output_hidden_states=True, return_dict=True)
        past = out.past_key_values
    toks = torch.cat(pred, dim=1)[0]
    fim = (toks == eos).nonzero()
    if len(fim):
        toks = toks[:fim[0, 0]]
    return toks
