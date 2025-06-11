from .semantic_attention_head import SemanticAttentionHead

def make_head(head_config):

    if head_config.type == 'semantic_attention_head':
        head = SemanticAttentionHead(head_config)
    else:
        raise ValueError(f'Unknown head type: {head_config.type}')

    return head

