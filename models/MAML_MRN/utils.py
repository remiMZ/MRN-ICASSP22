import torch

def get_accuracy(logits, targets):
    """Compute the accuracy (after adaptation) of MAML on the test/query points

    Parameters
    ----------
    logits : `torch.FloatTensor` instance
        Outputs/logits of the model on the query points. This tensor has shape
        `(num_examples, num_classes)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(num_examples,)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points
    """
    _, predictions = torch.max(logits, dim=-1)
    return torch.mean(predictions.eq(targets).float())


def fix_nn(model, phi):
    def k_param_fn(tmp_model, name=None):
        if len(tmp_model._modules)!=0:
            for(k,v) in tmp_model._modules.items():
                if name is None:
                    k_param_fn(v, name=str(k))
                else:
                    k_param_fn(v, name=str(name+'.'+k))
        else:
            for (k,v) in tmp_model._parameters.items():
                if not isinstance(v,torch.Tensor):
                    continue
                tmp_model._parameters[k] = phi[str(name + '.' + k)]

    k_param_fn(model)
    return model

