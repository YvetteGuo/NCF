import mxnet as mx

def neumf_model(factor_size, model_layers, max_user, max_item, dense):
    # input
    user = mx.sym.Variable('user')
    item = mx.sym.Variable('item')
    score = mx.sym.Variable('score')
    stype = 'default' if dense else 'row_sparse'
    sparse_grad = not dense
    user_gmf_weight = mx.sym.Variable('user_gmf_weight', stype=stype)
    item_gmf_weight = mx.sym.Variable('item_gmf_weight', stype=stype)
    user_mlp_weight = mx.sym.Variable('user_mlp_weight', stype=stype)
    item_mlp_weight = mx.sym.Variable('item_mlp_weight', stype=stype)
    '''GMF part'''
    # user feature lookup
    user_gmf = mx.sym.Embedding(data=user, weight=user_gmf_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=factor_size)
    # item feature lookup
    item_gmf = mx.sym.Embedding(data=item, weight=item_gmf_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=factor_size)
    # elementwise product of user and item embeddings
    gmf_vector = user_gmf * item_gmf
    # FC
    gmf_vector = mx.sym.FullyConnected(data=gmf_vector, num_hidden=1)                    
    '''MLP part'''
    # user feature lookup
    user_mlp = mx.sym.Embedding(data=user, weight=user_mlp_weight, sparse_grad=sparse_grad,
                            input_dim=max_user, output_dim=model_layers[0]//2)
    # item feature lookup
    item_mlp = mx.sym.Embedding(data=item, weight=item_mlp_weight, sparse_grad=sparse_grad,
                            input_dim=max_item, output_dim=model_layers[0]//2)               
    # Concatenation of two latent features
    mlp_vector=mx.sym.concat(user_mlp, item_mlp, dim=1)
    num_layer = len(model_layers)  
    for layer in range(1,num_layer):
        mlp_vector = mx.sym.FullyConnected(data=mlp_vector, num_hidden=model_layers[layer])
        mlp_vector=mx.sym.Activation(data=mlp_vector, act_type='relu')

    '''Concatenate MF and MLP parts'''
    predict_vector=mx.sym.concat(gmf_vector, mlp_vector, dim=1)

    # Final prediction layer  
    pred = mx.sym.FullyConnected(data=predict_vector, num_hidden=1)
    pred = mx.sym.Flatten(data=pred)  
    # loss layer
    pred = mx.sym.LogisticRegressionOutput(data=pred, label=score)
    return pred